"""Stage 1: sensitivity profiles.

For one consumer, measure how much each corruption family hurts its AUROC, using the
ICDM paired-clean protocol: score the same balanced sample clean and corrupted, and
take the difference. Done once per consumer, offline; validate() reuses the result.

Sign convention: `drop` = AUROC(clean) - AUROC(corrupted). Positive means the defect hurts.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from agentdq.consumers import Consumer
from agentdq.corruption import CANONICAL, ColumnRoles, apply


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (ties averaged). Returns nan if y has only one class."""
    y = np.asarray(y).astype(int)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(s, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@dataclass
class FamilyResult:
    family: str
    touches_channels: bool
    severities: list[float]
    mean_drop: list[float]            # one per severity, averaged over seeds
    ci_low: list[float]               # bootstrap 95% CI of the mean, per severity
    ci_high: list[float]
    slope: float                      # drop per unit severity, least squares through the origin


@dataclass
class SensitivityProfile:
    consumer: str
    channels: list[str]
    clean_auroc: float                # mean over seeds
    families: dict[str, FamilyResult] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SensitivityProfile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["families"] = {k: FamilyResult(**v) for k, v in d["families"].items()}
        return cls(**d)


# ---------------- Step 2a ----------------
def balanced_sample(df: pd.DataFrame, label_col: str, n_rows: int, seed: int) -> pd.DataFrame:
    """n_rows rows, half label 1 and half label 0, drawn without replacement with this seed.

    Index reset to 0..n_rows-1. Raise ValueError if either class has fewer than n_rows // 2 rows.
    """
    half = n_rows // 2
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    if len(pos) < half or len(neg) < half:
        raise ValueError(f"need at least {half} rows of each class, "
                         f"got {len(pos)} positive and {len(neg)} negative")
    rng = np.random.default_rng(seed)
    pos = pos.iloc[rng.choice(len(pos), half, replace=False)]
    neg = neg.iloc[rng.choice(len(neg), half, replace=False)]
    both = pd.concat([pos, neg])
    return both.iloc[rng.permutation(len(both))].reset_index(drop=True)


# ---------------- Step 2b ----------------
def touches_channels(clean: pd.DataFrame, corrupted: pd.DataFrame, channels: list[str]) -> bool:
    """True if the corruption changed anything the consumer reads.

    That is: a channel column is missing or renamed in `corrupted`, the row count changed,
    or any channel column's values differ (treat NaN == NaN as equal).
    """
    if len(clean) != len(corrupted):
        return True
    for c in channels:
        if c not in corrupted.columns or c not in clean.columns:
            return True
        a = clean[c].reset_index(drop=True)
        b = corrupted[c].reset_index(drop=True)
        if not a.equals(b):  # equals() treats NaN == NaN
            return True
    return False


# ---------------- Step 2c and 2d ----------------
def profile_sensitivity(consumer: Consumer, probe_df: pd.DataFrame, label_col: str,
                        roles: ColumnRoles | None = None, families="default",
                        severities=(0.1, 0.3), n_rows: int = 200, seeds: int = 3,
                        n_boot: int = 1000) -> SensitivityProfile:
    """Measure the consumer's AUROC drop per corruption family and severity.

    For each seed: draw balanced_sample(probe_df, label_col, n_rows, seed), score it clean
    once (cache it), then for each family and severity apply(sample, family, sev, seed, roles)
    and score the corrupted copy. If touches_channels() is False, record drop 0.0 and do NOT
    call consumer.score (this saves LLM cost).

    Corruptions that add rows (duplicate_rows) also add labels: take labels from the
    corrupted frame's label_col, not from the clean sample.

    families="default" means CANONICAL (the 6 ICDM families). The label column must be in
    roles.protected so it is never corrupted; add it if the caller forgot.
    """
    fams = list(CANONICAL if families == "default" else families)
    sevs = [float(s) for s in severities]
    roles = roles or ColumnRoles()
    if label_col not in roles.protected:
        roles = replace(roles, protected=roles.protected | {label_col})

    clean_scores = []
    drops = {f: [[] for _ in sevs] for f in fams}  # drops[family][severity_idx] -> one per seed
    touched = {f: False for f in fams}

    for seed in range(seeds):
        sample = balanced_sample(probe_df, label_col, n_rows, seed)
        clean_au = auroc(sample[label_col].to_numpy(), consumer.score(sample))
        clean_scores.append(clean_au)
        for fam in fams:
            for i, sev in enumerate(sevs):
                corrupted = apply(sample, fam, sev, seed, roles)
                if touches_channels(sample, corrupted, consumer.channels):
                    touched[fam] = True
                    bad = auroc(corrupted[label_col].to_numpy(), consumer.score(corrupted))
                    drops[fam][i].append(clean_au - bad)
                else:
                    drops[fam][i].append(0.0)

    rng = np.random.default_rng(0)
    results = {}
    for fam in fams:
        means, lows, highs = [], [], []
        for per_seed in drops[fam]:
            x = np.asarray(per_seed, dtype=float)
            m = float(x.mean())
            boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            means.append(m)
            lows.append(float(min(lo, m)))
            highs.append(float(max(hi, m)))
        s = np.asarray(sevs)
        slope = float((s * np.asarray(means)).sum() / (s ** 2).sum()) if len(s) else 0.0
        results[fam] = FamilyResult(fam, touched[fam], sevs, means, lows, highs, slope)

    return SensitivityProfile(consumer.name, list(consumer.channels),
                              float(np.mean(clean_scores)), results)
