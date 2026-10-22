"""Validator-paper experiments (RQ1-RQ3) on OULAD, offline consumers.

Task: early warning at day 28, y = 1 for Fail or Withdrawn (see build_oulad.py). Consumers:
  tree  - XGBoost on demographics, registration and early engagement (all 19 features)
  lr    - logistic regression on the 9 early-engagement features only
The LLM agent is added later (it needs an Azure OpenAI deployment).

Split by presentation: consumers fit on 2013B+2013J, profiles use 2014B (the labelled
probe set), test batches come from 2014J. Reuses the batch, scoring and evaluation code
of run_experiments.py so both datasets are measured the same way.

Run:
    python3 run_oulad.py --data data/oulad/oulad_day28.parquet --out results/oulad_b1000_rows500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import agentdq as dq
from agentdq.profile import auroc, balanced_sample

sys.path.insert(0, str(Path(__file__).parent))
import run_experiments as rx  # noqa: E402
from build_oulad import CATEGORICAL, ENGAGEMENT, LABEL, NUMERIC  # noqa: E402

# OULAD has no money or free text, so currency_unit_drift has nothing to touch
FAMILIES = [f for f in dq.CANONICAL if f != "currency_unit_drift"]
rx.LABEL, rx.FAMILIES = LABEL, FAMILIES


class Coerced:
    """Wraps a fitted pipeline with the numeric coercion a production loader would apply."""

    def __init__(self, pipe, num_cols, cat_cols):
        self.pipe, self.num_cols, self.cat_cols = pipe, num_cols, cat_cols

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in self.num_cols:
            if c in X:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in self.cat_cols:
            if c in X:
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return self.pipe.predict_proba(X)


def build_consumers(train: pd.DataFrame, seed: int = 0):
    feats = CATEGORICAL + NUMERIC
    pre = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ("num", "passthrough", NUMERIC)])
    tree_pipe = make_pipeline(pre, XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                 subsample=0.9, colsample_bytree=0.9,
                                                 random_state=seed, n_jobs=4))
    tr = train[feats].copy()
    for c in CATEGORICAL:
        tr[c] = tr[c].astype(object).where(tr[c].notna(), np.nan)
    tree_pipe.fit(tr, train[LABEL])
    tree = dq.SklearnConsumer(Coerced(tree_pipe, NUMERIC, CATEGORICAL), feats, name="tree")

    lr_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000))
    lr_pipe.fit(train[ENGAGEMENT], train[LABEL])
    lr = dq.SklearnConsumer(Coerced(lr_pipe, ENGAGEMENT, []), ENGAGEMENT, name="lr")
    return [tree, lr], feats


def oulad_roles() -> dq.ColumnRoles:
    return dq.ColumnRoles.of(protected={LABEL, "row_id", "split"},
                             day_deltas={"date_registration", "mean_days_early"},
                             taxonomy={"highest_education", "region", "imd_band", "age_band"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_parquet(args.data)
    train = df[df.split == "train"]
    consumers, feats = build_consumers(train)
    cols = ["row_id", LABEL] + feats
    val = df[df.split == "val"][cols].reset_index(drop=True)
    test = df[df.split == "test"][cols].reset_index(drop=True)
    roles = oulad_roles()
    reference = balanced_sample(val, LABEL, 1000, 999)
    meta = {"dataset": "OULAD day 28", "label": LABEL, "families": FAMILIES,
            "train": len(train), "val": len(val), "test": len(test),
            "prevalence_test": float(test[LABEL].mean()),
            "batches": args.batches, "rows": args.rows}
    for c in consumers:
        meta[f"test_auroc_{c.name}"] = auroc(test[LABEL].to_numpy(), c.score(test))
    print(f"[{time.time()-t0:.0f}s] consumers:", {k: round(v, 3) for k, v in meta.items()
                                                  if k.startswith("test_auroc")})

    profiles = {}
    for c in consumers:
        p = dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES)
        p.to_json(args.out / f"profile_{c.name}.json")
        profiles[c.name] = p
        meta[f"clean_auroc_probe_{c.name}"] = p.clean_auroc
    print(f"[{time.time()-t0:.0f}s] profiles done")

    batches = rx.make_batches(test, roles, args.batches, args.rows, (0.0, 0.4), args.seed)
    dmg = rx.realised_damage(consumers, batches)
    sc = rx.scores_for(list(profiles.values()), batches, reference, roles)
    rq1 = dmg.merge(sc, on="batch")
    rq1.to_csv(args.out / "rq1_batches.csv", index=False)
    names = [c.name for c in consumers]
    rx.evaluate(rq1, names).to_csv(args.out / "rq1_metrics.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ1 done")

    rq2 = []
    for n in (50, 100, 200, 500):
        profs = [dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES, n_rows=n)
                 for c in consumers]
        e = rx.evaluate(dmg.merge(rx.scores_for(profs, batches, reference, roles), on="batch"),
                        names, n_boot=500)
        e = e[e.score == "agentdq (per-consumer)"].assign(probe_rows=n)
        for c, p in zip(consumers, profs):
            touched = sum(fr.touches_channels for fr in p.families.values())
            e.loc[e.consumer == c.name, "scoring_calls_rows"] = 3 * n * (1 + 2 * touched)
        rq2.append(e)
    pd.concat(rq2).to_csv(args.out / "rq2_probe_size.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ2 done")

    low = [dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES, severities=(0.1,))
           for c in consumers]
    hi_batches = rx.make_batches(test, roles, args.batches, args.rows, (0.3, 0.4), args.seed + 1,
                                 k_choices=(1, 2, 3))
    hi_dmg = rx.realised_damage(consumers, hi_batches)
    rq3 = []
    for label, profs in (("profiled at 0.1 only", low),
                         ("profiled at 0.1 and 0.3", list(profiles.values()))):
        e = rx.evaluate(hi_dmg.merge(rx.scores_for(profs, hi_batches, reference, roles),
                                     on="batch"), names, n_boot=500)
        rq3.append(e[e.score == "agentdq (per-consumer)"].assign(profile=label))
    pd.concat(rq3).to_csv(args.out / "rq3_extrapolation.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ3 done")

    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
