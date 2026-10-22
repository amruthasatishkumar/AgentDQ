"""Validator-paper experiments (RQ1-RQ3) on Olist, offline consumers.

Task: negative review (paper T2, label y_t3). Consumers:
  tree  - XGBoost on the 21 purchase-time tabular features (the ICDM tree)
  text  - TF-IDF + logistic regression on the two review-text columns only
The LLM agent is added separately (run_oulad_agent.py; it needs an Azure OpenAI deployment).

Splits follow the frozen ICDM temporal split: consumers fit on `train`, profiles
use `val` (the labelled probe set), test batches come from `test`.

Run (from the paper repo, after building data/silver/order_features.parquet):
    PYTHONPATH=<agentdq>/src python3 run_experiments.py --repo <paper repo> --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import agentdq as dq
from agentdq.profile import auroc, balanced_sample
from agentdq.validate import consumer_risk

LABEL = "y_t3"
FAMILIES = list(dq.CANONICAL)
MONETARY = ["total_price", "total_freight", "max_price", "mean_price", "total_payment_value"]
TEXT = ["review_comment_title", "review_comment_message"]


# ---------------- consumers ----------------
class TabularTree:
    """The ICDM XGBoost pipeline, with the same numeric coercion it uses in production."""

    def __init__(self, pipe, feats, num_cols):
        self.pipe, self.feats, self.num_cols = pipe, feats, num_cols

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in X.columns.difference(self.num_cols):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return self.pipe.predict_proba(X)


def _join_text(X):
    X = pd.DataFrame(X)
    return X.fillna("").astype(str).agg(" ".join, axis=1).tolist()


def build_consumers(repo: Path, train: pd.DataFrame):
    sys.path.insert(0, str(repo))
    from consumers.features import T2_FEATURE_COLS
    from consumers.ml_pipeline import CATEGORICAL_COLS
    from eval.sweep import _train_xgb

    feats = list(T2_FEATURE_COLS)
    num_cols = [c for c in feats if c not in CATEGORICAL_COLS]
    tr = train.copy()
    tr["split"] = "train"
    pipe = _train_xgb(tr, LABEL, feats)
    tree = dq.SklearnConsumer(TabularTree(pipe, feats, num_cols), feats, name="tree")

    text_model = make_pipeline(FunctionTransformer(_join_text, validate=False),
                               TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5,
                                               sublinear_tf=True),
                               LogisticRegression(max_iter=2000, C=1.0))
    fit = train[train[LABEL].notna()]
    text_model.fit(fit[TEXT], fit[LABEL].astype(int))
    text = dq.SklearnConsumer(text_model, TEXT, name="text")
    return [tree, text], feats


def olist_roles(df: pd.DataFrame) -> dq.ColumnRoles:
    protected = {LABEL, "order_id"}
    return dq.ColumnRoles.of(protected=protected, monetary=MONETARY, hour_of_day={"purchase_hour"},
                             day_deltas={"days_to_estimated"}, text=TEXT,
                             taxonomy={"dominant_category"})


# ---------------- batches ----------------
def make_batches(test: pd.DataFrame, roles, n_batches: int, n_rows: int, sev_range, seed: int,
                 k_choices=(0, 1, 2, 3)):
    """Each batch: a balanced test sample plus 0-3 random families at random severities."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_batches):
        bseed = int(rng.integers(1_000_000))
        clean = balanced_sample(test, LABEL, n_rows, bseed)
        k = int(rng.choice(k_choices))
        fams = list(rng.choice(FAMILIES, size=k, replace=False)) if k else []
        sevs = [float(rng.uniform(*sev_range)) for _ in fams]
        corrupted = clean
        for i, (f, s) in enumerate(zip(fams, sevs)):
            corrupted = dq.apply(corrupted, f, s, bseed + i, roles)
        out.append(dict(batch=b, seed=bseed, families=fams, severities=sevs,
                        clean=clean, corrupted=corrupted))
    return out


def realised_damage(consumers, batches):
    """Paired clean-minus-corrupted AUROC per consumer and batch (positive = harm)."""
    rows = []
    for bt in batches:
        y = bt["clean"][LABEL].to_numpy()
        yc = bt["corrupted"][LABEL].to_numpy()
        row = {"batch": bt["batch"]}
        for c in consumers:
            row[f"clean_{c.name}"] = auroc(y, c.score(bt["clean"]))
            row[f"damage_{c.name}"] = row[f"clean_{c.name}"] - auroc(yc, c.score(bt["corrupted"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- scoring ----------------
def scores_for(profiles, batches, reference, roles, check_threshold=0.05):
    """Per-batch predicted risk per consumer, plus consumer-agnostic baselines."""
    rows = []
    for bt in batches:
        det = dq.detect_all(bt["corrupted"], reference, roles)
        true = {f: 0.0 for f in FAMILIES}
        for f, s in zip(bt["families"], bt["severities"]):
            true[f] = s
        row = {"batch": bt["batch"], "n_families": len(bt["families"]),
               "families": ",".join(bt["families"]),
               "baseline_failed_checks": sum(v >= check_threshold for v in det.values()),
               "baseline_severity_sum": float(sum(det.values()))}
        for f in FAMILIES:
            row[f"det_{f}"] = det.get(f, 0.0)
            row[f"true_{f}"] = true[f]
        for p in profiles:
            # schema drift is re-detected per consumer from the batch's columns (exact, so
            # the oracle uses the same value)
            row[f"risk_{p.consumer}"] = consumer_risk(p, det, 0.02, 0.05, bt["corrupted"]).risk
            row[f"oracle_{p.consumer}"] = consumer_risk(p, true, 0.02, 0.05, bt["corrupted"]).risk
        rows.append(row)
    return pd.DataFrame(rows)


def prec_at_k(score, damage, k=20):
    top_pred = set(np.argsort(-np.asarray(score), kind="stable")[:k])
    top_true = set(np.argsort(-np.asarray(damage), kind="stable")[:k])
    return len(top_pred & top_true) / k


def evaluate(df: pd.DataFrame, consumers, n_boot=2000, seed=0):
    """Spearman and P@20 between each score and each consumer's damage, with batch-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), size=(n_boot, len(df)))
    rows = []
    for c in consumers:
        dmg = df[f"damage_{c}"].to_numpy()
        cands = {"agentdq (per-consumer)": df[f"risk_{c}"],
                 "baseline: failed checks": df["baseline_failed_checks"],
                 "baseline: severity sum": df["baseline_severity_sum"],
                 "oracle severities x profile": df[f"oracle_{c}"]}
        for other in consumers:
            if other != c:
                cands[f"wrong profile ({other})"] = df[f"risk_{other}"]
        for name, sc in cands.items():
            sc = sc.to_numpy(dtype=float)
            rho = spearmanr(sc, dmg).statistic
            boots = [spearmanr(sc[i], dmg[i]).statistic for i in idx]
            p20 = prec_at_k(sc, dmg)
            p_boots = [prec_at_k(sc[i], dmg[i]) for i in idx[:500]]
            ours = df[f"risk_{c}"].to_numpy(dtype=float)
            gain = [spearmanr(ours[i], dmg[i]).statistic - spearmanr(sc[i], dmg[i]).statistic
                    for i in idx[:1000]]
            rows.append(dict(consumer=c, score=name, spearman=rho,
                             gain_vs_this_lo=np.nanpercentile(gain, 2.5),
                             gain_vs_this_hi=np.nanpercentile(gain, 97.5),
                             spearman_lo=np.nanpercentile(boots, 2.5),
                             spearman_hi=np.nanpercentile(boots, 97.5),
                             p_at_20=p20, p20_lo=np.percentile(p_boots, 2.5),
                             p20_hi=np.percentile(p_boots, 97.5)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    silver = pd.read_parquet(args.repo / "data" / "silver" / "order_features.parquet")
    silver = silver[silver[LABEL].notna()].copy()
    silver[LABEL] = silver[LABEL].astype(int)
    train = silver[silver.split == "train"]
    consumers, feats = build_consumers(args.repo, train)
    cols = ["order_id", LABEL] + feats + TEXT
    val = silver[silver.split == "val"][cols].reset_index(drop=True)
    test = silver[silver.split == "test"][cols].reset_index(drop=True)
    roles = olist_roles(val)
    reference = balanced_sample(val, LABEL, 1000, 999)
    meta = {"label": LABEL, "train": len(train), "val": len(val), "test": len(test),
            "prevalence_test": float(test[LABEL].mean())}

    # Stage 1: default profiles (200 probe rows, severities 0.1 and 0.3, 3 seeds)
    profiles = {}
    for c in consumers:
        p = dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES)
        p.to_json(args.out / f"profile_{c.name}.json")
        profiles[c.name] = p
        meta[f"clean_auroc_probe_{c.name}"] = p.clean_auroc
    print(f"[{time.time()-t0:.0f}s] profiles done")

    # RQ1: mixed batches, severities in [0, 0.4]
    batches = make_batches(test, roles, args.batches, args.rows, (0.0, 0.4), args.seed)
    dmg = realised_damage(consumers, batches)
    sc = scores_for(list(profiles.values()), batches, reference, roles)
    rq1 = dmg.merge(sc, on="batch")
    rq1.to_csv(args.out / "rq1_batches.csv", index=False)
    names = [c.name for c in consumers]
    ev = evaluate(rq1, names)
    ev.to_csv(args.out / "rq1_metrics.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ1 done")

    # RQ2: probe size
    rq2 = []
    for n in (50, 100, 200, 500):
        profs = [dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES, n_rows=n)
                 for c in consumers]
        e = evaluate(dmg.merge(scores_for(profs, batches, reference, roles), on="batch"), names,
                     n_boot=500)
        e = e[e.score == "agentdq (per-consumer)"].assign(probe_rows=n)
        # scoring calls a profile needs: per seed, 1 clean + one per (family, severity) it touches
        for c, p in zip(consumers, profs):
            touched = sum(fr.touches_channels for fr in p.families.values())
            e.loc[e.consumer == c.name, "scoring_calls_rows"] = 3 * n * (1 + 2 * touched)
        rq2.append(e)
    pd.concat(rq2).to_csv(args.out / "rq2_probe_size.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ2 done")

    # RQ3: profile at low severity only, test on high-severity batches
    low = [dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES, severities=(0.1,))
           for c in consumers]
    hi_batches = make_batches(test, roles, args.batches, args.rows, (0.3, 0.4), args.seed + 1,
                              k_choices=(1, 2, 3))
    hi_dmg = realised_damage(consumers, hi_batches)
    rq3 = []
    for label, profs in (("profiled at 0.1 only", low), ("profiled at 0.1 and 0.3", list(profiles.values()))):
        e = evaluate(hi_dmg.merge(scores_for(profs, hi_batches, reference, roles), on="batch"), names,
                     n_boot=500)
        rq3.append(e[e.score == "agentdq (per-consumer)"].assign(profile=label))
    pd.concat(rq3).to_csv(args.out / "rq3_extrapolation.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] RQ3 done")

    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
