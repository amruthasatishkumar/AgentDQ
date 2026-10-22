"""Triage view of RQ1: of the 10% most damaging batches per consumer, how many does
each score put in its own top 10%? Ties (failed-check counts) are broken at random and
averaged over 200 draws. Also: share of batches failing >=1 check that cost under 0.01.

Run: python3 triage.py  (reads the result folders next to this file)
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).parent / "results"
RUNS = {"Olist": ("kaggle_v2_b1000_rows500", ["tree", "text"]),
        "OULAD": ("oulad_v2_b1000_rows500", ["tree", "lr"]),
        "OULAD-agent": ("oulad_agent_v2", ["tree", "lr", "agent"])}
SCORES = {"ours": "risk_{c}", "checks": "baseline_failed_checks", "sev": "baseline_severity_sum"}


def top_recall(score, damage, q=0.1, draws=200, seed=0):
    k = max(1, int(round(q * len(damage))))
    truth = set(np.argsort(-damage, kind="stable")[:k])
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(draws):
        order = np.lexsort((rng.random(len(score)), -score))
        hits.append(len(truth & set(order[:k])) / k)
    return float(np.mean(hits))


def main():
    rows = []
    for data, (folder, consumers) in RUNS.items():
        df = pd.read_csv(R / folder / "rq1_batches.csv")
        flagged = df.baseline_failed_checks > 0
        for c in consumers:
            d = df[f"damage_{c}"].to_numpy()
            r = {"data": data, "consumer": c, "n": len(df),
                 "flagged_harmless": round(float((d[flagged] < 0.01).mean()), 2)}
            for name, col in SCORES.items():
                r[name] = round(top_recall(df[col.format(c=c)].to_numpy(), d), 2)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(R / "triage_top10.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
