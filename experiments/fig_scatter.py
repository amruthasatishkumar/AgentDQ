"""Figure: per-consumer risk vs realised AUROC damage, with the failed-check count for contrast."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).parent / "results"
PANELS = [("Olist, text", "kaggle_v2_b1000_rows500", "text"),
          ("OULAD, tree", "oulad_v2_b1000_rows500", "tree"),
          ("OULAD, agent", "oulad_agent_v2", "agent")]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharey="col")
for j, (title, folder, c) in enumerate(PANELS):
    df = pd.read_csv(R / folder / "rq1_batches.csv")
    d = df[f"damage_{c}"]
    for i, (col, lab) in enumerate([(f"risk_{c}", "our risk"), ("baseline_failed_checks", "failed checks")]):
        ax = axes[i, j]
        ax.scatter(df[col], d, s=4, alpha=0.4, color="#2a6f97" if i == 0 else "#999999", lw=0)
        ax.set_title(f"{title}: $\\rho$={spearmanr(df[col], d)[0]:.2f}", fontsize=8)
        ax.set_xlabel(lab, fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("AUROC damage", fontsize=7)
fig.tight_layout(pad=0.4)
out = Path(__file__).parent / "fig_scatter.pdf"
fig.savefig(out)
print(out)
