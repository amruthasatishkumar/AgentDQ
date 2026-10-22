"""Re-score the OULAD agent experiment with channel-scoped schema drift, without new LLM calls.

The agent reads all 19 features, so the channel-scoped schema-drift probe renames exactly
the columns the original probe renamed (same seed, same column order): its measured drops
are reused and only the recorded severities (share of its columns renamed) change. The
tree and engagement model are re-profiled offline. Realised damage is unchanged and read
from the original run.

Run: python3 rescore_oulad_agent.py --data data/oulad/oulad_day28.parquet \
         --old results/oulad_agent --out results/oulad_agent_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import agentdq as dq
from agentdq.profile import FamilyResult, SensitivityProfile, balanced_sample

sys.path.insert(0, str(Path(__file__).parent))
import run_experiments as rx  # noqa: E402
from build_oulad import CATEGORICAL, LABEL, NUMERIC  # noqa: E402
from run_oulad import FAMILIES, build_consumers, oulad_roles  # noqa: E402

FEATS = CATEGORICAL + NUMERIC


def agent_profile(path: Path) -> SensitivityProfile:
    d = json.loads(path.read_text())
    fams = {f: FamilyResult(**r) for f, r in d["families"].items()}
    sd = fams["schema_drift"]
    n = len(d["channels"])
    sevs = [max(1, int(round(s * n))) / n for s in sd.severities]
    s = np.asarray(sevs)
    slope = float((s * np.asarray(sd.mean_drop)).sum() / (s ** 2).sum())
    fams["schema_drift"] = FamilyResult(sd.family, sd.touches_channels, sevs, sd.mean_drop,
                                        sd.ci_low, sd.ci_high, slope)
    return SensitivityProfile(d["consumer"], d["channels"], d["clean_auroc"], fams)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.old / "meta.json").read_text())

    df = pd.read_parquet(args.data)
    train = df[df.split == "train"]
    cols = ["row_id", LABEL] + FEATS
    val = df[df.split == "val"][cols].reset_index(drop=True)
    test = df[df.split == "test"][cols].reset_index(drop=True)
    roles = oulad_roles()
    offline, _ = build_consumers(train)
    reference = balanced_sample(val, LABEL, 1000, 999)
    profiles = [dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES,
                                       n_rows=meta["probe_rows"]) for c in offline]
    profiles.append(agent_profile(args.old / "profile_agent.json"))
    for p in profiles:
        p.to_json(args.out / f"profile_{p.consumer}.json")

    batches = rx.make_batches(test, roles, meta["batches"], meta["rows"], (0.0, 0.4), 2026)
    old = pd.read_csv(args.old / "rq1_batches.csv")
    dmg = old[["batch"] + [c for c in old if c.startswith(("clean_", "damage_"))]]
    new = rx.scores_for(profiles, batches, reference, roles)
    check = old[[c for c in old if c.startswith("det_")]].to_numpy()
    assert np.allclose(check, new[[c for c in old if c.startswith("det_")]].to_numpy()), \
        "regenerated batches differ from the original run"
    rq1 = dmg.merge(new, on="batch")
    rq1.to_csv(args.out / "rq1_batches.csv", index=False)
    rx.evaluate(rq1, [p.consumer for p in profiles]).to_csv(args.out / "rq1_metrics.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({**meta, "rescored_from": str(args.old)}, indent=2))
    print(pd.read_csv(args.out / "rq1_metrics.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
