"""Add the LLM agent (Azure OpenAI gpt-4o-mini) as a third consumer on OULAD.

Reads environment variables, never a key on the command line:
    AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT   the deployment name for gpt-4o-mini
    AZURE_OPENAI_API_KEY      optional; if unset, signs in with your Microsoft (Entra ID) login
                              via azure-identity (run `az login` first, or be signed in to VS Code)
Optional: AZURE_OPENAI_API_VERSION (default 2024-06-01).

Two modes:
    --pilot   score 200 clean test rows only, report AUROC, calls and token use (cheap check
              that the agent does better than chance before spending on the full run)
    (default) profile the agent on 100 probe rows, then score --batches mixed batches of
              --rows rows and evaluate RQ1 for all three consumers. Identical rows are cached,
              so clean rows reused across batches and profiles are paid for once.

Run:
    python3 run_oulad_agent.py --data data/oulad/oulad_day28.parquet --out results/oulad_agent --pilot
    python3 run_oulad_agent.py --data data/oulad/oulad_day28.parquet --out results/oulad_agent
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import agentdq as dq
from agentdq.profile import auroc, balanced_sample

sys.path.insert(0, str(Path(__file__).parent))
import run_experiments as rx  # noqa: E402
from build_oulad import CATEGORICAL, LABEL, NUMERIC  # noqa: E402
from run_oulad import FAMILIES, build_consumers, oulad_roles  # noqa: E402

FEATS = CATEGORICAL + NUMERIC
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6  # gpt-4o-mini list price per token (check yours)


def training_profile(train: pd.DataFrame) -> str:
    """Short text summary of the training set the agent is given, by outcome."""
    lines = [f"Training students: {len(train)}; share who failed or withdrew: {train[LABEL].mean():.2f}."]
    for c in NUMERIC:
        g = train.groupby(LABEL)[c].median()
        lines.append(f"- {c}: median {g.get(0, np.nan):.1f} for pass, {g.get(1, np.nan):.1f} for fail/withdraw")
    for c in ("highest_education", "imd_band", "num_of_prev_attempts"):
        if c in CATEGORICAL:
            r = train.groupby(c)[LABEL].mean().round(2).to_dict()
            lines.append(f"- fail/withdraw rate by {c}: {r}")
    return "\n".join(lines)


class AzureAgent:
    def __init__(self, profile_text: str, log_path: Path):
        self.url = (os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/deployments/"
                    + os.environ["AZURE_OPENAI_DEPLOYMENT"] + "/chat/completions?api-version="
                    + os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"))
        self.key = os.environ.get("AZURE_OPENAI_API_KEY")
        self._cred = None
        if not self.key:
            from azure.identity import DefaultAzureCredential
            self._cred = DefaultAzureCredential()
        self.system = ("You are an analyst predicting whether an Open University student will fail "
                       "or withdraw from their course, using only what is known 28 days in.\n"
                       "Training-set profile:\n" + profile_text +
                       "\nReply with JSON only: {\"p_fail_or_withdraw\": <number between 0 and 1>}.")
        self.cache: dict[str, float] = {}
        self.lock = threading.Lock()
        self.calls = self.tok_in = self.tok_out = 0
        self.log_path = log_path

    def __call__(self, row: dict) -> float:
        key = json.dumps(row, sort_keys=True, default=str)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        body = json.dumps({"messages": [{"role": "system", "content": self.system},
                                        {"role": "user", "content": "Student record: " + key}],
                           "temperature": 0, "max_tokens": 20,
                           "response_format": {"type": "json_object"}}).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["api-key"] = self.key
        else:  # Entra ID token; azure-identity caches and refreshes it
            tok = self._cred.get_token("https://cognitiveservices.azure.com/.default").token
            headers["Authorization"] = "Bearer " + tok
        req = urllib.request.Request(self.url, data=body, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
        text = out["choices"][0]["message"]["content"]
        m = re.search(r"[-+]?\d*\.?\d+", text)
        p = float(m.group()) if m else 0.5
        with self.lock:
            self.cache[key] = p
            self.calls += 1
            self.tok_in += out.get("usage", {}).get("prompt_tokens", 0)
            self.tok_out += out.get("usage", {}).get("completion_tokens", 0)
        return p

    def cost(self) -> dict:
        return {"calls": self.calls, "tokens_in": self.tok_in, "tokens_out": self.tok_out,
                "usd_estimate": round(self.tok_in * PRICE_IN + self.tok_out * PRICE_OUT, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--probe-rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_parquet(args.data)
    train = df[df.split == "train"]
    cols = ["row_id", LABEL] + FEATS
    val = df[df.split == "val"][cols].reset_index(drop=True)
    test = df[df.split == "test"][cols].reset_index(drop=True)
    agent_fn = AzureAgent(training_profile(train), args.out / "agent_calls.jsonl")
    agent = dq.LLMConsumer(agent_fn, FEATS, name="agent", max_workers=args.workers,
                           log_path=args.out / "agent_calls.jsonl")

    if args.pilot:
        s = balanced_sample(test, LABEL, 200, 7)
        a = auroc(s[LABEL].to_numpy(), agent.score(s))
        per_call = agent_fn.cost()
        res = {"pilot_rows": 200, "agent_auroc": a, **per_call}
        full_calls = len(test) + args.batches * args.rows + 3 * args.probe_rows * 11
        if per_call["calls"]:
            res["full_run_calls_upper_bound"] = full_calls
            res["full_run_usd_estimate"] = round(per_call["usd_estimate"] / per_call["calls"] * full_calls, 2)
        (args.out / "pilot.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0

    roles = oulad_roles()
    offline, _ = build_consumers(train)
    consumers = offline + [agent]
    reference = balanced_sample(val, LABEL, 1000, 999)
    profiles = []
    for c in consumers:
        p = dq.profile_sensitivity(c, val, LABEL, roles=roles, families=FAMILIES,
                                   n_rows=args.probe_rows)
        p.to_json(args.out / f"profile_{c.name}.json")
        profiles.append(p)
        print(f"[{time.time()-t0:.0f}s] profile {c.name} done; agent so far {agent_fn.cost()}")
    batches = rx.make_batches(test, roles, args.batches, args.rows, (0.0, 0.4), args.seed)
    dmg = rx.realised_damage(consumers, batches)
    rq1 = dmg.merge(rx.scores_for(profiles, batches, reference, roles), on="batch")
    rq1.to_csv(args.out / "rq1_batches.csv", index=False)
    rx.evaluate(rq1, [c.name for c in consumers]).to_csv(args.out / "rq1_metrics.csv", index=False)
    meta = {"batches": args.batches, "rows": args.rows, "probe_rows": args.probe_rows,
            "agent": agent_fn.cost(), "seconds": round(time.time() - t0)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
