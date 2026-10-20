# agentdq

The data defects that break an ML model are not the ones that break an LLM agent reading the same table. `agentdq` measures that difference, then checks each new batch of data **per consumer**: the same batch can be safe for your agent and harmful for your model.

Based on *Asymmetric Data-Quality Sensitivity in Predictive Models and LLM Agents* (Satishkumar & Padmanabha Suwarna, ARRL workshop, IEEE ICDM 2026). Paper code: [AsymmetricDataQualitySensitivity](https://github.com/amruthasatishkumar/AsymmetricDataQualitySensitivity).

```
pip install agentdq
```

## How it works

1. **Profile each consumer once, offline.** `profile_sensitivity` corrupts a labelled sample in known ways and measures how much each consumer's AUROC drops per unit of each defect. Families that don't touch the columns a consumer reads are skipped without any scoring calls, so profiling an LLM agent stays cheap.
2. **Check every new batch, without labels.** `validate` runs six label-free detectors against a clean reference sample, estimates each defect's severity, and multiplies it by each consumer's sensitivity. The result is an expected AUROC drop and a pass, warn or fail per consumer, next to the verdict a one-size-fits-all checker would give.

## Example

```python
import agentdq as dq

roles = dq.ColumnRoles.of(protected={"order_id", "label"}, monetary={"price"},
                          text={"review"}, taxonomy={"category"})

tree = dq.SklearnConsumer(fitted_model, channels=["price", "qty", "category"], name="tree")
agent = dq.LLMConsumer(score_review_with_llm, channels=["review", "category"], name="agent")

profiles = [dq.profile_sensitivity(c, labelled_sample, "label", roles=roles) for c in (tree, agent)]
profiles[0].to_json("tree_profile.json")          # reuse later with SensitivityProfile.from_json

report = dq.validate(todays_batch, reference_sample, profiles, roles=roles, label_cols={"label"})
print(report.to_markdown())
```

```
Baseline (consumer-agnostic) verdict: **fail**

| Consumer | Risk | Verdict | Top defects |
|---|---|---|---|
| tree | 0.066 | fail | currency_unit_drift |
| agent | 0.000 | pass | none |

| Defect | Severity |
|---|---|
| currency_unit_drift | 0.297 |
```

Here 30% of prices arrived in cents. A classic gate fails the whole batch; agentdq says the tree model should not use it, but the review-reading agent can.

`validate` raises `LeakageError` if any consumer reads a label column, the bug the paper's leakage audit found.

## What's inside

| Module | What it does |
|---|---|
| `corruption` | 9 seeded corruption families (missing values, type flips, category swaps, duplicates, schema drift, unit drift, timezone shifts, taxonomy drift, mojibake) on any table, via `ColumnRoles` |
| `consumers` | `SklearnConsumer` for any fitted model; `LLMConsumer` for any function that scores a row, with concurrency, retries and a JSONL call log |
| `profile` | `profile_sensitivity` and `SensitivityProfile` (mean drop, bootstrap 95% CI and slope per family) |
| `detectors` | Label-free severity estimates for missing values, schema drift, type flips, unit drift, mojibake and taxonomy drift |
| `validate` | Per-consumer risk, verdicts, the leakage guard and `Report.to_markdown()` |

## Status

Alpha (0.1.0a1). The API may change before 0.1.0. Runs offline with no LLM needed; CI tests Python 3.10 and 3.12.

## Acknowledgements

The corruption families are ported from the ICDM paper code. The scaffolding, docstrings and tests were drafted with AI assistance (Claude), and the profiling, detection and validation code was implemented with AI assistance (GitHub Copilot). The design follows the author's paper, and the commit history records the AI co-authorship.

## Licence

MIT
