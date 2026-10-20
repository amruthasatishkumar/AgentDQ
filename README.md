# agentdq

The data defects that break an ML model are not the ones that break an LLM agent reading the same table. `agentdq` measures that difference and, from v0.1, validates incoming data **per consumer**.

Based on *Asymmetric Data-Quality Sensitivity in Predictive Models and LLM Agents* (Satishkumar & Padmanabha Suwarna, accepted at the ARRL workshop, IEEE ICDM 2026). Paper code: [AsymmetricDataQualitySensitivity](https://github.com/amruthasatishkumar/AsymmetricDataQualitySensitivity).

## What's in 0.0.1

- **9 corruption families** (missing values, type flips, category swaps, duplicates, schema drift, unit drift, timezone shifts, taxonomy drift, mojibake), deterministic and seeded, on any table via `ColumnRoles`.
- **Consumers**: `SklearnConsumer` for any fitted model, `LLMConsumer` for any function that scores a row (concurrency, retries, JSONL call log).

```python
import agentdq as dq
roles = dq.ColumnRoles.of(protected={"order_id", "label"}, monetary={"price"}, text={"review"})
bad = dq.apply(df, "currency_unit_drift", severity=0.2, seed=13, roles=roles)
```

## Coming in 0.1

`profile_sensitivity` (each consumer's damage per defect family), label-free detectors, and `validate` (a risk score per consumer for a new batch), plus a leakage guard.

## Licence

MIT
