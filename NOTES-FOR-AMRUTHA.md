# Notes for Amrutha (don't publish this file)

Starter drafted by Claude on 19 Oct 2026. 35 tests pass. Not on GitHub.

- `corruption.py` is your paper's corruption library, ported: same 9 families and semantics, with the Olist column lists replaced by `ColumnRoles`, and taxonomy drift made generic (Portuguese variant list becomes an optional mapping).
- `consumers.py` is the `Consumer` protocol plus the sklearn and LLM adapters from the Research thread's spec (global-talent/research/validator-paper-plan.md).
- Not yet: `profile_sensitivity`, detectors, `validate`, `Report`, leakage guard. These are the core of the AAAI paper, so they are the parts that should be visibly yours.
- Wait for the Research thread's corrected `profile.py` (leakage fix) before porting profiling code.
- Copyright line: the new package lists only you; the README credits the paper and co-author. Your call.
