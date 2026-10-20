"""Stage 2, part 2: consumer-aware validation.

Combine what the detectors see in a batch (stage 2) with what each consumer is sensitive
to (stage 1) into one risk score per consumer:

    risk(consumer) = sum over families of  detected_severity[family] * profile.slope[family]

Because slope is "AUROC drop per unit severity", risk reads as "expected AUROC drop on
this batch". The same batch can pass for one consumer and fail for another, which is the
paper's point. A consumer-agnostic baseline verdict is reported alongside for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agentdq.corruption import ColumnRoles
from agentdq.detectors import detect_all
from agentdq.profile import SensitivityProfile


class LeakageError(ValueError):
    """A consumer reads a label column, so any score it gets is meaningless."""


@dataclass
class ConsumerRisk:
    consumer: str
    risk: float                                   # expected AUROC drop on this batch
    verdict: str                                  # "pass", "warn" or "fail"
    contributions: dict[str, float]               # family -> severity * slope
    top_defects: list[str]                        # up to 3 families, largest contribution first


@dataclass
class Report:
    detected: dict[str, float]                    # family -> estimated severity
    consumers: list[ConsumerRisk]
    baseline_verdict: str                         # what a one-size-fits-all checker would say
    thresholds: dict[str, float] = field(default_factory=dict)

    def verdict_for(self, consumer: str) -> str:
        return next(c.verdict for c in self.consumers if c.consumer == consumer)

    # ---------------- Step 4d ----------------
    def to_markdown(self) -> str:
        """A short Markdown summary:

        1. a line with the baseline verdict
        2. a table with one row per consumer: | Consumer | Risk | Verdict | Top defects |
           (risk to 3 decimals; top defects comma-separated, or "none")
        3. a table of detected severities above 0: | Defect | Severity |
        """
        lines = [f"Baseline (consumer-agnostic) verdict: **{self.baseline_verdict}**", "",
                 "| Consumer | Risk | Verdict | Top defects |",
                 "|---|---|---|---|"]
        for c in self.consumers:
            lines.append(f"| {c.consumer} | {c.risk:.3f} | {c.verdict} | "
                         f"{', '.join(c.top_defects) or 'none'} |")
        lines += ["", "| Defect | Severity |", "|---|---|"]
        for family, sev in self.detected.items():
            if sev > 0:
                lines.append(f"| {family} | {sev:.3f} |")
        return "\n".join(lines)


# ---------------- Step 4a ----------------
def check_leakage(profiles: list[SensitivityProfile], label_cols) -> None:
    """Raise LeakageError naming the consumer and column if any profile's channels
    include a column in label_cols. This is the ICDM leakage bug, made impossible.
    """
    for p in profiles:
        leaked = set(label_cols) & set(p.channels)
        if leaked:
            raise LeakageError(f"consumer {p.consumer!r} reads label column(s) {sorted(leaked)}")


# ---------------- Step 4b ----------------
def consumer_risk(profile: SensitivityProfile, detected: dict[str, float],
                  warn: float, fail: float) -> ConsumerRisk:
    """Score one consumer.

    - contributions: for every family in BOTH profile.families and detected,
      detected[family] * profile.families[family].slope. Negative slopes (a defect that
      happens to help) count as 0, so noise can't cancel real harm.
    - risk = sum of contributions
    - verdict: "fail" if risk >= fail, else "warn" if risk >= warn, else "pass"
    - top_defects: families with contribution > 0, largest first, at most 3
    """
    contributions = {f: detected[f] * max(0.0, r.slope)
                     for f, r in profile.families.items() if f in detected}
    risk = sum(contributions.values())
    verdict = "fail" if risk >= fail else "warn" if risk >= warn else "pass"
    ranked = sorted((f for f, v in contributions.items() if v > 0),
                    key=lambda f: contributions[f], reverse=True)
    return ConsumerRisk(profile.consumer, risk, verdict, contributions, ranked[:3])


# ---------------- Step 4c ----------------
def validate(batch: pd.DataFrame, reference: pd.DataFrame, profiles: list[SensitivityProfile],
             roles: ColumnRoles | None = None, label_cols=(), warn: float = 0.02,
             fail: float = 0.05, baseline_threshold: float = 0.05) -> Report:
    """Check one batch for every consumer.

    1. check_leakage(profiles, label_cols); refuse to go on if it raises
    2. detected = detect_all(batch, reference, roles)
    3. one consumer_risk(...) per profile, in the order given
    4. baseline_verdict: "fail" if any detected severity >= baseline_threshold, else "pass"
       (a classic data-quality gate that ignores who reads the data)
    5. return a Report, with thresholds={"warn": warn, "fail": fail, "baseline": baseline_threshold}
    """
    check_leakage(profiles, label_cols)
    detected = detect_all(batch, reference, roles)
    consumers = [consumer_risk(p, detected, warn, fail) for p in profiles]
    baseline = "fail" if any(s >= baseline_threshold for s in detected.values()) else "pass"
    return Report(detected, consumers, baseline,
                  {"warn": warn, "fail": fail, "baseline": baseline_threshold})
