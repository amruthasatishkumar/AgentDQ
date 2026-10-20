"""Stage 2, part 1: label-free detectors.

Each detector compares a new batch with a clean reference sample and estimates the
severity (0 to 1) of one corruption family, in the same units the corruption library
uses. That is what lets validate() multiply it by a consumer's profile slope.

No labels and no consumer calls are needed, so this runs on every batch for free.
Every detector returns a float clipped to [0, 1], and 0.0 when it has nothing to look at
(for example, no monetary columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agentdq.corruption import ColumnRoles

MOJIBAKE_MARKERS = ("Ã", "Â")


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x))) if np.isfinite(x) else 0.0


def _features(df: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    return [c for c in df.columns if c not in roles.protected]


def _non_ascii_share(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    return float(s.map(lambda x: not x.isascii()).mean()) if len(s) else 0.0


def _marker_share(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    return float(s.map(lambda x: any(m in x for m in MOJIBAKE_MARKERS)).mean()) if len(s) else 0.0


# ---------------- Step 3a ----------------
def detect_schema_drift(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """Share of the reference's feature columns that are missing from the batch."""
    feats = _features(reference, roles)
    if not feats:
        return 0.0
    return _clip(sum(c not in batch.columns for c in feats) / len(feats))


def detect_missing_injection(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """Extra missingness, averaged over feature columns present in both frames.

    missing_injection blanks a fraction s of ALL rows, so a column that was r missing
    becomes r + (1 - r) * s missing. Invert that per column: s = (b - r) / (1 - r),
    where r and b are the reference and batch missing rates. Skip columns with r == 1.
    """
    vals = []
    for c in _features(reference, roles):
        if c not in batch.columns:
            continue
        r = reference[c].isna().mean()
        if r >= 1:
            continue
        b = batch[c].isna().mean()
        vals.append((b - r) / (1 - r))
    return _clip(float(np.mean(vals))) if vals else 0.0


# ---------------- Step 3b ----------------
def detect_type_flip(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """For feature columns that are numeric in the reference and present in the batch:
    the share of non-null batch values that are Python strings. Mean over those columns.
    """
    vals = []
    for c in _features(reference, roles):
        if c not in batch.columns or not pd.api.types.is_numeric_dtype(reference[c]):
            continue
        s = batch[c].dropna()
        if len(s):
            vals.append(s.map(lambda x: isinstance(x, str)).mean())
    return _clip(float(np.mean(vals))) if vals else 0.0


def detect_currency_unit_drift(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """For roles.monetary columns in both frames: the share of non-null batch values
    above 2 x the reference maximum (a x100 shift lands far above anything seen).
    Coerce with pd.to_numeric(errors="coerce") first. Mean over those columns.
    """
    vals = []
    for c in roles.monetary:
        if c not in batch.columns or c not in reference.columns:
            continue
        ref_max = pd.to_numeric(reference[c], errors="coerce").max()
        b = pd.to_numeric(batch[c], errors="coerce").dropna()
        if len(b) and np.isfinite(ref_max):
            vals.append((b > 2 * ref_max).mean())
    return _clip(float(np.mean(vals))) if vals else 0.0


# ---------------- Step 3c ----------------
def detect_mojibake_roundtrip(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """For roles.text and roles.taxonomy columns in both frames.

    Mojibake only changes values that contain non-ASCII characters, so divide the rise
    in marker rate by the reference's non-ASCII share:
        s = (marker share in batch - marker share in reference) / non-ASCII share in reference
    where "marker share" = share of non-null values containing any MOJIBAKE_MARKERS.
    Skip columns whose reference has no non-ASCII values. Mean over the rest.
    """
    vals = []
    for c in sorted(roles.text | roles.taxonomy):
        if c not in batch.columns or c not in reference.columns:
            continue
        non_ascii = _non_ascii_share(reference[c])
        if non_ascii == 0:
            continue
        vals.append((_marker_share(batch[c]) - _marker_share(reference[c])) / non_ascii)
    return _clip(float(np.mean(vals))) if vals else 0.0


def detect_category_taxonomy_drift(batch: pd.DataFrame, reference: pd.DataFrame, roles: ColumnRoles) -> float:
    """For roles.taxonomy columns in both frames: the share of non-null batch values
    (as str) that never appear in the reference column. Mean over those columns.
    """
    vals = []
    for c in sorted(roles.taxonomy):
        if c not in batch.columns or c not in reference.columns:
            continue
        s = batch[c].dropna().astype(str)
        if len(s):
            vocab = set(reference[c].dropna().astype(str))
            vals.append((~s.isin(vocab)).mean())
    return _clip(float(np.mean(vals))) if vals else 0.0


# ---------------- Step 3d ----------------
DETECTORS = {
    "missing_injection": detect_missing_injection,
    "mojibake_roundtrip": detect_mojibake_roundtrip,
    "category_taxonomy_drift": detect_category_taxonomy_drift,
    "currency_unit_drift": detect_currency_unit_drift,
    "schema_drift": detect_schema_drift,
    "type_flip": detect_type_flip,
}


def detect_all(batch: pd.DataFrame, reference: pd.DataFrame,
               roles: ColumnRoles | None = None) -> dict[str, float]:
    """Run every detector in DETECTORS and return {family: estimated severity}."""
    roles = roles or ColumnRoles()
    return {family: fn(batch, reference, roles) for family, fn in DETECTORS.items()}
