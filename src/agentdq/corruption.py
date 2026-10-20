"""Dataset-agnostic corruption library.

Ported from the ICDM 2026 paper code (corruption/library.py), with the
Olist-specific column lists replaced by a ColumnRoles mapping so it runs on
any table.

Each family is a pure function (df, severity, seed, roles) -> df' that:
  * never touches `roles.protected` columns (labels, IDs, split markers)
  * is deterministic given (family, severity, seed)
  * preserves row count unless the family is row-level (duplicate_rows)
  * preserves column count unless the family is schema-level (schema_drift)

Severity is a fraction in [0, 1] interpreted per family (see docstrings).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnRoles:
    """Tells the families which columns play which role.

    protected:   never corrupted (labels, IDs, split markers)
    monetary:    numeric amounts, targeted by currency_unit_drift
    hour_of_day: integer 0-23 columns, shifted by timezone_collapse
    day_deltas:  durations in days, shifted by timezone_collapse
    text:        free text, targeted by mojibake_roundtrip
    taxonomy:    categorical codes, targeted by category_taxonomy_drift and mojibake
    taxonomy_variants: optional {official label: [plausible wrong variants]}
    """
    protected: frozenset[str] = frozenset()
    monetary: frozenset[str] = frozenset()
    hour_of_day: frozenset[str] = frozenset()
    day_deltas: frozenset[str] = frozenset()
    text: frozenset[str] = frozenset()
    taxonomy: frozenset[str] = frozenset()
    taxonomy_variants: dict[str, list[str]] = field(default_factory=dict, hash=False)

    @classmethod
    def of(cls, **kw) -> "ColumnRoles":
        sets = {k: frozenset(v) for k, v in kw.items() if k != "taxonomy_variants"}
        return cls(**sets, taxonomy_variants=dict(kw.get("taxonomy_variants") or {}))


def _feature_cols(df: pd.DataFrame, roles: ColumnRoles) -> list[str]:
    return [c for c in df.columns if c not in roles.protected]


def _numeric_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [
        c for c in cols
        if (pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[c]))
    ]


def _present(df: pd.DataFrame, cols: frozenset[str], roles: ColumnRoles) -> list[str]:
    return [c for c in df.columns if c in cols and c not in roles.protected]


# ---------------- families ----------------

def missing_injection(df, severity, seed, roles):
    """Set `severity` fraction of values in each feature column to NaN.

    Real-world: optional fields, opt-outs, sensor drop-outs.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in _feature_cols(out, roles):
        mask = rng.random(len(out)) < severity
        if mask.any():
            if not (pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_object_dtype(out[col])):
                out[col] = out[col].astype("float64" if pd.api.types.is_numeric_dtype(out[col]) else "object")
            out.loc[mask, col] = np.nan
    return out


def type_flip(df, severity, seed, roles):
    """Coerce `severity` fraction of numeric values into strings.

    Real-world: ETL misconfiguration, schema-on-read inconsistencies.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in _numeric_cols(out, _feature_cols(out, roles)):
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        new = out[col].astype("object")
        new.loc[mask] = new[mask].apply(lambda v: f"{v}" if pd.notna(v) else v)
        out[col] = new
    return out


def label_noise_categorical(df, severity, seed, roles):
    """Swap `severity` fraction of values in each non-text categorical feature to a
    different level of the same column. Corrupts features, not the label.

    Real-world: data-entry errors, flaky normalisation of free-form fields.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    cats = [c for c in _categorical_cols(out, _feature_cols(out, roles)) if c not in roles.text]
    for col in cats:
        levels = out[col].dropna().astype(str).unique()
        if len(levels) < 2:
            continue
        mask = rng.random(len(out)) < severity
        if not mask.any():
            continue
        choices = rng.choice(levels, size=int(mask.sum()), replace=True)
        originals = out.loc[mask, col].astype(str).to_numpy()
        for i, orig in enumerate(originals):
            if choices[i] == orig:
                choices[i] = levels[(np.where(levels == orig)[0][0] + 1) % len(levels)]
        out[col] = out[col].astype("object")
        out.loc[mask, col] = choices
    return out


def duplicate_rows(df, severity, seed, roles):
    """Append duplicates of `severity` fraction of rows.

    Real-world: idempotency bugs, retry-without-dedup in ingestion.
    """
    n = len(df)
    k = int(round(severity * n))
    if n == 0 or k == 0:
        return df.copy()
    idx = np.random.default_rng(seed).integers(0, n, size=k)
    return pd.concat([df, df.iloc[idx]], ignore_index=True)


def schema_drift(df, severity, seed, roles):
    """Rename `severity` fraction of feature columns to synthetic names.

    Real-world: upstream schema migrations not propagated to consumers.
    """
    feats = _feature_cols(df, roles)
    n_rename = int(round(severity * len(feats)))
    if n_rename == 0:
        return df.copy()
    chosen = np.random.default_rng(seed).choice(len(feats), size=n_rename, replace=False)
    return df.rename(columns={feats[ci]: f"col_{seed}_{i:03d}" for i, ci in enumerate(chosen)})


def currency_unit_drift(df, severity, seed, roles):
    """Multiply `severity` fraction of monetary values by 100.

    Real-world: cents-vs-units migrations, one feed silently switching currency unit.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in _present(out, roles.monetary, roles):
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = rng.random(len(out)) < severity
        if mask.any():
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
            out.loc[mask, col] = out.loc[mask, col] * 100.0
    return out


def timezone_collapse(df, severity, seed, roles):
    """Shift time-derived features for `severity` fraction of rows, modelling a
    3-hour UTC/local confusion: hour-of-day by ±3 (mod 24), day deltas by ±0.125.

    Real-world: mixed UTC and local timestamps, daylight-saving bugs.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    for col in _present(out, roles.hour_of_day, roles):
        mask = rng.random(n) < severity
        if not mask.any():
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        shifts = pd.Series(0, index=out.index)
        shifts[mask] = rng.choice([-3, 3], size=int(mask.sum()))
        out[col] = ((vals + shifts) % 24).astype("Int64")
    for col in _present(out, roles.day_deltas, roles):
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        mask = rng.random(n) < severity
        if not mask.any():
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out.loc[mask, col] = out.loc[mask, col] + rng.choice([-0.125, 0.125], size=int(mask.sum()))
    return out


def _generic_variant(label: str, rng: np.random.Generator) -> str:
    """A plausible wrong spelling when no variant list is given."""
    options = [
        label.replace("_", "-") if "_" in label else label + "-",
        label.replace("a", "á", 1) if "a" in label else label + "_",
        label + "s" if not label.endswith("s") else label[:-1],
        label.upper() if label != label.upper() else label.lower(),
    ]
    return options[int(rng.integers(0, len(options)))]


def category_taxonomy_drift(df, severity, seed, roles):
    """Mutate category labels into plausible variants (accents, plurals,
    separators), using `roles.taxonomy_variants` where given.

    Real-world: vocabulary drift, inconsistent diacritics between systems.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in _present(out, roles.taxonomy, roles):
        s = out[col].astype("object")
        mask = rng.random(len(out)) < severity
        for j in np.where(mask)[0]:
            v = s.iloc[j]
            if pd.isna(v):
                continue
            key = str(v)
            variants = roles.taxonomy_variants.get(key)
            s.iloc[j] = variants[int(rng.integers(0, len(variants)))] if variants else _generic_variant(key, rng)
        out[col] = s
    return out


def _mojibake_one(text: str) -> str:
    """UTF-8 bytes misread as Latin-1: 'não' -> 'nÃ£o'."""
    try:
        return text.encode("utf-8").decode("latin-1")
    except UnicodeError:
        return text


def mojibake_roundtrip(df, severity, seed, roles):
    """Apply UTF-8 -> Latin-1 mis-decoding to `severity` fraction of rows in
    text and taxonomy columns.

    Real-world: encoding misconfiguration between source, ETL and warehouse.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col in _present(out, roles.text | roles.taxonomy, roles):
        s = out[col].astype("object")
        mask = rng.random(len(out)) < severity
        for j in np.where(mask)[0]:
            if pd.notna(s.iloc[j]):
                s.iloc[j] = _mojibake_one(str(s.iloc[j]))
        out[col] = s
    return out


# ---------------- registry ----------------

Family = Callable[[pd.DataFrame, float, int, ColumnRoles], pd.DataFrame]

REGISTRY: dict[str, Family] = {
    "missing_injection": missing_injection,
    "type_flip": type_flip,
    "label_noise_categorical": label_noise_categorical,
    "duplicate_rows": duplicate_rows,
    "schema_drift": schema_drift,
    "currency_unit_drift": currency_unit_drift,
    "timezone_collapse": timezone_collapse,
    "category_taxonomy_drift": category_taxonomy_drift,
    "mojibake_roundtrip": mojibake_roundtrip,
}

# The six families used in the ICDM sweep.
CANONICAL = ["missing_injection", "mojibake_roundtrip", "category_taxonomy_drift",
             "currency_unit_drift", "schema_drift", "type_flip"]


def list_families() -> list[str]:
    return list(REGISTRY)


def register(name: str, fn: Family) -> None:
    """Add a custom family."""
    REGISTRY[name] = fn


def apply(df: pd.DataFrame, family: str, severity: float, seed: int,
          roles: ColumnRoles | None = None) -> pd.DataFrame:
    roles = roles or ColumnRoles()
    if family == "clean":
        return df.copy()
    if family not in REGISTRY:
        raise KeyError(f"unknown corruption family: {family!r}; known: {list_families()}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity out of [0,1]: {severity}")
    return REGISTRY[family](df, severity, seed, roles)
