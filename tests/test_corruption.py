import numpy as np
import pandas as pd
import pytest

from agentdq import ColumnRoles, apply, list_families

ROLES = ColumnRoles.of(
    protected={"id", "label"}, monetary={"price"}, hour_of_day={"hour"},
    day_deltas={"days_late"}, text={"review"}, taxonomy={"category"},
    taxonomy_variants={"toys": ["toy", "brinquedos"]},
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "id": np.arange(n),
        "label": rng.integers(0, 2, n),
        "price": rng.uniform(5, 500, n).round(2),
        "qty": rng.integers(1, 5, n),
        "hour": rng.integers(0, 24, n),
        "days_late": rng.normal(0, 2, n),
        "category": rng.choice(["toys", "garden", "books"], n),
        "review": rng.choice(["não gostei", "ótimo produto", "ok"], n),
    })


@pytest.mark.parametrize("family", list_families())
def test_deterministic(df, family):
    pd.testing.assert_frame_equal(apply(df, family, 0.2, 13, ROLES), apply(df, family, 0.2, 13, ROLES))


@pytest.mark.parametrize("family", list_families())
def test_protected_columns_untouched(df, family):
    out = apply(df, family, 0.3, 17, ROLES).head(len(df))
    for col in ("id", "label"):
        assert out[col].reset_index(drop=True).equals(df[col]), f"{family} changed {col}"


@pytest.mark.parametrize("family", list_families())
def test_severity_zero_keeps_shape(df, family):
    out = apply(df, family, 0.0, 0, ROLES)
    assert out.shape == df.shape


def test_currency_drift_hits_expected_fraction(df):
    out = apply(df, "currency_unit_drift", 0.25, 5, ROLES)
    frac = (out["price"] / df["price"]).round().eq(100).mean()
    assert 0.18 < frac < 0.32


def test_taxonomy_uses_given_variants(df):
    out = apply(df, "category_taxonomy_drift", 1.0, 3, ROLES)
    assert set(out.loc[df["category"] == "toys", "category"]) <= {"toy", "brinquedos"}


def test_mojibake_produces_latin1_markers(df):
    out = apply(df, "mojibake_roundtrip", 1.0, 3, ROLES)
    assert out["review"].str.contains("Ã").any()


def test_hours_stay_in_range(df):
    out = apply(df, "timezone_collapse", 1.0, 3, ROLES)
    assert out["hour"].between(0, 23).all()


def test_unknown_family_and_bad_severity(df):
    with pytest.raises(KeyError):
        apply(df, "nope", 0.1, 0)
    with pytest.raises(ValueError):
        apply(df, "missing_injection", 1.5, 0)
