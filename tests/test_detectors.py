"""Acceptance tests for the detectors: each must recover a known injected severity."""
import numpy as np
import pandas as pd
import pytest

from agentdq import ColumnRoles, apply
from agentdq.detectors import DETECTORS, detect_all

ROLES = ColumnRoles.of(protected={"id", "label"}, monetary={"price"},
                       text={"review"}, taxonomy={"category"})


def _frame(seed, n=3000):
    rng = np.random.default_rng(seed)
    price = rng.uniform(5, 500, n).round(2)
    price[rng.random(n) < 0.05] = np.nan           # some missing already, as in real data
    return pd.DataFrame({
        "id": np.arange(n),
        "label": rng.integers(0, 2, n),
        "price": price,
        "qty": rng.integers(1, 5, n),
        "category": rng.choice(["toys", "garden", "books"], n),
        "review": rng.choice(["não gostei", "ótimo produto", "ok"], n),
    })


@pytest.fixture(scope="module")
def reference():
    return _frame(0)


@pytest.fixture(scope="module")
def batch():
    return _frame(1)      # same distribution, different rows


@pytest.mark.parametrize("family", list(DETECTORS))
def test_clean_batch_reads_near_zero(reference, batch, family):
    assert DETECTORS[family](batch, reference, ROLES) < 0.03


@pytest.mark.parametrize("family", list(DETECTORS))
@pytest.mark.parametrize("severity", [0.1, 0.3])
def test_recovers_injected_severity(reference, batch, family, severity):
    bad = apply(batch, family, severity, seed=7, roles=ROLES)
    est = DETECTORS[family](bad, reference, ROLES)
    if family == "schema_drift":   # renames round(s * 4 features) columns
        expected = round(severity * 4) / 4
        assert est == pytest.approx(expected)
    else:
        assert est == pytest.approx(severity, abs=0.05), f"{family}: {est:.3f}"


def test_nothing_to_look_at_returns_zero(reference, batch):
    bare = ColumnRoles.of(protected={"id", "label"})
    for fam in ("currency_unit_drift", "mojibake_roundtrip", "category_taxonomy_drift"):
        assert DETECTORS[fam](batch, reference, bare) == 0.0


def test_detect_all(reference, batch):
    bad = apply(batch, "currency_unit_drift", 0.3, seed=7, roles=ROLES)
    out = detect_all(bad, reference, ROLES)
    assert set(out) == set(DETECTORS)
    assert all(0.0 <= v <= 1.0 for v in out.values())
    assert max(out, key=out.get) == "currency_unit_drift"
