"""Acceptance tests for validate(): the same batch, different verdicts per consumer."""
import numpy as np
import pandas as pd
import pytest

from agentdq import ColumnRoles, apply
from agentdq.profile import FamilyResult, SensitivityProfile, profile_sensitivity
from agentdq.validate import LeakageError, check_leakage, consumer_risk, validate

ROLES = ColumnRoles.of(protected={"id", "label"}, monetary={"price"},
                       text={"review"}, taxonomy={"category"})
FAMS = ["missing_injection", "mojibake_roundtrip", "category_taxonomy_drift",
        "currency_unit_drift", "schema_drift", "type_flip"]


def _profile(name, channels, slopes):
    fams = {f: FamilyResult(f, slopes.get(f, 0) != 0, [0.1, 0.3], [0, 0], [0, 0], [0, 0], slopes.get(f, 0.0))
            for f in FAMS}
    return SensitivityProfile(name, channels, 0.85, fams)


TREE = _profile("tree", ["price", "qty", "category"], {"currency_unit_drift": 0.5, "missing_injection": 0.1})
AGENT = _profile("agent", ["review", "category"], {"mojibake_roundtrip": 0.4, "category_taxonomy_drift": 0.05})


def _frame(seed, n=3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n), "label": rng.integers(0, 2, n),
        "price": rng.uniform(5, 500, n).round(2), "qty": rng.integers(1, 5, n),
        "category": rng.choice(["toys", "garden", "books"], n),
        "review": rng.choice(["não gostei", "ótimo produto", "ok"], n),
    })


@pytest.fixture(scope="module")
def reference():
    return _frame(0)


@pytest.fixture(scope="module")
def batch():
    return _frame(1)


# Step 4a
def test_leakage_guard():
    check_leakage([TREE, AGENT], label_cols={"label"})          # fine
    leaky = _profile("leaky", ["price", "label"], {})
    with pytest.raises(LeakageError, match="leaky"):
        check_leakage([TREE, leaky], label_cols={"label"})


# Step 4b
def test_consumer_risk_maths():
    detected = {"currency_unit_drift": 0.3, "missing_injection": 0.1, "mojibake_roundtrip": 0.2}
    r = consumer_risk(TREE, detected, warn=0.02, fail=0.05)
    assert r.risk == pytest.approx(0.3 * 0.5 + 0.1 * 0.1)
    assert r.verdict == "fail"
    assert r.top_defects == ["currency_unit_drift", "missing_injection"]
    neg = _profile("neg", ["x"], {"type_flip": -0.4, "schema_drift": 0.1})
    r = consumer_risk(neg, {"type_flip": 0.5, "schema_drift": 0.25}, warn=0.02, fail=0.05)
    assert r.risk == pytest.approx(0.025) and r.verdict == "warn"
    assert r.top_defects == ["schema_drift"]


# Step 4c
def test_same_batch_different_verdicts(reference, batch):
    bad = apply(batch, "currency_unit_drift", 0.3, seed=7, roles=ROLES)
    rep = validate(bad, reference, [TREE, AGENT], roles=ROLES, label_cols={"label"})
    assert rep.verdict_for("tree") == "fail"
    assert rep.verdict_for("agent") == "pass"
    assert rep.baseline_verdict == "fail"
    assert rep.consumers[0].top_defects[0] == "currency_unit_drift"
    assert rep.thresholds == {"warn": 0.02, "fail": 0.05, "baseline": 0.05}


def test_clean_batch_passes_everyone(reference, batch):
    rep = validate(batch, reference, [TREE, AGENT], roles=ROLES)
    assert [c.verdict for c in rep.consumers] == ["pass", "pass"]
    assert rep.baseline_verdict == "pass"


def test_validate_refuses_leaky_profiles(reference, batch):
    with pytest.raises(LeakageError):
        validate(batch, reference, [_profile("leaky", ["label"], {})], roles=ROLES, label_cols={"label"})


def test_end_to_end_with_a_real_profile(reference, batch):
    class PriceReader:
        name, channels = "price_reader", ["price"]
        def score(self, df):
            p = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df else pd.Series(0, index=df.index)
            return (p.fillna(0) > 250).astype(float).to_numpy()
    probe = reference.assign(label=(reference["price"] > 250).astype(int))
    prof = profile_sensitivity(PriceReader(), probe, "label", roles=ROLES, seeds=2)
    rep = validate(apply(batch, "currency_unit_drift", 0.3, 7, ROLES), reference, [prof], roles=ROLES)
    assert rep.consumers[0].risk > 0 and "currency_unit_drift" in rep.consumers[0].top_defects


# Step 4d
def test_markdown(reference, batch):
    bad = apply(batch, "mojibake_roundtrip", 0.3, seed=7, roles=ROLES)
    md = validate(bad, reference, [TREE, AGENT], roles=ROLES).to_markdown()
    assert "| Consumer |" in md and "tree" in md and "agent" in md
    assert "mojibake_roundtrip" in md and "fail" in md and "pass" in md
