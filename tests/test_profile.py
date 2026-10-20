"""Acceptance tests for Stage 1. Each block says which step makes it pass."""
import numpy as np
import pandas as pd
import pytest

from agentdq import ColumnRoles
from agentdq.profile import (SensitivityProfile, auroc, balanced_sample,
                             profile_sensitivity, touches_channels)


@pytest.fixture(scope="module")
def probe():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "y": y,
        "x": y * 4 + rng.normal(0, 1.5, n),            # what the consumer reads
        "price": rng.uniform(5, 500, n).round(2),        # nobody reads this
        "note": rng.choice(["não gostei", "ótimo"], n),  # nobody reads this
    })


class CountingStub:
    """Score = x. Counts how often it is called."""
    name, channels = "stub", ["x"]

    def __init__(self):
        self.calls = 0

    def score(self, df):
        self.calls += 1
        return pd.to_numeric(df.get("x"), errors="coerce").fillna(0).to_numpy(dtype=float) \
            if "x" in df else np.zeros(len(df))


ROLES = ColumnRoles.of(protected={"y"}, monetary={"price"}, text={"note"})


def test_auroc_basics():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


# Step 2a
def test_balanced_sample(probe):
    s = balanced_sample(probe, "y", 200, seed=1)
    assert len(s) == 200 and s["y"].sum() == 100
    assert list(s.index) == list(range(200))
    pd.testing.assert_frame_equal(s, balanced_sample(probe, "y", 200, seed=1))
    assert not s.equals(balanced_sample(probe, "y", 200, seed=2))
    with pytest.raises(ValueError):
        balanced_sample(probe.head(50), "y", 200, seed=1)


# Step 2b
def test_touches_channels(probe):
    clean = probe.head(20)
    assert not touches_channels(clean, clean.copy(), ["x"])
    other = clean.copy(); other["price"] *= 100
    assert not touches_channels(clean, other, ["x"])
    hit = clean.copy(); hit.loc[0, "x"] = np.nan
    assert touches_channels(clean, hit, ["x"])
    assert touches_channels(clean, clean.rename(columns={"x": "x_v2"}), ["x"])
    assert touches_channels(clean, pd.concat([clean, clean.head(2)]), ["x"])


# Step 2c
def test_profile_shape_and_skips(probe):
    stub = CountingStub()
    p = profile_sensitivity(stub, probe, "y", roles=ROLES, severities=(0.1, 0.3), seeds=2)
    assert set(p.families) == {"missing_injection", "mojibake_roundtrip", "category_taxonomy_drift",
                               "currency_unit_drift", "schema_drift", "type_flip"}
    assert p.clean_auroc > 0.9
    # currency drift and mojibake only touch columns the stub doesn't read
    for fam in ("currency_unit_drift", "mojibake_roundtrip"):
        assert not p.families[fam].touches_channels
        assert p.families[fam].mean_drop == [0.0, 0.0]
    # 2 seeds x (1 clean + at most 4 touching families x 2 severities); skipped families cost nothing
    assert stub.calls <= 2 * (1 + 4 * 2)


def test_missing_hurts_more_at_higher_severity(probe):
    p = profile_sensitivity(CountingStub(), probe, "y", roles=ROLES,
                            families=["missing_injection"], severities=(0.1, 0.5), seeds=3)
    r = p.families["missing_injection"]
    assert r.touches_channels and 0 < r.mean_drop[0] < r.mean_drop[1]
    assert r.slope > 0
    assert all(lo <= m <= hi for lo, m, hi in zip(r.ci_low, r.mean_drop, r.ci_high))


def test_deterministic(probe):
    a = profile_sensitivity(CountingStub(), probe, "y", roles=ROLES, families=["missing_injection"])
    b = profile_sensitivity(CountingStub(), probe, "y", roles=ROLES, families=["missing_injection"])
    assert a == b


# Step 2d
def test_json_roundtrip(probe, tmp_path):
    p = profile_sensitivity(CountingStub(), probe, "y", roles=ROLES, families=["missing_injection"])
    p.to_json(tmp_path / "p.json")
    assert SensitivityProfile.from_json(tmp_path / "p.json") == p
