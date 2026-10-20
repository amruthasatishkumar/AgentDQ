import json

import numpy as np
import pandas as pd

from agentdq import Consumer, LLMConsumer, SklearnConsumer


class _Stub:
    """Stands in for a fitted scikit-learn model: P(positive) = x / 10."""

    def predict_proba(self, X):
        p = X["x"].fillna(5).to_numpy(dtype=float) / 10
        return np.column_stack([1 - p, p])


def test_sklearn_consumer_treats_missing_columns_as_nan():
    c = SklearnConsumer(_Stub(), channels=["x"])
    assert isinstance(c, Consumer)
    np.testing.assert_allclose(c.score(pd.DataFrame({"x": [2, 8]})), [0.2, 0.8])
    np.testing.assert_allclose(c.score(pd.DataFrame({"renamed": [2, 8]})), [0.5, 0.5])


def test_llm_consumer_retries_and_logs(tmp_path):
    calls = {"n": 0}

    def flaky(row):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return 0.9 if row["x"] > 5 else 0.1

    log = tmp_path / "calls.jsonl"
    c = LLMConsumer(flaky, channels=["x", "missing_col"], max_workers=1, backoff_s=0, log_path=log)
    scores = c.score(pd.DataFrame({"x": [1, 9]}))
    np.testing.assert_allclose(scores, [0.1, 0.9])
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(recs) == 2 and recs[0]["input"]["missing_col"] is None


def test_llm_consumer_falls_back_after_retries():
    c = LLMConsumer(lambda row: 1 / 0, channels=["x"], retries=2, backoff_s=0, fallback=0.5)
    np.testing.assert_allclose(c.score(pd.DataFrame({"x": [1]})), [0.5])
