"""Consumers: anything that reads a table and scores rows.

A consumer is an ML model or an LLM agent. agentdq treats both the same way so
their sensitivity to data defects can be compared on identical inputs.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Consumer(Protocol):
    name: str
    channels: list[str]  # columns this consumer reads

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """One score per row; higher means the positive class."""
        ...


class SklearnConsumer:
    """Wraps a fitted scikit-learn estimator or pipeline with predict_proba."""

    def __init__(self, model: Any, channels: list[str], name: str = "model"):
        self.model = model
        self.channels = list(channels)
        self.name = name

    def score(self, df: pd.DataFrame) -> np.ndarray:
        # Columns lost to schema drift reach the model as missing, as they would in production.
        X = df.reindex(columns=self.channels)
        return np.asarray(self.model.predict_proba(X))[:, 1]


def _row_dict(row: pd.Series, channels: list[str]) -> dict[str, Any]:
    out = {}
    for c in channels:
        v = row.get(c, None)
        if isinstance(v, (np.generic,)):
            v = v.item()
        out[c] = None if (v is not None and not isinstance(v, str) and pd.isna(v)) else v
    return out


class LLMConsumer:
    """Wraps any function row -> probability, e.g. a call to an LLM agent.

    `fn` receives a dict of the consumer's channels for one row (columns missing
    from the batch arrive as None) and returns a float in [0, 1]. Calls run
    concurrently with retries; each call is logged to `log_path` as JSONL.
    """

    def __init__(self, fn: Callable[[dict], float], channels: list[str], name: str = "agent",
                 max_workers: int = 8, retries: int = 3, backoff_s: float = 1.0,
                 log_path: str | Path | None = None, fallback: float = 0.5):
        self.fn = fn
        self.channels = list(channels)
        self.name = name
        self.max_workers = max_workers
        self.retries = retries
        self.backoff_s = backoff_s
        self.fallback = fallback
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        with self._lock, self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _call(self, row: dict) -> float:
        err = None
        for attempt in range(self.retries):
            try:
                p = float(self.fn(row))
                self._log({"consumer": self.name, "input": row, "score": p, "attempt": attempt})
                return min(1.0, max(0.0, p))
            except Exception as e:  # noqa: BLE001 - any provider error is retried
                err = e
                time.sleep(self.backoff_s * (2 ** attempt))
        self._log({"consumer": self.name, "input": row, "error": repr(err)})
        return self.fallback

    def score(self, df: pd.DataFrame) -> np.ndarray:
        rows = [_row_dict(r, self.channels) for _, r in df.iterrows()]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return np.array(list(pool.map(self._call, rows)))
