"""Chronos-Bolt surprise detector. Time-series foundation cousin (pack Mode 6).

Not a world model. Included because it is a real open pretrained model that
accepts 15-minute load without pretending a robot-pixel checkpoint transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sandbox.mode1_anomaly.champion import ewma
from sandbox.mode1_anomaly.generate import Trajectory

DEFAULT_MODEL_ID = "amazon/chronos-bolt-tiny"


class ForecastBackend(Protocol):
    def predict_median(self, contexts: list[np.ndarray], horizon: int = 1) -> np.ndarray:
        """Return (N,) median forecasts for the next step after each context."""


class FakeForecastBackend:
    """Last-value predictor so tests do not download Chronos."""

    def predict_median(self, contexts: list[np.ndarray], horizon: int = 1) -> np.ndarray:
        del horizon
        return np.array([float(ctx[-1]) if ctx.size else 0.0 for ctx in contexts], dtype=np.float64)


class ChronosBoltBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cpu", batch_size: int = 16) -> None:
        import torch
        from chronos import BaseChronosPipeline

        self.batch_size = batch_size
        self.torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float32,
        )

    def predict_median(self, contexts: list[np.ndarray], horizon: int = 1) -> np.ndarray:
        preds: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            batch = [self.torch.tensor(ctx, dtype=self.torch.float32) for ctx in contexts[start : start + self.batch_size]]
            forecast = self.pipeline.predict(batch, prediction_length=horizon)
            # Chronos-Bolt: [N, quantiles, horizon]. Use middle quantile as median.
            arr = forecast.detach().cpu().numpy() if hasattr(forecast, "detach") else np.asarray(forecast)
            if arr.ndim == 3:
                mid = arr.shape[1] // 2
                step = arr[:, mid, 0]
            else:
                step = arr.reshape(arr.shape[0], -1)[:, 0]
            preds.extend(float(v) for v in step)
        return np.asarray(preds, dtype=np.float64)


def _fill_scores(n_steps: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(n_steps, dtype=np.float64)
    return np.interp(np.arange(n_steps), indices.astype(np.float64), values.astype(np.float64))


@dataclass
class ChronosSurpriseDetector:
    backend: ForecastBackend = field(default_factory=FakeForecastBackend)
    model_id: str = "fake"
    context_len: int = 96 * 7
    stride: int = 8
    ewma_lam: float = 0.25
    train_score_quantile: float = 0.0

    def _eval_indices(self, n_steps: int) -> np.ndarray:
        first = self.context_len
        if first >= n_steps:
            return np.array([], dtype=np.int32)
        return np.arange(first, n_steps, self.stride, dtype=np.int32)

    def _raw_surprise(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
        idx = self._eval_indices(traj.n_steps)
        contexts = [traj.load_kw[int(i) - self.context_len : int(i)] for i in idx]
        pred = self.backend.predict_median(contexts, horizon=1)
        actual = traj.load_kw[idx]
        return idx, np.abs(actual - pred)

    def fit(self, traj: Trajectory) -> ChronosSurpriseDetector:
        scores = self.score(traj)
        self.train_score_quantile = float(np.quantile(scores[traj.train_mask], 0.98))
        return self

    def score(self, traj: Trajectory) -> np.ndarray:
        idx, err = self._raw_surprise(traj)
        return ewma(_fill_scores(traj.n_steps, idx, err), self.ewma_lam)

    def alarms(self, traj: Trajectory, threshold: float | None = None) -> np.ndarray:
        cut = self.train_score_quantile if threshold is None else threshold
        return self.score(traj) >= cut
