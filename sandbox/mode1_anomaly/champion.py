"""TOW-P-shaped baseline plus EWMA/CUSUM residual SPC. Not CalTRACK-certified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sandbox.mode1_anomaly.generate import STEPS_PER_DAY, Trajectory

N_TOW_BINS = STEPS_PER_DAY * 7
MIN_BIN_POINTS = 8


def tow_bin(dow: np.ndarray, interval: np.ndarray) -> np.ndarray:
    return dow.astype(np.int32) * STEPS_PER_DAY + interval.astype(np.int32)


class TimeOfWeekTempBaseline:
    """Per time-of-week bin: load ~ a + b * temp, else the bin mean."""

    def __init__(self) -> None:
        self.intercept = np.zeros(N_TOW_BINS, dtype=np.float64)
        self.slope = np.zeros(N_TOW_BINS, dtype=np.float64)
        self.fitted = np.zeros(N_TOW_BINS, dtype=bool)

    def fit(self, traj: Trajectory) -> TimeOfWeekTempBaseline:
        bins = tow_bin(traj.dow, traj.interval)
        train = traj.train_mask
        for b in range(N_TOW_BINS):
            sel = train & (bins == b)
            if int(sel.sum()) < MIN_BIN_POINTS:
                continue
            y = traj.load_kw[sel]
            x = traj.temp_c[sel]
            x_c = x - x.mean()
            denom = float(np.dot(x_c, x_c))
            slope = float(np.dot(x_c, y - y.mean()) / denom) if denom > 1e-9 else 0.0
            intercept = float(y.mean() - slope * x.mean())
            self.intercept[b] = intercept
            self.slope[b] = slope
            self.fitted[b] = True
        if not self.fitted.any():
            raise RuntimeError("TOW-P baseline: no bin had enough train points")
        return self

    def predict(self, traj: Trajectory) -> np.ndarray:
        bins = tow_bin(traj.dow, traj.interval)
        pred = self.intercept[bins] + self.slope[bins] * traj.temp_c
        fallback = float(traj.load_kw[traj.train_mask].mean())
        pred = np.where(self.fitted[bins], pred, fallback)
        return pred.astype(np.float64)


def ewma(values: np.ndarray, lam: float) -> np.ndarray:
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    one_m = 1.0 - lam
    for i in range(1, values.shape[0]):
        out[i] = lam * values[i] + one_m * out[i - 1]
    return out


def two_sided_cusum(values: np.ndarray, k: float) -> np.ndarray:
    """Return max(S+, S-) at each step."""
    s_pos = 0.0
    s_neg = 0.0
    out = np.empty(values.shape[0], dtype=np.float64)
    for i, v in enumerate(values):
        s_pos = max(0.0, s_pos + v - k)
        s_neg = max(0.0, s_neg - v - k)
        out[i] = max(s_pos, s_neg)
    return out


def windowed_cusum(values: np.ndarray, k: float, window: int) -> np.ndarray:
    """CUSUM on a trailing window so a finished shift does not latch forever."""
    if window < 2:
        raise ValueError("window must be >= 2")
    out = np.empty(values.shape[0], dtype=np.float64)
    for i in range(values.shape[0]):
        start = max(0, i - window + 1)
        out[i] = two_sided_cusum(values[start : i + 1], k)[-1]
    return out


@dataclass
class ChampionDetector:
    """Residual SPC: rank on |EWMA|; alarm if EWMA or CUSUM exceeds its train cut."""

    lam: float = 0.25
    cusum_k_sigma: float = 0.5
    cusum_window: int = 48
    baseline: TimeOfWeekTempBaseline | None = None
    residual_std: float = 1.0
    ewma_cut: float = 0.0
    cusum_cut: float = 5.0
    train_score_quantile: float = 0.0

    def _standardized_residual(self, traj: Trajectory) -> np.ndarray:
        if self.baseline is None:
            raise RuntimeError("ChampionDetector.fit() was not called")
        residual = traj.load_kw - self.baseline.predict(traj)
        return residual / self.residual_std

    def fit(self, traj: Trajectory) -> ChampionDetector:
        self.baseline = TimeOfWeekTempBaseline().fit(traj)
        train_r = (traj.load_kw - self.baseline.predict(traj))[traj.train_mask]
        self.residual_std = float(np.std(train_r)) or 1.0
        z = self._standardized_residual(traj)
        train_z = z[traj.train_mask]
        ewma_abs = np.abs(ewma(train_z, self.lam))
        cusum = windowed_cusum(train_z, k=self.cusum_k_sigma, window=self.cusum_window)
        self.ewma_cut = float(np.quantile(ewma_abs, 0.98))
        self.cusum_cut = max(float(np.quantile(cusum, 0.98)), 5.0)
        self.train_score_quantile = self.ewma_cut
        return self

    def score(self, traj: Trajectory) -> np.ndarray:
        z = self._standardized_residual(traj)
        return np.abs(ewma(z, self.lam))

    def alarms(self, traj: Trajectory, threshold: float | None = None) -> np.ndarray:
        z = self._standardized_residual(traj)
        ewma_abs = np.abs(ewma(z, self.lam))
        cusum = windowed_cusum(z, k=self.cusum_k_sigma, window=self.cusum_window)
        ewma_cut = self.ewma_cut if threshold is None else threshold
        return (ewma_abs >= ewma_cut) | (cusum >= self.cusum_cut)
