"""Render a load+temp lookback window as an RGB frame for a frozen vision encoder."""

from __future__ import annotations

import numpy as np

from sandbox.mode1_anomaly.generate import Trajectory

DEFAULT_SIZE = 224
DEFAULT_LOOKBACK = 96


def _draw_polyline(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w, _ = img.shape
    for i in range(xs.shape[0] - 1):
        n = max(abs(int(xs[i + 1]) - int(xs[i])), abs(int(ys[i + 1]) - int(ys[i])), 1)
        x = np.linspace(xs[i], xs[i + 1], n + 1).astype(int)
        y = np.linspace(ys[i], ys[i + 1], n + 1).astype(int)
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        img[y, x] = color


def render_window(
    load_kw: np.ndarray,
    temp_c: np.ndarray,
    end_step: int,
    lookback: int = DEFAULT_LOOKBACK,
    size: int = DEFAULT_SIZE,
) -> np.ndarray:
    """Return a uint8 HxWx3 image. Green = load, blue = temperature."""
    if end_step < 1:
        raise ValueError("end_step must be >= 1")
    start = max(0, end_step - lookback)
    load = np.asarray(load_kw[start:end_step], dtype=np.float64)
    temp = np.asarray(temp_c[start:end_step], dtype=np.float64)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = 12
    xs = np.linspace(8, size - 9, load.shape[0])

    def series_ys(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min()) if values.size else 0.0
        norm = (values - values.min()) / (span + 1e-6) if values.size else values
        return (1.0 - norm) * (size - 17) + 8

    if load.size >= 2:
        _draw_polyline(img, xs, series_ys(load), (80, 220, 120))
    if temp.size >= 2:
        _draw_polyline(img, xs, series_ys(temp), (80, 140, 255))
    img[:, size - 3 : size - 1] = (200, 60, 60)
    return img


def render_indices(n_steps: int, lookback: int, stride: int) -> np.ndarray:
    first = lookback
    if first >= n_steps:
        return np.array([], dtype=np.int32)
    return np.arange(first, n_steps, stride, dtype=np.int32)


def render_trajectory_frames(
    traj: Trajectory,
    indices: np.ndarray,
    lookback: int = DEFAULT_LOOKBACK,
    size: int = DEFAULT_SIZE,
) -> list[np.ndarray]:
    return [render_window(traj.load_kw, traj.temp_c, int(i), lookback=lookback, size=size) for i in indices]
