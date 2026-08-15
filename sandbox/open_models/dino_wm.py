"""DINO-WM-style challenger: frozen vision encoder + a plant-specific predictor head.

Pretrained DINO-WM checkpoints (PushT / PointMaze / Wall) are pixel-robot models.
They are not used here. We reuse the recipe: freeze DINO, train a head on plant frames.

Default encoder is DINOv2-small (open, CPU-feasible). DINOv3 is the same adapter
when weights and a GPU are available — see research/maps notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge

from sandbox.mode1_anomaly.champion import ewma
from sandbox.mode1_anomaly.generate import Trajectory
from sandbox.open_models.render import (
    DEFAULT_LOOKBACK,
    DEFAULT_SIZE,
    render_indices,
    render_trajectory_frames,
)

DEFAULT_ENCODER_ID = "facebook/dinov2-small"


class ImageEncoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Return (N, D) float embeddings for RGB uint8 images."""


class FakeImageEncoder:
    """Deterministic stand-in so tests do not download weights."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float64)
        for i, img in enumerate(images):
            x = img.astype(np.float64).reshape(-1)
            if x.size == 0:
                continue
            step = max(1, x.size // self.dim)
            chunk = x[::step][: self.dim]
            out[i, : chunk.size] = chunk / 255.0
            out[i] += img.mean() / 255.0
        return out


class HuggingFaceDinoEncoder:
    """Frozen Hugging Face DINO / DINOv2 / DINOv3 encoder (CLS token)."""

    def __init__(self, model_id: str = DEFAULT_ENCODER_ID, device: str = "cpu", batch_size: int = 8) -> None:
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self.device = device
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._torch = torch

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        rows: list[np.ndarray] = []
        torch = self._torch
        for start in range(0, len(images), self.batch_size):
            batch = [Image.fromarray(img) for img in images[start : start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            rows.append(hidden[:, 0].float().cpu().numpy())
        if not rows:
            return np.zeros((0, 0), dtype=np.float64)
        return np.concatenate(rows, axis=0)


def _fill_scores(n_steps: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(n_steps, dtype=np.float64)
    return np.interp(np.arange(n_steps), indices.astype(np.float64), values.astype(np.float64))


@dataclass
class DinoWmSurpriseDetector:
    """Frozen DINO embeddings + Ridge z_t -> z_{t+1}. Head is the only trained piece."""

    encoder: ImageEncoder = field(default_factory=FakeImageEncoder)
    encoder_id: str = "fake"
    lookback: int = DEFAULT_LOOKBACK
    stride: int = 8
    image_size: int = DEFAULT_SIZE
    ridge_alpha: float = 1.0
    ewma_lam: float = 0.25
    ridge: Ridge | None = None
    train_score_quantile: float = 0.0

    def _embed(self, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
        idx = render_indices(traj.n_steps, self.lookback, self.stride)
        frames = render_trajectory_frames(traj, idx, lookback=self.lookback, size=self.image_size)
        z = self.encoder.encode(frames)
        return idx, z

    def fit(self, traj: Trajectory) -> DinoWmSurpriseDetector:
        idx, z = self._embed(traj)
        in_train = traj.train_mask[idx]
        pair = np.flatnonzero(in_train[:-1] & in_train[1:])
        if pair.size < 8:
            raise RuntimeError("not enough DINO train pairs")
        self.ridge = Ridge(alpha=self.ridge_alpha).fit(z[pair], z[pair + 1])
        scores = self.score(traj)
        self.train_score_quantile = float(np.quantile(scores[traj.train_mask], 0.98))
        return self

    def score(self, traj: Trajectory) -> np.ndarray:
        if self.ridge is None:
            raise RuntimeError("DinoWmSurpriseDetector.fit() was not called")
        idx, z = self._embed(traj)
        pred = self.ridge.predict(z)
        err = np.zeros(idx.shape[0], dtype=np.float64)
        err[1:] = np.linalg.norm(z[1:] - pred[:-1], axis=1)
        if err.shape[0] > 1:
            err[0] = err[1]
        return ewma(_fill_scores(traj.n_steps, idx, err), self.ewma_lam)

    def alarms(self, traj: Trajectory, threshold: float | None = None) -> np.ndarray:
        cut = self.train_score_quantile if threshold is None else threshold
        return self.score(traj) >= cut
