"""Next-step latent predictor. Residual-surprise proxy, not JEPA/RSSM/Dreamer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sandbox.mode1_anomaly.champion import ewma
from sandbox.mode1_anomaly.generate import Trajectory

LATENT_DIM = 4


def observation_features(traj: Trajectory) -> np.ndarray:
    hour_ang = 2.0 * np.pi * traj.hour / 24.0
    dow_ang = 2.0 * np.pi * traj.dow / 7.0
    return np.column_stack(
        [
            traj.load_kw,
            traj.temp_c,
            np.sin(hour_ang),
            np.cos(hour_ang),
            np.sin(dow_ang),
            np.cos(dow_ang),
        ]
    )


@dataclass
class NextStepSurpriseDetector:
    """PCA state + Ridge z_t -> z_{t+1}. Surprise is next-step latent error."""

    ridge_alpha: float = 1.0
    ewma_lam: float = 0.25
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    ridge: Ridge | None = None
    train_score_quantile: float = 0.0

    def fit(self, traj: Trajectory) -> NextStepSurpriseDetector:
        feats = observation_features(traj)
        train_idx = np.flatnonzero(traj.train_mask)
        # Need consecutive train pairs.
        pair_t = train_idx[:-1]
        pair_t = pair_t[pair_t + 1 <= train_idx[-1]]
        pair_t = pair_t[np.isin(pair_t + 1, train_idx)]
        self.scaler = StandardScaler().fit(feats[train_idx])
        scaled = self.scaler.transform(feats)
        n_comp = min(LATENT_DIM, scaled.shape[1], max(1, train_idx.shape[0] - 1))
        self.pca = PCA(n_components=n_comp, random_state=0).fit(scaled[train_idx])
        z = self.pca.transform(scaled)
        self.ridge = Ridge(alpha=self.ridge_alpha).fit(z[pair_t], z[pair_t + 1])
        scores = self.score(traj)
        self.train_score_quantile = float(np.quantile(scores[traj.train_mask], 0.98))
        return self

    def _latent(self, traj: Trajectory) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError("NextStepSurpriseDetector.fit() was not called")
        return self.pca.transform(self.scaler.transform(observation_features(traj)))

    def score(self, traj: Trajectory) -> np.ndarray:
        if self.ridge is None:
            raise RuntimeError("NextStepSurpriseDetector.fit() was not called")
        z = self._latent(traj)
        pred = self.ridge.predict(z)
        surprise = np.zeros(traj.n_steps, dtype=np.float64)
        err = np.linalg.norm(z[1:] - pred[:-1], axis=1)
        surprise[1:] = err
        surprise[0] = surprise[1] if traj.n_steps > 1 else 0.0
        return ewma(surprise, self.ewma_lam)

    def alarms(self, traj: Trajectory, threshold: float | None = None) -> np.ndarray:
        cut = self.train_score_quantile if threshold is None else threshold
        return self.score(traj) >= cut
