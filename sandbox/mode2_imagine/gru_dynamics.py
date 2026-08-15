"""Deterministic action-conditioned GRU world model. Second model for agreement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sandbox.mode2_imagine.generate import ActuatedTrajectory, observation_features

OBS_DIM = 6
ACTION_DIM = 1


class GRUDynamics(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ELU())
        self.gru = nn.GRUCell(hidden + ACTION_DIM, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """obs/action (T, B, *). Predicts next-step normalized load at each t."""
        t_steps, batch, _ = obs.shape
        h = obs.new_zeros(batch, self.hidden)
        preds = []
        for t in range(t_steps):
            z = self.encoder(obs[t])
            h = self.gru(torch.cat([z, action[t]], dim=-1), h)
            preds.append(self.decoder(h).squeeze(-1))
        return torch.stack(preds)

    def imagine_from(
        self,
        h: torch.Tensor,
        actions: torch.Tensor,
        start_obs: torch.Tensor,
        exo_obs: torch.Tensor,
    ) -> torch.Tensor:
        horizon = actions.shape[0]
        loads = []
        obs = start_obs
        for t in range(horizon):
            z = self.encoder(obs)
            h = self.gru(torch.cat([z, actions[t]], dim=-1), h)
            load_hat = self.decoder(h).squeeze(-1)
            loads.append(load_hat)
            next_obs = exo_obs[t].clone()
            next_obs[:, 0] = load_hat
            obs = next_obs
        return torch.stack(loads)


@dataclass
class GRUWorldModel:
    hidden: int = 32
    seq_len: int = 32
    batch_size: int = 16
    epochs: int = 8
    lr: float = 3e-3
    device: str = "cpu"
    model: GRUDynamics | None = None
    load_mean: float = 0.0
    load_std: float = 1.0
    temp_mean: float = 0.0
    temp_std: float = 1.0

    def _norm_obs(self, traj: ActuatedTrajectory, load: np.ndarray | None = None) -> np.ndarray:
        feats = observation_features(traj, load=load)
        feats[:, 0] = (feats[:, 0] - self.load_mean) / self.load_std
        feats[:, 1] = (feats[:, 1] - self.temp_mean) / self.temp_std
        return feats

    def _windows(self, feats: np.ndarray, action: np.ndarray, load: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.flatnonzero(mask)
        starts = idx[idx + self.seq_len < feats.shape[0] - 1]
        starts = starts[mask[starts + self.seq_len]]
        starts = starts[::2]
        if starts.size == 0:
            raise RuntimeError("no GRU training windows")
        obs = np.stack([feats[s : s + self.seq_len] for s in starts])
        act = np.stack([action[s : s + self.seq_len, None] for s in starts])
        y = np.stack([load[s + 1 : s + self.seq_len + 1] for s in starts])
        return obs, act, y

    def fit(self, traj: ActuatedTrajectory) -> GRUWorldModel:
        train_load = traj.load_kw[traj.train_mask]
        train_temp = traj.temp_c[traj.train_mask]
        self.load_mean = float(train_load.mean())
        self.load_std = float(train_load.std()) or 1.0
        self.temp_mean = float(train_temp.mean())
        self.temp_std = float(train_temp.std()) or 1.0
        feats = self._norm_obs(traj)
        obs_np, act_np, y_np = self._windows(feats, traj.action, traj.load_kw, traj.train_mask)
        y_np = (y_np - self.load_mean) / self.load_std
        self.model = GRUDynamics(self.hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = obs_np.shape[0]
        self.model.train()
        for _ in range(self.epochs):
            perm = np.random.default_rng(1).permutation(n)
            for i in range(0, n, self.batch_size):
                sl = perm[i : i + self.batch_size]
                obs = torch.tensor(obs_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                act = torch.tensor(act_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                y = torch.tensor(y_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                pred = self.model(obs, act)
                loss = F.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model.eval()
        return self

    def _warmup(self, traj: ActuatedTrajectory, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("GRUWorldModel.fit() was not called")
        start = max(0, end - self.seq_len)
        feats = self._norm_obs(traj)
        hist = feats[start:end]
        act = traj.action[start:end, None]
        obs = torch.tensor(hist, dtype=torch.float32, device=self.device).unsqueeze(1)
        action = torch.tensor(act, dtype=torch.float32, device=self.device).unsqueeze(1)
        h = obs.new_zeros(1, self.model.hidden)
        for t in range(obs.shape[0]):
            z = self.model.encoder(obs[t])
            h = self.model.gru(torch.cat([z, action[t]], dim=-1), h)
        start_obs = torch.tensor(feats[end], dtype=torch.float32, device=self.device).unsqueeze(0)
        return h, start_obs

    @torch.no_grad()
    def predict_1step(self, traj: ActuatedTrajectory) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GRUWorldModel.fit() was not called")
        feats = self._norm_obs(traj)
        obs = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(1)
        act = torch.tensor(traj.action[:, None], dtype=torch.float32, device=self.device).unsqueeze(1)
        pred = self.model(obs, act).squeeze(1).cpu().numpy() * self.load_std + self.load_mean
        out = np.empty(traj.n_steps, dtype=np.float64)
        out[0] = traj.load_kw[0]
        out[1:] = pred[:-1]
        return out

    @torch.no_grad()
    def imagine(self, traj: ActuatedTrajectory, start: int, horizon: int, action: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GRUWorldModel.fit() was not called")
        h, last_obs = self._warmup(traj, start)
        exo = self._norm_obs(traj)
        exo_t = torch.tensor(exo[start : start + horizon], dtype=torch.float32, device=self.device).unsqueeze(1)
        act = torch.tensor(action[start : start + horizon, None], dtype=torch.float32, device=self.device).unsqueeze(1)
        loads = self.model.imagine_from(h, act, last_obs, exo_t)
        return loads.squeeze(1).cpu().numpy() * self.load_std + self.load_mean
