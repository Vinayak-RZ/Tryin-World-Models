"""Compact Gaussian RSSM for vector plant observations. Imagination only — no actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sandbox.mode2_imagine.generate import ActuatedTrajectory, observation_features

OBS_DIM = 6
ACTION_DIM = 1


class _MLP(nn.Module):
    def __init__(self, ins: int, outs: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ins, hidden), nn.ELU(), nn.Linear(hidden, outs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RSSM(nn.Module):
    def __init__(self, hidden: int = 32, latent: int = 8) -> None:
        super().__init__()
        self.hidden = hidden
        self.latent = latent
        self.encoder = _MLP(OBS_DIM, hidden, hidden)
        self.gru = nn.GRUCell(latent + ACTION_DIM, hidden)
        self.prior_net = _MLP(hidden, 2 * latent, hidden)
        self.post_net = _MLP(hidden + hidden, 2 * latent, hidden)
        self.decoder = _MLP(hidden + latent, 1, hidden)

    def _split(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mean + std * torch.randn_like(mean)

    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced posterior rollout. obs/action: (T, B, *)."""
        t_steps, batch, _ = obs.shape
        h = obs.new_zeros(batch, self.hidden)
        z = obs.new_zeros(batch, self.latent)
        posts, priors, decodes, kls, next_decs = [], [], [], [], []
        for t in range(t_steps):
            h = self.gru(torch.cat([z, action[t]], dim=-1), h)
            prior_m, prior_s = self._split(self.prior_net(h))
            embed = self.encoder(obs[t])
            post_m, post_s = self._split(self.post_net(torch.cat([h, embed], dim=-1)))
            z = self._sample(post_m, post_s)
            load_hat = self.decoder(torch.cat([h, z], dim=-1)).squeeze(-1)
            kl = _kl_gauss(post_m, post_s, prior_m, prior_s)
            posts.append(z)
            priors.append(self._sample(prior_m, prior_s))
            decodes.append(load_hat)
            kls.append(kl)
            h_next = self.gru(torch.cat([z, action[t]], dim=-1), h)
            next_m, _ = self._split(self.prior_net(h_next))
            next_decs.append(self.decoder(torch.cat([h_next, next_m], dim=-1)).squeeze(-1))
        return torch.stack(posts), torch.stack(priors), torch.stack(decodes), torch.stack(kls), torch.stack(next_decs), h

    def imagine_from(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        actions: torch.Tensor,
        start_obs: torch.Tensor,
        load_mean: float,
        load_std: float,
        exo_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Open-loop: actions (H, B, 1), exo_obs (H, B, OBS_DIM) with load column overwritten."""
        horizon = actions.shape[0]
        loads = []
        obs = start_obs
        for t in range(horizon):
            h = self.gru(torch.cat([z, actions[t]], dim=-1), h)
            prior_m, prior_s = self._split(self.prior_net(h))
            z = prior_m
            load_hat = self.decoder(torch.cat([h, z], dim=-1)).squeeze(-1)
            loads.append(load_hat)
            next_obs = exo_obs[t].clone()
            next_obs[:, 0] = load_hat
            obs = next_obs
            embed = self.encoder(obs)
            # keep z as prior mean; embed unused in imagination (prior only)
            del embed
        return torch.stack(loads)


def _kl_gauss(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2 * var_p) - 0.5
    return kl.sum(dim=-1)


@dataclass
class RSSMWorldModel:
    """Trainable RSSM wrapper. Predicts load kW. No rupee fields."""

    hidden: int = 32
    latent: int = 8
    seq_len: int = 32
    batch_size: int = 16
    epochs: int = 8
    lr: float = 3e-3
    kl_w: float = 0.1
    device: str = "cpu"
    model: RSSM | None = None
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
        starts = idx[idx + self.seq_len < feats.shape[0]]
        starts = starts[mask[starts + self.seq_len - 1]]
        starts = starts[::2]
        if starts.size == 0:
            raise RuntimeError("no RSSM training windows")
        obs = np.stack([feats[s : s + self.seq_len] for s in starts])
        act = np.stack([action[s : s + self.seq_len, None] for s in starts])
        y = np.stack([load[s : s + self.seq_len] for s in starts])
        return obs, act, y

    def fit(self, traj: ActuatedTrajectory) -> RSSMWorldModel:
        train_load = traj.load_kw[traj.train_mask]
        train_temp = traj.temp_c[traj.train_mask]
        self.load_mean = float(train_load.mean())
        self.load_std = float(train_load.std()) or 1.0
        self.temp_mean = float(train_temp.mean())
        self.temp_std = float(train_temp.std()) or 1.0
        feats = self._norm_obs(traj)
        obs_np, act_np, y_np = self._windows(feats, traj.action, traj.load_kw, traj.train_mask)
        y_np = (y_np - self.load_mean) / self.load_std
        self.model = RSSM(self.hidden, self.latent).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = obs_np.shape[0]
        self.model.train()
        for _ in range(self.epochs):
            perm = np.random.default_rng(0).permutation(n)
            for i in range(0, n, self.batch_size):
                sl = perm[i : i + self.batch_size]
                obs = torch.tensor(obs_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                act = torch.tensor(act_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                y = torch.tensor(y_np[sl], dtype=torch.float32, device=self.device).transpose(0, 1)
                _, _, dec, kl, next_dec, _ = self.model.observe(obs, act)
                rec = F.mse_loss(dec, y)
                if next_dec.shape[0] > 1:
                    rec = rec + F.mse_loss(next_dec[:-1], y[1:])
                loss = rec + self.kl_w * kl.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model.eval()
        return self

    def _warmup(self, traj: ActuatedTrajectory, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("RSSMWorldModel.fit() was not called")
        start = max(0, end - self.seq_len + 1)
        feats = self._norm_obs(traj)[start : end + 1]
        act = traj.action[start : end + 1, None]
        obs = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(1)
        action = torch.tensor(act, dtype=torch.float32, device=self.device).unsqueeze(1)
        posts, _, _, _, _, h = self.model.observe(obs, action)
        return h, posts[-1], obs[-1]

    @torch.no_grad()
    def predict_1step(self, traj: ActuatedTrajectory) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RSSMWorldModel.fit() was not called")
        feats = self._norm_obs(traj)
        obs = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(1)
        act = torch.tensor(traj.action[:, None], dtype=torch.float32, device=self.device).unsqueeze(1)
        _, _, _, _, next_dec, _ = self.model.observe(obs, act)
        pred = next_dec.squeeze(1).cpu().numpy() * self.load_std + self.load_mean
        out = np.empty(traj.n_steps, dtype=np.float64)
        out[0] = traj.load_kw[0]
        out[1:] = pred[:-1]
        return out

    @torch.no_grad()
    def imagine(self, traj: ActuatedTrajectory, start: int, horizon: int, action: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RSSMWorldModel.fit() was not called")
        h, z, last_obs = self._warmup(traj, start)
        exo = self._norm_obs(traj)
        exo_t = torch.tensor(exo[start : start + horizon], dtype=torch.float32, device=self.device).unsqueeze(1)
        act = torch.tensor(action[start : start + horizon, None], dtype=torch.float32, device=self.device).unsqueeze(1)
        loads = self.model.imagine_from(h, z, act, last_obs, self.load_mean, self.load_std, exo_t)
        return loads.squeeze(1).cpu().numpy() * self.load_std + self.load_mean
