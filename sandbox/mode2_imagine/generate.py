"""Synthetic plant whose next load depends on a process setpoint action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STEPS_PER_DAY = 96
DAYS_PER_WEEK = 7
STEPS_PER_WEEK = STEPS_PER_DAY * DAYS_PER_WEEK


@dataclass(frozen=True)
class PlantConfig:
    plant_id: str
    n_weeks: int = 12
    train_weeks: int = 8
    base_kw: float = 120.0
    weekend_scale: float = 0.72
    temp_beta: float = 1.8
    noise_std: float = 2.5
    action_floor: float = 0.35
    lag: float = 0.35


@dataclass
class ActuatedTrajectory:
    plant_id: str
    load_kw: np.ndarray
    temp_c: np.ndarray
    action: np.ndarray
    interval: np.ndarray
    dow: np.ndarray
    hour: np.ndarray
    noise: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    config: PlantConfig

    @property
    def n_steps(self) -> int:
        return int(self.load_kw.shape[0])


def weekly_shape(interval: np.ndarray, dow: np.ndarray, weekend_scale: float) -> np.ndarray:
    hour_frac = interval / STEPS_PER_DAY
    diurnal = 0.55 + 0.45 * np.sin(2 * np.pi * (hour_frac - 0.25))
    occupancy = np.where(dow >= 5, weekend_scale, 1.0)
    return diurnal * occupancy


def _temperature(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_steps)
    diurnal = 8.0 * np.sin(2 * np.pi * (t % STEPS_PER_DAY) / STEPS_PER_DAY - np.pi / 2)
    slow = 3.0 * np.sin(2 * np.pi * t / (STEPS_PER_WEEK * 6))
    return 24.0 + diurnal + slow + rng.normal(0.0, 1.0, size=n_steps)


def _setpoint_schedule(n_steps: int, rng: np.random.Generator, hold: int = 16) -> np.ndarray:
    """Piecewise-constant process setpoint in [0.2, 1.0]."""
    n_holds = int(np.ceil(n_steps / hold))
    levels = rng.uniform(0.2, 1.0, size=n_holds)
    action = np.repeat(levels, hold)[:n_steps]
    return action.astype(np.float64)


def step_load(
    prev_load: float,
    action: float,
    temp_c: float,
    shape: float,
    config: PlantConfig,
    noise: float,
) -> float:
    """One 15-min plant step. Setpoint scales the production-driven load."""
    drive = config.action_floor + (1.0 - config.action_floor) * float(np.clip(action, 0.0, 1.0))
    target = config.base_kw * shape * drive + config.temp_beta * (temp_c - 24.0)
    return (1.0 - config.lag) * target + config.lag * prev_load + noise


def simulate_load(
    config: PlantConfig,
    action: np.ndarray,
    temp_c: np.ndarray,
    interval: np.ndarray,
    dow: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    shape = weekly_shape(interval, dow, config.weekend_scale)
    load = np.empty(action.shape[0], dtype=np.float64)
    prev = config.base_kw * (config.action_floor + 0.3)
    for t in range(action.shape[0]):
        prev = step_load(prev, float(action[t]), float(temp_c[t]), float(shape[t]), config, float(noise[t]))
        load[t] = prev
    return load


def generate_plant(config: PlantConfig, seed: int) -> ActuatedTrajectory:
    n_steps = config.n_weeks * STEPS_PER_WEEK
    train_end = config.train_weeks * STEPS_PER_WEEK
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)
    interval = (steps % STEPS_PER_DAY).astype(np.int32)
    dow = ((steps // STEPS_PER_DAY) % DAYS_PER_WEEK).astype(np.int32)
    hour = (interval // 4).astype(np.int32)
    temp = _temperature(n_steps, rng)
    action = _setpoint_schedule(n_steps, rng)
    noise = rng.normal(0.0, config.noise_std, size=n_steps)
    load = simulate_load(config, action, temp, interval, dow, noise)
    train_mask = steps < train_end
    return ActuatedTrajectory(
        plant_id=config.plant_id,
        load_kw=load,
        temp_c=temp.astype(np.float64),
        action=action,
        interval=interval,
        dow=dow,
        hour=hour,
        noise=noise.astype(np.float64),
        train_mask=train_mask,
        test_mask=~train_mask,
        config=config,
    )


def resimulate(traj: ActuatedTrajectory, action: np.ndarray) -> np.ndarray:
    """Replay the same weather and noise under a different setpoint path."""
    if action.shape != traj.action.shape:
        raise ValueError("action length must match the trajectory")
    return simulate_load(traj.config, action, traj.temp_c, traj.interval, traj.dow, traj.noise)


def counterfactual_action(traj: ActuatedTrajectory, scale: float = 0.5) -> np.ndarray:
    """Alternate setpoint: scale the factual action, keep it in [0.2, 1.0]."""
    return np.clip(traj.action * scale, 0.2, 1.0)


def default_plants() -> tuple[PlantConfig, PlantConfig]:
    return (
        PlantConfig(plant_id="plant_a", base_kw=140.0, weekend_scale=0.68, temp_beta=2.2, noise_std=2.2),
        PlantConfig(plant_id="plant_b", base_kw=95.0, weekend_scale=0.85, temp_beta=0.9, noise_std=2.8),
    )


def observation_features(traj: ActuatedTrajectory, load: np.ndarray | None = None) -> np.ndarray:
    """State vector: load, temp, hour/dow Fourier features. Action is separate."""
    y = traj.load_kw if load is None else load
    hour_ang = 2.0 * np.pi * traj.hour / 24.0
    dow_ang = 2.0 * np.pi * traj.dow / 7.0
    return np.column_stack(
        [
            y,
            traj.temp_c,
            np.sin(hour_ang),
            np.cos(hour_ang),
            np.sin(dow_ang),
            np.cos(dow_ang),
        ]
    ).astype(np.float64)
