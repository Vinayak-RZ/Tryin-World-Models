"""Seeded synthetic 15-minute plant curves with labeled test events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STEPS_PER_DAY = 96
DAYS_PER_WEEK = 7
STEPS_PER_WEEK = STEPS_PER_DAY * DAYS_PER_WEEK

EVENT_DRIFT = "drift"
EVENT_SENSOR = "sensor_fault"
EVENT_UPSET = "process_upset"


@dataclass(frozen=True)
class EventSpec:
    kind: str
    start_step: int
    duration_steps: int
    magnitude_kw: float

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration_steps


@dataclass(frozen=True)
class PlantConfig:
    plant_id: str
    n_weeks: int = 12
    train_weeks: int = 8
    base_kw: float = 120.0
    weekend_scale: float = 0.72
    temp_beta: float = 1.8
    noise_std: float = 3.5
    events: tuple[EventSpec, ...] = field(default_factory=tuple)


@dataclass
class Trajectory:
    plant_id: str
    load_kw: np.ndarray
    temp_c: np.ndarray
    interval: np.ndarray
    dow: np.ndarray
    hour: np.ndarray
    event_mask: np.ndarray
    event_kind: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    events: tuple[EventSpec, ...]

    @property
    def n_steps(self) -> int:
        return int(self.load_kw.shape[0])


def _weekly_shape(interval: np.ndarray, dow: np.ndarray, weekend_scale: float) -> np.ndarray:
    """Daytime hump plus a weekday/weekend split."""
    hour_frac = interval / STEPS_PER_DAY
    diurnal = 0.55 + 0.45 * np.sin(2 * np.pi * (hour_frac - 0.25))
    is_weekend = dow >= 5
    occupancy = np.where(is_weekend, weekend_scale, 1.0)
    return diurnal * occupancy


def _temperature(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_steps)
    diurnal = 8.0 * np.sin(2 * np.pi * (t % STEPS_PER_DAY) / STEPS_PER_DAY - np.pi / 2)
    slow = 3.0 * np.sin(2 * np.pi * t / (STEPS_PER_WEEK * 6))
    return 24.0 + diurnal + slow + rng.normal(0.0, 1.2, size=n_steps)


def _apply_events(load: np.ndarray, events: tuple[EventSpec, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = load.copy()
    mask = np.zeros(load.shape[0], dtype=bool)
    kind = np.full(load.shape[0], "", dtype=object)
    for event in events:
        sl = slice(event.start_step, event.end_step)
        if event.kind == EVENT_SENSOR:
            observed[sl] = observed[sl] + event.magnitude_kw
        else:
            observed[sl] = observed[sl] + event.magnitude_kw
        mask[sl] = True
        kind[sl] = event.kind
    return observed, mask, kind


def generate_plant(config: PlantConfig, seed: int) -> Trajectory:
    """Build one plant. Events land only in the test window."""
    n_steps = config.n_weeks * STEPS_PER_WEEK
    train_end = config.train_weeks * STEPS_PER_WEEK
    for event in config.events:
        if event.start_step < train_end:
            raise ValueError(f"event {event.kind} starts in train window")
        if event.end_step > n_steps:
            raise ValueError(f"event {event.kind} overruns series")

    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)
    interval = steps % STEPS_PER_DAY
    dow = (steps // STEPS_PER_DAY) % DAYS_PER_WEEK
    hour = interval // 4
    temp = _temperature(n_steps, rng)
    shape = _weekly_shape(interval, dow, config.weekend_scale)
    clean = config.base_kw * shape + config.temp_beta * (temp - 24.0)
    clean = clean + rng.normal(0.0, config.noise_std, size=n_steps)
    load, event_mask, event_kind = _apply_events(clean, config.events)
    train_mask = steps < train_end
    return Trajectory(
        plant_id=config.plant_id,
        load_kw=load.astype(np.float64),
        temp_c=temp.astype(np.float64),
        interval=interval.astype(np.int32),
        dow=dow.astype(np.int32),
        hour=hour.astype(np.int32),
        event_mask=event_mask,
        event_kind=event_kind,
        train_mask=train_mask,
        test_mask=~train_mask,
        events=config.events,
    )


def default_plants() -> tuple[PlantConfig, PlantConfig]:
    """Two plants, three labeled test events each."""
    week = STEPS_PER_WEEK
    plant_a = PlantConfig(
        plant_id="plant_a",
        base_kw=140.0,
        weekend_scale=0.68,
        temp_beta=2.2,
        noise_std=3.2,
        events=(
            EventSpec(EVENT_DRIFT, start_step=9 * week + 12, duration_steps=3 * STEPS_PER_DAY, magnitude_kw=18.0),
            EventSpec(EVENT_SENSOR, start_step=11 * week + 20, duration_steps=32, magnitude_kw=45.0),
            EventSpec(EVENT_UPSET, start_step=11 * week + 400, duration_steps=16, magnitude_kw=28.0),
        ),
    )
    plant_b = PlantConfig(
        plant_id="plant_b",
        base_kw=95.0,
        weekend_scale=0.85,
        temp_beta=0.9,
        noise_std=4.0,
        events=(
            EventSpec(EVENT_DRIFT, start_step=8 * week + 48, duration_steps=3 * STEPS_PER_DAY, magnitude_kw=14.0),
            EventSpec(EVENT_UPSET, start_step=10 * week + 80, duration_steps=24, magnitude_kw=22.0),
            EventSpec(EVENT_SENSOR, start_step=11 * week + 200, duration_steps=40, magnitude_kw=38.0),
        ),
    )
    return plant_a, plant_b
