import numpy as np

from sandbox.mode1_anomaly.champion import (
    ChampionDetector,
    TimeOfWeekTempBaseline,
    ewma,
    tow_bin,
    two_sided_cusum,
    windowed_cusum,
)
from sandbox.mode1_anomaly.generate import STEPS_PER_DAY, PlantConfig, generate_plant


def test_ewma_recurrence() -> None:
    x = np.array([1.0, 2.0, 3.0])
    got = ewma(x, lam=0.5)
    assert got[0] == 1.0
    assert abs(got[1] - 1.5) < 1e-12
    assert abs(got[2] - 2.25) < 1e-12


def test_cusum_rises_on_mean_shift() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(0.0, 1.0, size=200)
    shifted = np.concatenate([null, rng.normal(3.0, 1.0, size=80)])
    score = two_sided_cusum(shifted, k=0.5)
    assert score[250:].max() > score[:200].max()


def test_windowed_cusum_releases_after_shift() -> None:
    x = np.concatenate([np.zeros(20), np.ones(30) * 4.0, np.zeros(80)])
    latched = two_sided_cusum(x, k=0.5)
    windowed = windowed_cusum(x, k=0.5, window=16)
    assert latched[-1] > 10.0
    assert windowed[-1] < 1.0


def test_towp_recovers_bin_mean_on_noiseless_series() -> None:
    cfg = PlantConfig(plant_id="t", noise_std=0.0, temp_beta=0.0, events=())
    traj = generate_plant(cfg, seed=5)
    model = TimeOfWeekTempBaseline().fit(traj)
    pred = model.predict(traj)
    bins = tow_bin(traj.dow, traj.interval)
    train = traj.train_mask
    for b in range(0, STEPS_PER_DAY * 7, 17):
        sel = train & (bins == b)
        if sel.sum() < 8:
            continue
        assert abs(pred[sel][0] - traj.load_kw[sel].mean()) < 1e-6


def test_champion_alarms_on_large_test_offset() -> None:
    from sandbox.mode1_anomaly.generate import EVENT_DRIFT, STEPS_PER_WEEK, EventSpec

    cfg = PlantConfig(
        plant_id="t",
        noise_std=0.4,
        events=(
            EventSpec(EVENT_DRIFT, start_step=8 * STEPS_PER_WEEK, duration_steps=200, magnitude_kw=40.0),
        ),
    )
    traj = generate_plant(cfg, seed=6)
    det = ChampionDetector().fit(traj)
    alarms = det.alarms(traj)
    assert alarms[traj.event_mask].mean() > 0.5
