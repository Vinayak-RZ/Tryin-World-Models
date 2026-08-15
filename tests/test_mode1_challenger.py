import numpy as np

from sandbox.mode1_anomaly.challenger import NextStepSurpriseDetector, observation_features
from sandbox.mode1_anomaly.generate import EVENT_SENSOR, STEPS_PER_WEEK, EventSpec, PlantConfig, generate_plant


def test_observation_features_shape() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", events=()), seed=8)
    feats = observation_features(traj)
    assert feats.shape == (traj.n_steps, 6)


def test_surprise_rises_on_sensor_fault() -> None:
    start = 8 * STEPS_PER_WEEK + 10
    cfg = PlantConfig(
        plant_id="t",
        noise_std=0.8,
        events=(EventSpec(EVENT_SENSOR, start_step=start, duration_steps=48, magnitude_kw=50.0),),
    )
    traj = generate_plant(cfg, seed=9)
    det = NextStepSurpriseDetector().fit(traj)
    scores = det.score(traj)
    baseline = float(np.median(scores[traj.train_mask]))
    event_mean = float(np.mean(scores[start : start + 48]))
    assert event_mean > baseline
