import numpy as np

from sandbox.mode1_anomaly.evaluate import (
    DetectorMetrics,
    compare_detectors,
    detector_metrics,
    precision_at_k,
)
from sandbox.mode1_anomaly.generate import EVENT_DRIFT, STEPS_PER_WEEK, EventSpec, PlantConfig, generate_plant


def test_precision_at_k_perfect_ranking() -> None:
    scores = np.array([0.1, 0.2, 9.0, 8.0, 0.0])
    events = np.array([False, False, True, True, False])
    eval_mask = np.ones(5, dtype=bool)
    assert precision_at_k(scores, events, eval_mask, k=2) == 1.0
    assert precision_at_k(scores, events, eval_mask, k=4) == 0.5


def test_detector_metrics_counts_test_alarms() -> None:
    start = 8 * STEPS_PER_WEEK
    cfg = PlantConfig(
        plant_id="t",
        events=(EventSpec(EVENT_DRIFT, start_step=start, duration_steps=20, magnitude_kw=5.0),),
    )
    traj = generate_plant(cfg, seed=10)
    scores = np.zeros(traj.n_steps)
    scores[start : start + 10] = 5.0
    alarms = scores >= 1.0
    metrics = detector_metrics("toy", traj, scores, alarms, k=10)
    assert metrics.n_events_detected == 1
    assert metrics.event_recall == 1.0
    assert metrics.mean_detection_delay_steps == 0.0
    assert metrics.precision_at_k == 1.0


def test_compare_requires_fpr_not_worse() -> None:
    champ = DetectorMetrics(
        name="c",
        plant_id="p",
        n_test=100,
        n_event_steps=10,
        n_alarms_test=5,
        false_positive_rate=0.02,
        alarm_precision=0.5,
        event_recall=1.0,
        mean_detection_delay_steps=4.0,
        precision_at_k=0.4,
        k=10,
        n_events=1,
        n_events_detected=1,
    )
    better = DetectorMetrics(
        name="w",
        plant_id="p",
        n_test=100,
        n_event_steps=10,
        n_alarms_test=5,
        false_positive_rate=0.01,
        alarm_precision=0.6,
        event_recall=1.0,
        mean_detection_delay_steps=2.0,
        precision_at_k=0.7,
        k=10,
        n_events=1,
        n_events_detected=1,
    )
    worse_fpr = DetectorMetrics(
        name="w",
        plant_id="p",
        n_test=100,
        n_event_steps=10,
        n_alarms_test=20,
        false_positive_rate=0.10,
        alarm_precision=0.3,
        event_recall=1.0,
        mean_detection_delay_steps=1.0,
        precision_at_k=0.9,
        k=10,
        n_events=1,
        n_events_detected=1,
    )
    assert compare_detectors(champ, better)["beats_champion_on_this_plant"] is True
    assert compare_detectors(champ, worse_fpr)["beats_champion_on_this_plant"] is False
