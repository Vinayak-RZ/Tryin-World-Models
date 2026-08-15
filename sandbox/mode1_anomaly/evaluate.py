"""Metrics vs a named champion. No rupee or M&V fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sandbox.mode1_anomaly.generate import EventSpec, Trajectory


@dataclass(frozen=True)
class DetectorMetrics:
    name: str
    plant_id: str
    n_test: int
    n_event_steps: int
    n_alarms_test: int
    false_positive_rate: float
    alarm_precision: float
    event_recall: float
    mean_detection_delay_steps: float
    precision_at_k: float
    k: int
    n_events: int
    n_events_detected: int


def _event_windows(events: tuple[EventSpec, ...], n_steps: int) -> list[tuple[int, int]]:
    windows = []
    for event in events:
        start = max(0, event.start_step)
        end = min(n_steps, event.end_step)
        if end > start:
            windows.append((start, end))
    return windows


def precision_at_k(scores: np.ndarray, event_mask: np.ndarray, eval_mask: np.ndarray, k: int) -> float:
    idx = np.flatnonzero(eval_mask)
    if idx.size == 0 or k <= 0:
        return float("nan")
    order = idx[np.argsort(scores[idx])[::-1]]
    top = order[: min(k, order.size)]
    return float(event_mask[top].mean())


def event_detection(alarms: np.ndarray, events: tuple[EventSpec, ...], n_steps: int) -> tuple[int, float]:
    detected = 0
    delays: list[float] = []
    for start, end in _event_windows(events, n_steps):
        hits = np.flatnonzero(alarms[start:end])
        if hits.size == 0:
            continue
        detected += 1
        delays.append(float(hits[0]))
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return detected, mean_delay


def detector_metrics(
    name: str,
    traj: Trajectory,
    scores: np.ndarray,
    alarms: np.ndarray,
    k: int | None = None,
) -> DetectorMetrics:
    test = traj.test_mask
    event = traj.event_mask & test
    non_event = test & ~traj.event_mask
    n_test = int(test.sum())
    n_event = int(event.sum())
    test_alarms = alarms & test
    n_alarms = int(test_alarms.sum())
    fp = int((test_alarms & non_event).sum())
    tp = int((test_alarms & event).sum())
    fpr = float(fp / non_event.sum()) if non_event.any() else float("nan")
    prec = float(tp / n_alarms) if n_alarms else float("nan")
    k_use = n_event if k is None else k
    detected, mean_delay = event_detection(alarms, traj.events, traj.n_steps)
    return DetectorMetrics(
        name=name,
        plant_id=traj.plant_id,
        n_test=n_test,
        n_event_steps=n_event,
        n_alarms_test=n_alarms,
        false_positive_rate=fpr,
        alarm_precision=prec,
        event_recall=float(detected / len(traj.events)) if traj.events else float("nan"),
        mean_detection_delay_steps=mean_delay,
        precision_at_k=precision_at_k(scores, traj.event_mask, test, k_use),
        k=k_use,
        n_events=len(traj.events),
        n_events_detected=detected,
    )


def compare_detectors(champion: DetectorMetrics, challenger: DetectorMetrics) -> dict[str, object]:
    """Deltas are challenger minus champion. Positive precision/recall is better."""

    def _delta(attr: str) -> float:
        a = getattr(challenger, attr)
        b = getattr(champion, attr)
        if np.isnan(a) or np.isnan(b):
            return float("nan")
        return float(a - b)

    fpr_ok = challenger.false_positive_rate <= champion.false_positive_rate + 1e-9
    better_p_at_k = challenger.precision_at_k > champion.precision_at_k
    better_delay = challenger.mean_detection_delay_steps < champion.mean_detection_delay_steps
    beats = bool(fpr_ok and (better_p_at_k or better_delay))
    return {
        "plant_id": champion.plant_id,
        "delta_precision_at_k": _delta("precision_at_k"),
        "delta_false_positive_rate": _delta("false_positive_rate"),
        "delta_event_recall": _delta("event_recall"),
        "delta_mean_detection_delay_steps": _delta("mean_detection_delay_steps"),
        "fpr_not_worse": fpr_ok,
        "beats_champion_on_this_plant": beats,
        "champion": asdict(champion),
        "challenger": asdict(challenger),
    }
