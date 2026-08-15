"""Mode 2 metrics. No rupee / M&V / SEC fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sandbox.mode2_imagine.generate import STEPS_PER_DAY, ActuatedTrajectory


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(yhat) + 1e-6
    return float(np.mean(2.0 * np.abs(y - yhat) / denom))


def action_blind_towp(traj: ActuatedTrajectory) -> np.ndarray:
    """Champion: time-of-week mean load. Ignores the setpoint on purpose."""
    bins = traj.dow.astype(np.int32) * 96 + traj.interval.astype(np.int32)
    pred = np.zeros(traj.n_steps, dtype=np.float64)
    for b in range(96 * 7):
        train = traj.train_mask & (bins == b)
        if not train.any():
            pred[bins == b] = float(traj.load_kw[traj.train_mask].mean())
        else:
            pred[bins == b] = float(traj.load_kw[train].mean())
    return pred


def persistence(traj: ActuatedTrajectory) -> np.ndarray:
    out = np.empty(traj.n_steps, dtype=np.float64)
    out[0] = traj.load_kw[0]
    out[1:] = traj.load_kw[:-1]
    return out


@dataclass(frozen=True)
class HorizonScore:
    horizon: int
    mae: float
    smape: float
    n_windows: int


@dataclass(frozen=True)
class ModelScore:
    name: str
    plant_id: str
    one_step_mae: float
    one_step_smape: float
    beats_champion_1step: bool
    horizon_16: HorizonScore
    horizon_96: HorizonScore
    counterfactual_mae: float
    counterfactual_smape: float


def _test_1step(y: np.ndarray, yhat: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    mask = test.copy()
    mask[0] = False
    return mae(y[mask], yhat[mask]), smape(y[mask], yhat[mask])


def score_horizons(
    truth: np.ndarray,
    imaginer,
    traj: ActuatedTrajectory,
    action: np.ndarray,
    horizons: tuple[int, ...],
    stride: int = 32,
) -> dict[int, HorizonScore]:
    test_idx = np.flatnonzero(traj.test_mask)
    out: dict[int, HorizonScore] = {}
    for h in horizons:
        last = traj.n_steps - h - 1
        starts = test_idx[(test_idx >= 64) & (test_idx <= last)]
        starts = starts[::stride]
        if starts.size == 0:
            out[h] = HorizonScore(horizon=h, mae=float("nan"), smape=float("nan"), n_windows=0)
            continue
        errors = []
        smapes = []
        for s in starts:
            pred = imaginer(traj, int(s), h, action)
            tgt = truth[int(s) + 1 : int(s) + 1 + h]
            if pred.shape[0] != tgt.shape[0]:
                n = min(pred.shape[0], tgt.shape[0])
                pred, tgt = pred[:n], tgt[:n]
            errors.append(mae(tgt, pred))
            smapes.append(smape(tgt, pred))
        out[h] = HorizonScore(horizon=h, mae=float(np.mean(errors)), smape=float(np.mean(smapes)), n_windows=len(starts))
    return out


def score_model(
    name: str,
    traj: ActuatedTrajectory,
    one_step: np.ndarray,
    imaginer,
    champion_1step_mae: float,
    cf_truth: np.ndarray,
    cf_action: np.ndarray,
) -> ModelScore:
    one_mae, one_smape = _test_1step(traj.load_kw, one_step, traj.test_mask)
    factual = score_horizons(traj.load_kw, imaginer, traj, traj.action, (16, STEPS_PER_DAY))
    cf = score_horizons(cf_truth, imaginer, traj, cf_action, (16,))
    cf16 = cf[16]
    return ModelScore(
        name=name,
        plant_id=traj.plant_id,
        one_step_mae=one_mae,
        one_step_smape=one_smape,
        beats_champion_1step=one_mae < champion_1step_mae,
        horizon_16=factual[16],
        horizon_96=factual[STEPS_PER_DAY],
        counterfactual_mae=cf16.mae,
        counterfactual_smape=cf16.smape,
    )


def agreement_mae(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    n = min(pred_a.shape[0], pred_b.shape[0])
    return mae(pred_a[:n], pred_b[:n])


def as_json(score: ModelScore) -> dict[str, object]:
    payload = asdict(score)
    return payload
