import numpy as np

from sandbox.mode2_imagine.evaluate import action_blind_towp, mae, score_model, smape
from sandbox.mode2_imagine.generate import (
    PlantConfig,
    counterfactual_action,
    generate_plant,
    resimulate,
    step_load,
)


def test_setpoint_changes_load() -> None:
    cfg = PlantConfig(plant_id="t", n_weeks=2, train_weeks=1, noise_std=0.0)
    low = generate_plant(cfg, seed=1)
    high_action = np.ones_like(low.action)
    high = resimulate(low, high_action)
    assert high.mean() > low.load_kw.mean()


def test_resimulate_same_action_matches() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", n_weeks=2, train_weeks=1), seed=2)
    replay = resimulate(traj, traj.action)
    assert np.allclose(replay, traj.load_kw)


def test_step_load_respects_action() -> None:
    cfg = PlantConfig(plant_id="t")
    low = step_load(100.0, 0.2, 24.0, 1.0, cfg, 0.0)
    high = step_load(100.0, 1.0, 24.0, 1.0, cfg, 0.0)
    assert high > low


def test_counterfactual_is_scaled() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", n_weeks=2, train_weeks=1), seed=3)
    cf = counterfactual_action(traj, scale=0.5)
    assert cf.max() <= 1.0
    assert (cf <= traj.action + 1e-9).all()


def test_mae_smape_perfect() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_action_blind_ignores_setpoint_bins() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", n_weeks=3, train_weeks=2), seed=4)
    pred = action_blind_towp(traj)
    assert pred.shape == traj.load_kw.shape
    assert np.isfinite(pred).all()


def test_gru_fits_short_series_and_scores() -> None:
    import pytest

    pytest.importorskip("torch")
    from sandbox.mode2_imagine.gru_dynamics import GRUWorldModel

    traj = generate_plant(PlantConfig(plant_id="t", n_weeks=3, train_weeks=2, noise_std=0.5), seed=5)
    model = GRUWorldModel(epochs=2, seq_len=16, batch_size=8, hidden=16).fit(traj)
    pred = model.predict_1step(traj)
    assert pred.shape == traj.load_kw.shape
    imagined = model.imagine(traj, start=80, horizon=8, action=traj.action)
    assert imagined.shape == (8,)
    champ = action_blind_towp(traj)
    champ_mae = float(np.mean(np.abs(traj.load_kw[traj.test_mask] - champ[traj.test_mask])))
    cf = counterfactual_action(traj)
    score = score_model("gru_dynamics", traj, pred, model.imagine, champ_mae, resimulate(traj, cf), cf)
    assert score.plant_id == "t"
    assert np.isfinite(score.one_step_mae)
