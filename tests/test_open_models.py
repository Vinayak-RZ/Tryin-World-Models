import numpy as np

from sandbox.mode1_anomaly.generate import PlantConfig, generate_plant
from sandbox.open_models.chronos_bolt import ChronosSurpriseDetector, FakeForecastBackend
from sandbox.open_models.dino_wm import DinoWmSurpriseDetector, FakeImageEncoder
from sandbox.open_models.render import render_indices, render_window
from sandbox.open_models.run import run_all


def test_render_window_shape_and_channels() -> None:
    load = np.linspace(80, 120, 40)
    temp = np.linspace(18, 26, 40)
    img = render_window(load, temp, end_step=40, lookback=40, size=64)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img.max() > 0


def test_render_indices_respect_lookback() -> None:
    idx = render_indices(n_steps=100, lookback=20, stride=10)
    assert idx[0] == 20
    assert (np.diff(idx) == 10).all()


def test_fake_dino_head_fits_and_scores() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", events=()), seed=1)
    det = DinoWmSurpriseDetector(encoder=FakeImageEncoder(dim=8), stride=16, lookback=32).fit(traj)
    scores = det.score(traj)
    assert scores.shape == (traj.n_steps,)
    assert np.isfinite(scores).all()
    assert det.alarms(traj).dtype == bool


def test_fake_chronos_fits_and_scores() -> None:
    traj = generate_plant(PlantConfig(plant_id="t", events=()), seed=2)
    det = ChronosSurpriseDetector(backend=FakeForecastBackend(), context_len=64, stride=16).fit(traj)
    scores = det.score(traj)
    assert scores.shape == (traj.n_steps,)
    assert np.isfinite(scores).all()


def test_run_all_fake_backends_lists_three_models() -> None:
    result = run_all(seed=3, use_real=False)
    names = [m["name"] for m in result["models"]]
    assert names == ["pca_ridge_proxy", "dino_wm_head", "chronos_bolt"]
    assert result["use_real_weights"] is False
    assert result["product_commitment"] == "none"
