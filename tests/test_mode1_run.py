from sandbox.mode1_anomaly.generate import EventSpec, PlantConfig, generate_plant
from sandbox.mode1_anomaly.run import run_all


def test_generate_rejects_train_window_events() -> None:
    cfg = PlantConfig(
        plant_id="t",
        events=(EventSpec("drift", start_step=10, duration_steps=4, magnitude_kw=1.0),),
    )
    try:
        generate_plant(cfg, seed=1)
    except ValueError as exc:
        assert "train window" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_all_has_handoff_fields() -> None:
    result = run_all(seed=7)
    assert result["mode"] == 1
    assert result["product_commitment"] == "none"
    assert result["recommendation"] in {"hold-shadow", "kill", "promote"}
    assert len(result["plants"]) == 2
    assert "plants_beaten" in result
