from sandbox.mode1_anomaly.generate import (
    EVENT_DRIFT,
    STEPS_PER_WEEK,
    EventSpec,
    PlantConfig,
    default_plants,
    generate_plant,
)


def test_events_stay_in_test_window() -> None:
    cfg = PlantConfig(
        plant_id="t",
        events=(EventSpec(EVENT_DRIFT, start_step=8 * STEPS_PER_WEEK, duration_steps=10, magnitude_kw=5.0),),
    )
    traj = generate_plant(cfg, seed=1)
    assert traj.event_mask[traj.train_mask].sum() == 0
    assert traj.event_mask[traj.test_mask].sum() == 10


def test_event_injection_shifts_load() -> None:
    import numpy as np

    start = 8 * STEPS_PER_WEEK
    cfg = PlantConfig(
        plant_id="t",
        noise_std=0.0,
        events=(EventSpec(EVENT_DRIFT, start_step=start, duration_steps=4, magnitude_kw=10.0),),
    )
    dirty = generate_plant(cfg, seed=2)
    clean = generate_plant(PlantConfig(plant_id="t", noise_std=0.0), seed=2)
    assert np.allclose(dirty.load_kw[start : start + 4], clean.load_kw[start : start + 4] + 10.0)


def test_default_plants_are_two_and_seeded() -> None:
    a, b = default_plants()
    t1 = generate_plant(a, seed=3)
    t2 = generate_plant(a, seed=3)
    t3 = generate_plant(b, seed=4)
    assert t1.plant_id != t3.plant_id
    assert (t1.load_kw == t2.load_kw).all()
    assert t1.n_steps == 12 * STEPS_PER_WEEK
    assert len(a.events) == 3
    assert len(b.events) == 3
