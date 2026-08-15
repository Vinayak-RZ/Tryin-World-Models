"""Compare residual SPC to real open models. FRONTIER only. No rupee / CORE writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sandbox.mode1_anomaly.champion import ChampionDetector
from sandbox.mode1_anomaly.challenger import NextStepSurpriseDetector
from sandbox.mode1_anomaly.evaluate import compare_detectors, detector_metrics
from sandbox.mode1_anomaly.generate import default_plants, generate_plant
from sandbox.open_models.chronos_bolt import ChronosSurpriseDetector
from sandbox.open_models.dino_wm import DinoWmSurpriseDetector

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _recommendation(n_win: int) -> tuple[str, str]:
    if n_win >= 2:
        return (
            "hold-shadow",
            "Beats residual SPC on both synthetic plants. Still not an EE-trusted eval. Do not promote.",
        )
    if n_win == 1:
        return (
            "hold-shadow",
            "Beats the champion on one synthetic plant only. Gate B wants >=2 plants. Hold shadow.",
        )
    return (
        "hold-shadow",
        "Did not beat residual SPC on two synthetic plants. Pause Path B on this evidence. Do not promote.",
    )


def _score_challenger(name: str, factory: Callable[[], Any], traj: Any, champ_m: Any) -> dict[str, object]:
    det = factory().fit(traj)
    metrics = detector_metrics(name, traj, det.score(traj), det.alarms(traj))
    return compare_detectors(champ_m, metrics)


def run_plant(plant_id: str, seed: int, factories: dict[str, Callable[[], Any]]) -> dict[str, object]:
    configs = {cfg.plant_id: cfg for cfg in default_plants()}
    traj = generate_plant(configs[plant_id], seed=seed)
    champion = ChampionDetector().fit(traj)
    champ_m = detector_metrics("ewma_cusum_towp", traj, champion.score(traj), champion.alarms(traj))
    models = {name: _score_challenger(name, factory, traj, champ_m) for name, factory in factories.items()}
    return {"plant_id": plant_id, "models": models}


def _summarize_model(name: str, family: str, pretrained: str, plant_rows: list[dict[str, object]]) -> dict[str, object]:
    comparisons = [row["models"][name] for row in plant_rows]  # type: ignore[index]
    n_win = int(sum(bool(c["beats_champion_on_this_plant"]) for c in comparisons))
    rec, why = _recommendation(n_win)
    return {
        "name": name,
        "family": family,
        "pretrained": pretrained,
        "plants_beaten": n_win,
        "recommendation": rec,
        "rationale": why,
        "plants": comparisons,
    }


def default_factories(*, use_real: bool, dino_id: str, chronos_id: str, device: str) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "pca_ridge_proxy": NextStepSurpriseDetector,
    }
    if use_real:
        from sandbox.open_models.chronos_bolt import ChronosBoltBackend
        from sandbox.open_models.dino_wm import HuggingFaceDinoEncoder

        dino_enc = HuggingFaceDinoEncoder(model_id=dino_id, device=device)
        chrono = ChronosBoltBackend(model_id=chronos_id, device=device)
        factories["dino_wm_head"] = lambda: DinoWmSurpriseDetector(encoder=dino_enc, encoder_id=dino_id)
        factories["chronos_bolt"] = lambda: ChronosSurpriseDetector(backend=chrono, model_id=chronos_id)
    else:
        factories["dino_wm_head"] = DinoWmSurpriseDetector
        factories["chronos_bolt"] = ChronosSurpriseDetector
    return factories


def run_all(
    seed: int = 7,
    *,
    use_real: bool = False,
    dino_id: str = "facebook/dinov2-small",
    chronos_id: str = "amazon/chronos-bolt-tiny",
    device: str = "cpu",
) -> dict[str, object]:
    factories = default_factories(use_real=use_real, dino_id=dino_id, chronos_id=chronos_id, device=device)
    plant_rows = [run_plant(cfg.plant_id, seed=seed + i, factories=factories) for i, cfg in enumerate(default_plants())]
    meta = {
        "pca_ridge_proxy": ("linear latent proxy — not a world model", "none (fit on this series only)"),
        "dino_wm_head": (
            "DINO-WM recipe: frozen vision encoder + Ridge predictor head",
            dino_id if use_real else "fake encoder (no weights)",
        ),
        "chronos_bolt": (
            "Chronos-Bolt forecast residual — TS foundation cousin, not a WM",
            chronos_id if use_real else "fake last-value backend",
        ),
    }
    models = [_summarize_model(name, family, pretrained, plant_rows) for name, (family, pretrained) in meta.items()]
    return {
        "mode": 1,
        "mode_name": "prediction-error anomaly vs real open models",
        "champion": "EWMA/CUSUM on TOW-P-shaped residuals",
        "data": "synthetic 15-min load+temp, 2 plants, 12 weeks, events in test only",
        "use_real_weights": use_real,
        "seed": seed,
        "product_commitment": "none",
        "lane": "FRONTIER",
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mode 1 vs open pretrained models")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real", action="store_true", help="Download and run DINOv2 + Chronos-Bolt")
    parser.add_argument("--dino-id", default="facebook/dinov2-small")
    parser.add_argument("--chronos-id", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR / "open_model_results.json")
    args = parser.parse_args(argv)
    result = run_all(
        seed=args.seed,
        use_real=args.real,
        dino_id=args.dino_id,
        chronos_id=args.chronos_id,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(json.dumps(_json_ready(result), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
