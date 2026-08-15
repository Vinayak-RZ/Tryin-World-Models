"""CLI: offline Mode 1 run. Writes JSON artifacts, no CORE / rupee fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sandbox.mode1_anomaly.champion import ChampionDetector
from sandbox.mode1_anomaly.challenger import NextStepSurpriseDetector
from sandbox.mode1_anomaly.evaluate import compare_detectors, detector_metrics
from sandbox.mode1_anomaly.generate import default_plants, generate_plant

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


def run_plant(plant_id: str, seed: int) -> dict[str, object]:
    configs = {cfg.plant_id: cfg for cfg in default_plants()}
    traj = generate_plant(configs[plant_id], seed=seed)
    champion = ChampionDetector().fit(traj)
    challenger = NextStepSurpriseDetector().fit(traj)
    champ_scores = champion.score(traj)
    chall_scores = challenger.score(traj)
    champ_m = detector_metrics("ewma_cusum_towp", traj, champ_scores, champion.alarms(traj))
    chall_m = detector_metrics("next_step_surprise", traj, chall_scores, challenger.alarms(traj))
    return compare_detectors(champ_m, chall_m)


def run_all(seed: int = 7) -> dict[str, object]:
    plants = [run_plant(cfg.plant_id, seed=seed + i) for i, cfg in enumerate(default_plants())]
    wins = [bool(p["beats_champion_on_this_plant"]) for p in plants]
    n_win = int(sum(wins))
    if n_win >= 2:
        recommendation = "hold-shadow"
        rationale = (
            "Challenger beat residual SPC on drift/events in both synthetic plants "
            "without a worse train-calibrated FPR. Evidence is synthetic only — not "
            "an EE-trusted plant eval. Do not promote."
        )
    elif n_win == 1:
        recommendation = "hold-shadow"
        rationale = (
            "Challenger beat the champion on one synthetic plant only. PATHS-style "
            "gate B asks for residual-SPC wins on >=2 plants. Hold shadow."
        )
    else:
        recommendation = "hold-shadow"
        rationale = (
            "Challenger did not beat residual SPC on two synthetic plants. "
            "On this evidence Path B stays paused. Do not promote."
        )
    return {
        "mode": 1,
        "mode_name": "prediction-error anomaly",
        "champion": "EWMA/CUSUM on TOW-P-shaped residuals",
        "challenger": "PCA+Ridge next-step latent surprise (not JEPA/RSSM)",
        "data": "synthetic 15-min load+temp, 2 plants, 12 weeks, events in test only",
        "seed": seed,
        "plants": plants,
        "plants_beaten": n_win,
        "recommendation": recommendation,
        "rationale": rationale,
        "product_commitment": "none",
        "lane": "FRONTIER",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mode 1 offline anomaly experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR / "mode1_results.json")
    args = parser.parse_args(argv)
    result = run_all(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(json.dumps(_json_ready(result), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
