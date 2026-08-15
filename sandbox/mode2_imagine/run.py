"""CLI: Mode 2 imagination. Display / Lab only. No rupee fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sandbox.mode2_imagine.evaluate import (
    action_blind_towp,
    agreement_mae,
    as_json,
    persistence,
    score_horizons,
    score_model,
)
from sandbox.mode2_imagine.generate import (
    counterfactual_action,
    default_plants,
    generate_plant,
    resimulate,
)
from sandbox.mode2_imagine.gru_dynamics import GRUWorldModel
from sandbox.mode2_imagine.rssm import RSSMWorldModel

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


def _towp_imagine(pred: np.ndarray):
    def imaginer(traj, start: int, horizon: int, action: np.ndarray) -> np.ndarray:
        del traj, action
        return pred[start + 1 : start + 1 + horizon]

    return imaginer


def run_plant(plant_id: str, seed: int, epochs: int) -> dict[str, object]:
    configs = {cfg.plant_id: cfg for cfg in default_plants()}
    traj = generate_plant(configs[plant_id], seed=seed)
    cf_action = counterfactual_action(traj, scale=0.5)
    cf_truth = resimulate(traj, cf_action)

    champ = action_blind_towp(traj)
    persist = persistence(traj)
    champ_mae = float(np.mean(np.abs(traj.load_kw[traj.test_mask] - champ[traj.test_mask])))

    rssm = RSSMWorldModel(epochs=epochs).fit(traj)
    gru = GRUWorldModel(epochs=epochs).fit(traj)

    rssm_1 = rssm.predict_1step(traj)
    gru_1 = gru.predict_1step(traj)

    rssm_score = score_model("rssm", traj, rssm_1, rssm.imagine, champ_mae, cf_truth, cf_action)
    gru_score = score_model("gru_dynamics", traj, gru_1, gru.imagine, champ_mae, cf_truth, cf_action)
    champ_score = score_model("towp_action_blind", traj, champ, _towp_imagine(champ), champ_mae, cf_truth, cf_action)
    persist_score = score_model("persistence", traj, persist, _towp_imagine(persist), champ_mae, cf_truth, cf_action)

    starts = np.flatnonzero(traj.test_mask)
    starts = starts[(starts >= 64) & (starts < traj.n_steps - 17)][::32]
    agree = []
    for s in starts:
        a = rssm.imagine(traj, int(s), 16, traj.action)
        b = gru.imagine(traj, int(s), 16, traj.action)
        agree.append(agreement_mae(a, b))
    agree_mae = float(np.mean(agree)) if agree else float("nan")

    champ_h = score_horizons(traj.load_kw, _towp_imagine(champ), traj, traj.action, (16, 96))
    return {
        "plant_id": plant_id,
        "champion_1step_mae": champ_mae,
        "rssm": as_json(rssm_score),
        "gru_dynamics": as_json(gru_score),
        "towp_action_blind": as_json(champ_score),
        "persistence": as_json(persist_score),
        "agreement_mae_h16": agree_mae,
        "champion_horizon_16_mae": champ_h[16].mae,
        "both_beat_champion_1step": bool(rssm_score.beats_champion_1step and gru_score.beats_champion_1step),
    }


def _recommendation(plant_rows: list[dict[str, object]]) -> tuple[str, str]:
    both = all(bool(p["both_beat_champion_1step"]) for p in plant_rows)
    if both:
        return (
            "hold-shadow",
            "Both WMs beat the action-blind champion on 1-step MAE on both synthetic plants. "
            "Still not an EE-trusted eval. Imagined kW is display only. Do not promote.",
        )
    return (
        "hold-shadow",
        "At least one WM did not beat the action-blind champion on both plants. "
        "Pause Mode 2 as a decision tool. Do not promote.",
    )


def run_all(seed: int = 7, epochs: int = 8) -> dict[str, object]:
    plants = [run_plant(cfg.plant_id, seed=seed + i, epochs=epochs) for i, cfg in enumerate(default_plants())]
    rec, why = _recommendation(plants)
    return {
        "mode": 2,
        "mode_name": "action-conditioned next-state imagination",
        "champion": "action-blind TOW-P (and persistence)",
        "world_models": ["rssm", "gru_dynamics"],
        "action": "process / production setpoint in [0, 1]",
        "data": "synthetic 15-min load+temp+setpoint, 2 plants, 12 weeks",
        "seed": seed,
        "epochs": epochs,
        "product_commitment": "none",
        "lane": "FRONTIER",
        "display_only": True,
        "refuses": ["invoice_rupees", "sec", "live_setpoints", "m_and_v"],
        "plants": plants,
        "recommendation": rec,
        "rationale": why,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mode 2 action-conditioned imagination")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR / "mode2_results.json")
    args = parser.parse_args(argv)
    result = run_all(seed=args.seed, epochs=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(json.dumps(_json_ready(result), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
