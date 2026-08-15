# Tryin-World-Models

FRONTIER sandbox for trying world-model ideas against Stamped plant-intelligence jobs.

**Lane:** shadow / Lab only. Nothing here is product-of-record. World-model outputs do **not** write M&V, SEC, or bill rupees.

This repo is the throwaway sandbox named in the world-models pack. The pack stays docs-only. Experiment code lives under `sandbox/`.

## Start here

1. [research/packs/world-models-for-stamped/README.md](research/packs/world-models-for-stamped/README.md) — Stamped + world-model snapshot
2. [research/packs/world-models-for-stamped/HOW_WE_USE_THEM.md](research/packs/world-models-for-stamped/HOW_WE_USE_THEM.md) — usage modes and CORE comparison
3. [research/packs/world-models-for-stamped/AGENT_HANDOFF.md](research/packs/world-models-for-stamped/AGENT_HANDOFF.md) — experiment brief

Default first try: **Mode 1 — prediction-error anomaly** vs EWMA / CUSUM on TOW-P-shaped residuals.

Latest proxy run: [research/maps/wm-experiment-2026-08-15-mode1.md](research/maps/wm-experiment-2026-08-15-mode1.md) — PCA+Ridge only.

Latest **real weights** run: [research/maps/wm-experiment-2026-08-15-open-models.md](research/maps/wm-experiment-2026-08-15-open-models.md) — DINOv2-small + head and Chronos-Bolt Tiny. Both **hold-shadow**. Chronos is usable; DINO-on-sparklines is not. Details: [research/MODELS.md](research/MODELS.md).

Latest **Mode 2** run: [research/maps/wm-experiment-2026-08-15-mode2.md](research/maps/wm-experiment-2026-08-15-mode2.md) — RSSM + GRU imagine next load given a process setpoint. Both beat action-blind TOW-P on 1-step. **hold-shadow.** Display only.

## What is here

| Path | Role |
|------|------|
| `research/packs/world-models-for-stamped/` | Curated pack from the source zip (unchanged) |
| `research/CATALOG.yml` | Index of notes and experiment results |
| `research/MISSING_CORPUS.md` | Linked `stamped-external` files that are not in this repo |
| `research/maps/` | Dated experiment notes |
| `sandbox/mode1_anomaly/` | Offline Mode 1 simulation (PCA+Ridge proxy) |
| `sandbox/open_models/` | Chronos-Bolt + DINO-WM-style frozen encoder + head |
| `sandbox/mode2_imagine/` | Action-conditioned RSSM + GRU (Mode 2 what-if) |
| `research/MODELS.md` | Which weights we use vs robot-pixel checkpoints |

## What is not here

The pack links a larger research monorepo (`stamped-external`): primers, PATHS, ADRs, L3 eval corpus. Those files are **not** in this zip and **not** public. Claims that depend on them are marked `[UNVERIFIED]`. See [research/MISSING_CORPUS.md](research/MISSING_CORPUS.md).

## Run Mode 1

```bash
python -m pip install -e ".[dev]"
pytest
python -m sandbox.mode1_anomaly.run          # PCA+Ridge proxy only
python -m pip install -e ".[open]"
python -m sandbox.open_models.run --real     # DINOv2-small + Chronos-Bolt Tiny
python -m sandbox.mode2_imagine.run          # RSSM + GRU, action → next load
```

Product commitment: **none**.
