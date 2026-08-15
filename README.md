# Tryin-World-Models

FRONTIER sandbox for trying world-model ideas against Stamped plant-intelligence jobs.

**Lane:** shadow / Lab only. Nothing here is product-of-record. World-model outputs do **not** write M&V, SEC, or bill rupees.

This repo is the throwaway sandbox named in the world-models pack. The pack stays docs-only. Experiment code lives under `sandbox/`.

## Start here

1. [research/packs/world-models-for-stamped/README.md](research/packs/world-models-for-stamped/README.md) — Stamped + world-model snapshot
2. [research/packs/world-models-for-stamped/HOW_WE_USE_THEM.md](research/packs/world-models-for-stamped/HOW_WE_USE_THEM.md) — usage modes and CORE comparison
3. [research/packs/world-models-for-stamped/AGENT_HANDOFF.md](research/packs/world-models-for-stamped/AGENT_HANDOFF.md) — experiment brief

Default first try: **Mode 1 — prediction-error anomaly** vs EWMA / CUSUM on TOW-P-shaped residuals.

Latest run: [research/maps/wm-experiment-2026-08-15-mode1.md](research/maps/wm-experiment-2026-08-15-mode1.md) — **hold-shadow**. The cheap next-step surprise proxy did not beat residual SPC on two synthetic plants. Product commitment: none.

## What is here

| Path | Role |
|------|------|
| `research/packs/world-models-for-stamped/` | Curated pack from the source zip (unchanged) |
| `research/CATALOG.yml` | Index of notes and experiment results |
| `research/MISSING_CORPUS.md` | Linked `stamped-external` files that are not in this repo |
| `research/maps/` | Dated experiment notes |
| `sandbox/mode1_anomaly/` | Offline Mode 1 simulation (CPU only) |

## What is not here

The pack links a larger research monorepo (`stamped-external`): primers, PATHS, ADRs, L3 eval corpus. Those files are **not** in this zip and **not** public. Claims that depend on them are marked `[UNVERIFIED]`. See [research/MISSING_CORPUS.md](research/MISSING_CORPUS.md).

## Run Mode 1

```bash
python -m pip install -e ".[dev]"
pytest
python -m sandbox.mode1_anomaly.run
```

Product commitment: **none**.
