---
type: Research Note
title: "Which world models this sandbox actually uses"
lane: FRONTIER
status: curated
timestamp: "2026-08-15T00:00:00Z"
---

# Which models we use

The first Mode 1 run did **not** use a published world model. It used a 4-D PCA + Ridge next-step fit on `[load, temp, hour, dow]`. That is a residual-surprise *proxy*. It is useful as a dumb baseline. It is not DINO-WM, DINOv3, JEPA, Dreamer, or Chronos.

## What “DINO v3 world model” actually is

| Name | What it is | Fits Stamped 15-min tags? |
|------|------------|---------------------------|
| **DINOv3** | Meta vision foundation model (pixels → patch embeddings). [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Not by itself. No load/kW input. |
| **DINO-WM** | Frozen **DINOv2** encoder + a learned predictor on those embeddings. Checkpoints are for PointMaze / PushT / Wall. [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | Those weights do **not** transfer to plant meters. The *recipe* does. |
| **JEPA-WMs** | Meta’s later stack; some variants use a **DINOv3** encoder on robot/video frames. [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | Same issue: pretrained on DROID / RoboCasa pixels, not SCADA. |
| **LeWM** | Small JEPA trained from pixels on control envs. | Same: env-specific pixels. |
| **Chronos-Bolt** | Amazon time-series foundation model (9M–205M). Open weights. | **Yes.** This is the pack’s Mode 6 cousin, not a world model. |
| **TimesFM** | Google TS foundation model. | Yes, heavier than Chronos-Bolt Tiny. |

A PushT DINO-WM checkpoint asked “what does this robot image do next?” It cannot be dropped onto a kW series.

## What this repo runs

1. **Champion** — EWMA + windowed CUSUM on a TOW-P-shaped residual. Not a WM.
2. **pca_ridge_proxy** — the original cheap latent stand-in. Not a WM.
3. **dino_wm_head** — DINO-WM *recipe* on Stamped-shaped data:
   - Render a 1-day load+temp window as a 224×224 RGB frame
   - Frozen open encoder (default `facebook/dinov2-small`; DINOv3 is a drop-in `--dino-id` when you have license + GPU)
   - Train only a Ridge head: `z_t → z_{t+1}`
   - Surprise = next-embedding error
4. **chronos_bolt** — `amazon/chronos-bolt-tiny` (9M), zero-shot next-step residual on load.

DINOv2-small is the encoder we can download and run on CPU without a gated DINOv3 license. The head is the “load on top” you asked for. Swapping the backbone to DINOv3 later does not change the adapter.

## Fine-tune / head path (later, still Lab)

| Step | Do | Do not |
|------|----|--------|
| Now | Frozen DINO + linear/MLP head; Chronos zero-shot | Claim product win |
| Next | Larger predictor (ViT, as in DINO-WM); Chronos-Bolt LoRA on plant tags | Train Dreamer in CORE |
| Later | Unfreeze last DINO block on plant-rendered frames, only if Mode 1 beats residual SPC on real plants | Write M&V or bill rupees |

## How to run

```bash
python -m pip install -e ".[dev,open]"
pytest
python -m sandbox.open_models.run --real
```

Fake backends (no downloads) are the default for tests.

## Latest real-weight run (2026-08-15)

See [maps/wm-experiment-2026-08-15-open-models.md](maps/wm-experiment-2026-08-15-open-models.md).

| Model | Plants beaten | How it behaved |
|-------|---------------|----------------|
| Residual SPC champion | — | precision@k ≈ 0.98, delay ≈ 0 |
| Chronos-Bolt Tiny | 0 / 2 | Caught every event, FPR ~4–5%, worse ranking/delay |
| DINOv2-small + Ridge head | 0 / 2 | FPR ~99.8% — photo DINO on sparklines is not a plant WM |
| PCA+Ridge proxy | 0 / 2 | Conservative, misses events |

**hold-shadow.** Chronos is the only open model worth putting a head / LoRA on next. DINOv3 will not fix sparkline-as-image.
