---
type: Experiment Note
title: "Mode 1 — real open models vs EWMA/CUSUM"
description: "DINOv2-small + plant head (DINO-WM recipe) and Chronos-Bolt Tiny vs residual SPC."
lane: FRONTIER
status: completed
mode: 1
recommendation: hold-shadow
product_commitment: none
timestamp: "2026-08-15T00:00:00Z"
---

# wm-experiment-2026-08-15-open-models

> **Lane:** FRONTIER / shadow. **Product commitment:** none.  
> **Code:** `python -m sandbox.open_models.run --real` · seed `7` · [VERIFIED]

## What we actually loaded

The earlier Mode 1 note used **PCA + Ridge**. That is not a world model.

This run loaded published weights:

| Challenger | Weights | What it is |
|------------|---------|------------|
| `pca_ridge_proxy` | none | Same cheap stand-in as before |
| `dino_wm_head` | `facebook/dinov2-small` (frozen) + Ridge head fit on this series | DINO-WM *recipe*, not a PushT/Maze checkpoint |
| `chronos_bolt` | `amazon/chronos-bolt-tiny` (9M, zero-shot) | Time-series foundation cousin (pack Mode 6) |

We did **not** load `gaoyuezhou/dino_wm` PushT/PointMaze/Wall checkpoints. Those are robot-pixel models. DINOv3 is the same adapter as DINOv2 here; we used DINOv2-small because it is ungated and CPU-runnable. See [MODELS.md](../MODELS.md).

Champion is unchanged: EWMA + 12-hour windowed CUSUM on a TOW-P-shaped residual.

## Metrics

Better means: higher precision@k **or** earlier detection, **without** raising FPR vs residual SPC.

### plant_a (k = 336 event steps)

| Model | precision@k | FPR | event recall | delay (steps) | beats? |
|-------|-------------|-----|--------------|---------------|--------|
| Champion EWMA/CUSUM | 0.979 | 0.145 | 1.00 | 0.0 | — |
| PCA+Ridge proxy | 0.176 | 0.017 | 0.33 | 84 | no |
| DINOv2-small + head | 0.158 | 0.998 | 1.00 | 0.0 | no (FPR collapse) |
| Chronos-Bolt Tiny | 0.244 | 0.036 | 1.00 | 31 | no |

### plant_b (k = 352 event steps)

| Model | precision@k | FPR | event recall | delay (steps) | beats? |
|-------|-------------|-----|--------------|---------------|--------|
| Champion EWMA/CUSUM | 0.980 | 0.176 | 1.00 | 0.33 | — |
| PCA+Ridge proxy | 0.151 | 0.021 | 0.67 | 32 | no |
| DINOv2-small + head | 0.151 | 0.997 | 1.00 | 0.0 | no (FPR collapse) |
| Chronos-Bolt Tiny | 0.287 | 0.051 | 1.00 | 12.7 | no |

Plants beaten: **0 / 2** for every challenger. [VERIFIED]

## How well each one worked

**Residual SPC (champion)** still ranks event steps almost perfectly on this synthetic set. That is the bar.

**Chronos-Bolt Tiny** is the only *real* model that behaved like a detector. It caught all six injected windows and ran at ~4–5% FPR (tighter than the champion). It was slower to fire and much worse at ranking (precision@k 0.24–0.29 vs 0.98). Useful as a conservative shadow cousin. Not a win under the pack rule.

**DINOv2-small + Ridge head** did not work. Train-calibrated surprise alarmed ~99.8% of test steps. A photo-trained ViT on green/blue sparkline plots does not give a stable plant state. Swapping the backbone to DINOv3 will not fix that observation. A PushT DINO-WM checkpoint would be worse (wrong domain entirely).

**PCA+Ridge** remains a weak, conservative proxy. Same story as the first note.

## Source claim / Stamped implication / product commitment

| Layer | Text |
|-------|------|
| **Source claim** | Frozen DINO features + a predictor can be a world model on pixels (DINO-WM). Chronos-Bolt is a strong zero-shot TS forecaster. |
| **Stamped implication** | On 15-min load+temp, Chronos-Bolt is the open model worth iterating (LoRA / a plant head). DINO-WM-style vision on rendered tags is not. Residual SPC still wins Mode 1 on this synthetic replay. |
| **Product commitment** | **none** |

## Promote / hold-shadow / kill

**hold-shadow** for all three challengers.

Not promote. Not a hard kill of Chronos as a Mode 6 cousin — it is the only pretrained model that produced a usable surprise signal. Pause Path B (world-model backend) until something beats residual SPC on ranking or delay at a matched FPR, on real plants.

## What would change the recommendation

- Fine-tune Chronos-Bolt (LoRA / AutoGluon) on plant tags and re-run Mode 1 at a matched FPR
- A non-vision plant encoder (don’t render sparklines into DINO)
- DINOv3 only after the *observation* is image-like (cameras, thermal, not kW charts)
- Labelled replay from ≥2 real plants

## Fine-tune / head (next Lab step)

1. Keep Chronos-Bolt Tiny frozen, train a small residual head on plant load (cheap).
2. If that still loses: LoRA Chronos-Bolt on the synthetic + any anonymized exports.
3. Do not spend GPU on DINOv3-WM or Dreamer until the observation is not a sparkline.
