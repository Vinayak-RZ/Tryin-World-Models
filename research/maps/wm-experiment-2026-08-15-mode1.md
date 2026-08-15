---
type: Experiment Note
title: "Mode 1 — prediction-error anomaly vs EWMA/CUSUM"
description: "Offline synthetic replay. Residual-surprise proxy, not JEPA/RSSM/Dreamer."
lane: FRONTIER
status: completed
mode: 1
recommendation: hold-shadow
product_commitment: none
timestamp: "2026-08-15T00:00:00Z"
---

# wm-experiment-2026-08-15-mode1

> **Lane:** FRONTIER / shadow. **Product commitment:** none.  
> **Code:** `sandbox/mode1_anomaly/` · seed `7` · [VERIFIED] by `python -m sandbox.mode1_anomaly.run`

## Mode + champion + data

| | |
|--|--|
| **Mode** | 1 — prediction-error anomaly |
| **Champion** | EWMA + 12-hour windowed CUSUM on residuals of a TOW-P-shaped baseline (per time-of-week bin, load ~ a + b·temp). Not CalTRACK-certified. |
| **Challenger** | Standardize → PCA(4) → Ridge `z_t → z_{t+1}`. Score = EWMA of `‖z_{t+1} − ẑ_{t+1}‖`. Residual-surprise **proxy**, not JEPA / RSSM / Dreamer. |
| **Data** | Two seeded synthetic plants, 15-minute load + outdoor temp + calendar features, 12 weeks. Train = weeks 1–8 (clean). Test = weeks 9–12 with labeled drift, sensor-fault, and process-upset windows (3 days / 8 h / 4 h class). No real plant data. |
| **Thresholds** | Each detector uses its own 98th-percentile train score (plus CUSUM floor `h ≥ 5` for the champion). |

Allowed data class from the handoff: synthetic plant / load curves, documented.

## Metrics table

Better means (from the pack): higher precision@k **or** earlier drift catch **without** raising FPR vs residual SPC.

k = number of labeled event steps in test.

### plant_a

| Metric | Champion | Challenger | Delta (challenger − champion) | n | Caveat |
|--------|----------|------------|-------------------------------|---|--------|
| precision@k | 0.979 | 0.176 | −0.804 | k=336 | [VERIFIED] |
| false-positive rate | 0.145 | 0.017 | −0.128 | 2352 non-event test steps | [VERIFIED] operating points not FPR-matched |
| event recall | 1.00 | 0.33 | −0.67 | 3 events | [VERIFIED] |
| mean detection delay (steps) | 0.0 | 84.0 | +84 | detected events only | [VERIFIED] 15 min/step |
| alarm precision | 0.496 | 0.184 | −0.313 | 677 vs 49 test alarms | [VERIFIED] |
| beats champion? | — | no | — | — | FPR better, ranking and delay worse |

### plant_b

| Metric | Champion | Challenger | Delta | n | Caveat |
|--------|----------|------------|-------|---|--------|
| precision@k | 0.980 | 0.151 | −0.830 | k=352 | [VERIFIED] |
| false-positive rate | 0.176 | 0.021 | −0.155 | 2336 non-event test steps | [VERIFIED] operating points not FPR-matched |
| event recall | 1.00 | 0.67 | −0.33 | 3 events | [VERIFIED] |
| mean detection delay (steps) | 0.33 | 32.0 | +31.7 | detected events only | [VERIFIED] |
| alarm precision | 0.461 | 0.111 | −0.350 | 761 vs 54 test alarms | [VERIFIED] |
| beats champion? | — | no | — | — | same pattern as plant_a |

Plants beaten under the pack rule: **0 / 2**.

## Source claim / Stamped implication / product commitment

| Layer | Text |
|-------|------|
| **Source claim** | If a model usually predicts the next plant state, a sudden large residual is a candidate anomaly (drift, sensor fault, process upset). |
| **Stamped implication** | A next-step surprise score *might* challenge EWMA/CUSUM on TOW-P residuals as an L3 Lab detector. On this synthetic replay it did not: residual SPC ranked event steps far better and caught every injected window immediately. The linear latent proxy was more conservative (≈2% FPR) and missed most events. |
| **Product commitment** | **none**. No CORE, M&V, SEC, or bill-rupee change. |

PATHS / ADR-014 / ADR-015 text was not readable in this repo. Kill criteria applied are only those copied into the pack. Those ADR citations are [UNVERIFIED].

## Promote / hold-shadow / kill

**hold-shadow** (pause Path B on this evidence).

Not promote: the challenger lost precision@k and detection delay on both plants.

Not a hard kill of Path B: the eval is synthetic, not plant-EE-trusted, and the challenger is not a JEPA/RSSM-class model. The pack gate “never beats residual SPC on drift in ≥2 plants” is failed **here**, which is enough to pause, not enough to retire the family.

## What would change the recommendation

- A labelled replay from ≥2 real plants that an EE trusts
- Matching FPR operating points (champion still ~15% test FPR vs ~2% for the challenger)
- A stronger FRONTIER model (JEPA / RSSM) if and only if it still lives in Lab
- Event classes where a univariate TOW-P residual is blind (cross-tag, slow multivariate drift)
- Evidence that residual SPC FPR is unacceptable in production and a lower-FPR surprise score recovers recall

## Honesty

This run succeeded as a test: the cheap world-model-shaped proxy lost to residual SPC. That is the useful answer. Do not train Dreamer next unless a later pass explicitly skips this result.
