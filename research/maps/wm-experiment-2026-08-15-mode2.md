---
type: Experiment Note
title: "Mode 2 — action-conditioned next-state imagination"
description: "RSSM + GRU dynamics on a synthetic plant with a process setpoint. Display only."
lane: FRONTIER
status: completed
mode: 2
recommendation: hold-shadow
product_commitment: none
timestamp: "2026-08-15T00:00:00Z"
---

# wm-experiment-2026-08-15-mode2

> **Lane:** FRONTIER / shadow. **Product commitment:** none.  
> **Code:** `python -m sandbox.mode2_imagine.run --epochs 8` · seed `7` · [VERIFIED]  
> Imagined kW is **display only**. No invoice rupees, SEC, live setpoints, or M&V.

## Mode + action + data

| | |
|--|--|
| **Mode** | 2 — what happens to the plant if I take an action? |
| **Action** | Process / production setpoint in `[0, 1]`. Weather is exogenous, not an action. |
| **State** | 15-min load (kW), outdoor temp, calendar features |
| **World models** | Compact Gaussian **RSSM** (PlaNet/Dreamer family, vector obs, no actor). Deterministic **action-conditioned GRU**. Both trained on this series only. |
| **Champion** | Action-blind TOW-P. Persistence reported as a second naive. |
| **Data** | Two seeded synthetic plants, 12 weeks, train 1–8. Counterfactual = same weather/noise, setpoint × 0.5. |

Pretrained DINO-WM / DINOv3 / PushT checkpoints were **not** used. They cannot take a setpoint.

## Metrics

Decent Lab bar: 1-step MAE below the action-blind champion on **both** plants, and counterfactual error not many times worse than 1-step.

### plant_a

| Model | 1-step MAE | 1-step sMAPE | H16 MAE | H96 MAE | CF H16 MAE | beats TOW-P 1-step? |
|-------|------------|--------------|---------|---------|------------|---------------------|
| TOW-P (action-blind) | 11.73 | 0.44 | 11.12 | 11.96 | 19.04 | — |
| Persistence | 3.93 | 0.25 | 3.78 | 3.95 | 12.70 | yes (factual only) |
| RSSM | 5.79 | 0.41 | 7.10 | 34.70 | 6.94 | **yes** |
| GRU dynamics | 2.91 | 0.20 | 4.13 | 5.46 | 5.30 | **yes** |

Cross-model agreement (H16): **5.66 kW** MAE. [VERIFIED]

### plant_b

| Model | 1-step MAE | 1-step sMAPE | H16 MAE | H96 MAE | CF H16 MAE | beats TOW-P 1-step? |
|-------|------------|--------------|---------|---------|------------|---------------------|
| TOW-P (action-blind) | 7.69 | 0.44 | 7.05 | 7.66 | 11.73 | — |
| Persistence | 3.35 | 0.30 | 3.37 | 3.35 | 9.58 | yes (factual only) |
| RSSM | 3.40 | 0.34 | 4.22 | 13.03 | 3.92 | **yes** |
| GRU dynamics | 2.64 | 0.28 | 3.38 | 3.89 | 3.68 | **yes** |

Cross-model agreement (H16): **2.89 kW** MAE. [VERIFIED]

Both WMs beat TOW-P on 1-step on both plants. Plants beaten under the Lab bar: **2 / 2**.

## How well “take an action → next state” worked

**Yes, on this synthetic plant, with caveats.**

The GRU is the useful world model. It beats TOW-P by a wide margin on next-step load and, more importantly, on **counterfactual setpoints** (5.3 vs 19.0 kW on plant_a; 3.7 vs 11.7 on plant_b). Persistence looks good on factual 1-step because the plant has lag; it **fails** the what-if (it cannot see the knob). That is the Mode 2 point.

The RSSM also beats TOW-P on 1-step and on 4-hour what-ifs. It **collapses on 1-day rollouts** (plant_a H96 MAE 34.7). Expected risk for a tiny RSSM on CPU. Do not use it for long dreams.

The two WMs agree within ~3–6 kW on 4-hour factual imaginations. Not identical. Close enough to say both learned the same handle (setpoint → load).

## Source claim / Stamped implication / product commitment

| Layer | Text |
|-------|------|
| **Source claim** | An action-conditioned world model can roll a compact simulator: `(state, action) → next state`, then a short trajectory. |
| **Stamped implication** | On a plant where a process setpoint actually moves load, a small GRU (and a short-horizon RSSM) can preview “if I turn this knob…” better than a time-of-week mean. That is Lab explanation / L6 display, not a verified savings number. |
| **Product commitment** | **none**. Do not treat imagined kW as a bill, SEC, or setpoint. |

## Promote / hold-shadow / kill

**hold-shadow.**

Not promote: synthetic only, not EE-trusted, RSSM 1-day rollouts are unusable, no rupee path.

Not kill: both models cleared the 1-step bar on two plants, and counterfactual scoring shows they actually use the action. Mode 2 is worth another Lab pass on real tags.

## What would change the recommendation

- Real plant trajectories with a logged, EE-trusted action (setpoint / production rate)
- GRU (not RSSM) as the default imaginer; keep RSSM only if 1-day error is fixed
- Matched comparison against a static “if load were X” spreadsheet judged by a plant EE
- Anyone quoting imagined kW as ₹ → abort that design
