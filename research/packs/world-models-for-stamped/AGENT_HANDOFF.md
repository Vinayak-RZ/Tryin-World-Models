---
type: Research Note
title: "Agent handoff — try and simulate world models for Stamped"
description: "Paste-ready brief: offline simulation, compare against CORE champions, report promote/kill. No CORE edits, no bill ₹, no training stack in this repo."
tags: [world-models, agents, handoff, frontier, industrial-intelligence]
lane: FRONTIER
status: curated
stamped_hooks: [baselines, plant-transfer, anomaly, counterfactual, agents]
timestamp: "2026-08-15T00:00:00Z"
---

# Agent handoff — try world models

> **Audience:** An agent asked to *try / simulate* world models and see whether they beat Stamped CORE on a plant-relevant job.  
> **This pack stays docs.** Do not train Dreamer, stand up a GPU stack, or edit L3 CORE in `stamped-external`.  
> **Authority:** [README.md](README.md) · [HOW_WE_USE_THEM.md](HOW_WE_USE_THEM.md) · [north-star](../../strategy/north-star.md) · [ADR-014](../../../decisions/011-015/ADR-014-ts-foundation-model-role.md) · [ADR-015](../../../decisions/016-020/ADR-015-l3-dual-lane-lab-detections.md)

---

## Copy-paste prompt (start here)

```text
MISSION — World-model shadow experiment for Stamped (FRONTIER only).

You will try ONE usage mode, simulate it OFFLINE, and report whether it
beats the existing CORE champion for that job. You will not change product
behavior, write M&V, or invent bill ₹.

### Forbidden (read this first)

- Do not train Dreamer / MuZero / JEPA / DinoWM inside stamped-external.
- Do not edit L3 CORE, finding.json trust, tariff math, or L5 M&V.
- Do not emit invoice numbers, SEC of record, or live OT setpoints.
- Do not claim a product win. Default product commitment is none.
- Do not become a hardware / construction-OS / CMMS company in the write-up.
- If you need code, put it in a sandbox or consumers/stamped-l3-eval Lab
  lane — never in this pack, never in stamped-l3-core engines of record.

### 1) Read (mandatory, in order)

  1. research/packs/world-models-for-stamped/README.md
  2. research/packs/world-models-for-stamped/HOW_WE_USE_THEM.md
  3. research/strategy/north-star.md
  4. research/CATALOG.yml  (then open only the notes you need)
  5. research/maps/stamped-stack-hooks.md
  6. research/strategy/PATHS_FOR_STAMPED.md  (Path B + kill criteria)
  7. decisions/011-015/ADR-014-ts-foundation-model-role.md
  8. decisions/016-020/ADR-015-l3-dual-lane-lab-detections.md

Ponytail first. Separate every claim into:
  source claim · Stamped implication · product commitment (none).

### 2) Pick ONE usage mode

Default if the human did not name one:
  Mode 1 — prediction-error anomaly (cheapest honest test).

Allowed modes (see HOW_WE_USE_THEM.md):
  1. Prediction-error anomaly vs EWMA/CUSUM/SPC
  2. Counterfactual / what-if (display only, not ₹)
  3. Plant transfer / cold-start (JEPA / DinoWM / TimesFM)
  4. WM as L4 query tool (query only, no actuation)
  5. Dense-sensing fusion (protocol / data-shape only unless data exists)
  6. TimesFM/Chronos cousin (ADR-014 shadow — do this before a Dreamer stack)

Do not run mode 5 as a hardware project. Do not run Dreamer-style control
except as imagination on offline data, last in the list.

### 3) Simulate offline only

Allowed data:
  - Public WM / TS-FM demos and paper reproductions
  - Synthetic plant / load curves you generate and document
  - Replay of existing eval corpus (consumers/stamped-l3-eval) if present
  - Anonymized plant exports the human explicitly points you at

Not allowed:
  - Live plant actuation
  - Writing setpoints
  - Touching production CORE paths

### 4) Compare against the champion for THAT job

| Mode | Champion you must beat |
|------|------------------------|
| 1 anomaly | EWMA/CUSUM on TOW-P residuals (or SPC) |
| 2 what-if | Current L6 stub / static spreadsheet — EE-judged, no ₹ |
| 3 transfer | Fleet priors + TOW-P; ADR-014 pinball gates if using TimesFM |
| 4 L4 tool | Deterministic L4 templates — zero numeric leakage |
| 6 TS-FM | Seasonal-naive AND LightGBM pinball, ≥3 plants, no FP regression |

"Feels smarter" is not a metric. Use a table: metric, champion, challenger,
delta, n, caveat. Mark [VERIFIED] / [UNVERIFIED].

### 5) Kill criteria — stop and report

From PATHS_FOR_STAMPED:
  - Cannot define a plant-EE-trusted eval → pause Path B
  - Shadow WM never beats residual SPC on drift in ≥2 plants → kill/pause B
  - Densification cost >> modeled ₹ upside → do not pivot into hardware
  - Anyone would treat the output as a bill number → abort that design

Also abort if you are about to edit CORE or invent bibliographic facts.

### 6) Write the results note

Create ONE dated note, then ONE CATALOG.yml row. Do not spam files.

  Path: research/maps/wm-experiment-<YYYY-MM-DD>-<mode>.md
  (or research/papers/ only if you are noting an external paper, not a run)

Required sections:
  - Mode + champion + data used
  - Metrics table
  - Source claim / Stamped implication / product commitment
  - Promote / hold-shadow / kill recommendation
  - What would change the recommendation

Do not update client-facing citations as if this shipped.

### Exit criteria

- [ ] One mode, offline only
- [ ] Metrics vs the named champion
- [ ] Dated results note + CATALOG.yml row
- [ ] Promote/hold/kill stated
- [ ] Zero CORE / M&V / ₹ mutations
- [ ] Conventional commit on a feature branch if you wrote files
```

---

## First experiments (lazy order)

Run the next one only if the previous is done or explicitly skipped.

1. **Residual / surprise vs EWMA-CUSUM** on replay or synthetic load — Mode 1
2. **TimesFM / Chronos shadow** as the cousin baseline — Mode 6 (already ADR-014)
3. **JEPA-style latent prediction** on multivariate 15-min tags — Mode 3, only if data exists
4. **Counterfactual rollouts** vs the L6 stub — Mode 2, display only
5. **Dreamer-style control** — last; imagination only; never actuate

## Where code may live (if a later pass writes any)

| Place | Allowed? |
|-------|----------|
| `research/packs/world-models-for-stamped/` | **No** — docs only |
| L3 CORE / rulepacks of-record | **No** |
| `consumers/stamped-l3-eval/` Lab lane | Yes — shadow artifacts, `lab_only` |
| A throwaway sandbox repo the human names | Yes |

Eval corpus / Lab UI already exist under [`consumers/stamped-l3-eval/`](../../../consumers/stamped-l3-eval/). Prefer logging a challenger artifact there over inventing a new training repo.

## Honesty

World models might help transfer, surprise-as-anomaly, and shadow what-ifs. They are not a promise to customers. If the experiment loses, say so. That is a successful run.
