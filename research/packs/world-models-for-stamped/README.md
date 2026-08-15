---
type: Research Note
title: "World-models pack for Stamped — one-folder entry"
description: "Single link for Stamped context, world-model research, usage modes, stack attachment, and the experiment-agent handoff."
tags: [world-models, stamped, pack, frontier, industrial-intelligence]
lane: FRONTIER
status: curated
stamped_hooks: [baselines, plant-transfer, anomaly, counterfactual, agents, work-routing, dense-sensing]
timestamp: "2026-08-15T00:00:00Z"
---

# World-models pack for Stamped

> **This is the one folder.** Open it, paste the path, or hand it to an agent.  
> **Lane:** FRONTIER. Nothing here is product-of-record unless a future ADR says so.  
> **Identity:** [north-star](../../strategy/north-star.md) — Industrial Intelligence / make plants smarter.

## 60-second lock

**Stamped** is an Industrial Intelligence company. Outcomes stay fixed: energy efficiency (bill-verified ₹ where we claim it), plant / process efficiency, better manufacturing. Backends may change.

**Today’s CORE backend:** SPC, TOW-P / CalTRACK-style baselines, LightGBM, rules / physics, deterministic tariff math. Product SSOT: [STAMPED_ARCHITECTURE.md](../../../technical/STAMPED_ARCHITECTURE.md) · [L3 intelligence core](../../../technical/layers/l3/L3-intelligence-core.md).

**World models** are a FRONTIER backend family under study (Path B in [PATHS_FOR_STAMPED](../../strategy/PATHS_FOR_STAMPED.md)). They learn compressed dynamics, imagine futures, and treat surprise as a signal. They do **not** write M&V or bill ₹. Shadow / Lab only until an ADR promotes them ([ADR-014](../../../decisions/011-015/ADR-014-ts-foundation-model-role.md) spirit; [ADR-015](../../../decisions/016-020/ADR-015-l3-dual-lane-lab-detections.md) Lab lane).

| Layer | Meaning |
|-------|---------|
| **Source claim** | What a paper or blog asserts |
| **Stamped implication** | How it *might* help plant intelligence |
| **Product commitment** | What we ship today — default **none** in this pack |

## What a world model is (plant language)

A world model is a learned predictor of “what happens next.” It compresses noisy plant observations into a smaller latent state, learns how that state evolves, and can roll forward under hypothetical actions or conditions — a compact simulator of energy and process evolution, not a CAD twin of every bolt.

Three ideas that matter: **latent dynamics**, **imagination / planning**, **prediction error as anomaly**. Full primer: [00-world-models-primer](../../concepts/00-world-models-primer.md).

## This pack

| File | Role |
|------|------|
| [README.md](README.md) | This page — Stamped + WM snapshot + research index |
| [HOW_WE_USE_THEM.md](HOW_WE_USE_THEM.md) | Usage modes, L1–L6 attachment, CORE comparison |
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | Paste-ready brief: simulate, compare, report |

**Humans trying models later:** read this page → [HOW_WE_USE_THEM.md](HOW_WE_USE_THEM.md) → pick a mode.  
**Agents asked to try them:** paste [AGENT_HANDOFF.md](AGENT_HANDOFF.md). Do not train Dreamer in this repo.

## Research index (links only)

The evidence room is [`research/`](../../README.md). Agents: prefer [`CATALOG.yml`](../../CATALOG.yml) over grep. Do not copy claims from this pack if a linked note disagrees — the note wins.

### Primers

| Note | What it is |
|------|------------|
| [00 — World models primer](../../concepts/00-world-models-primer.md) | Plant analogy |
| [01 — Survey digest](../../concepts/01-world-models-survey-digest.md) | Ha → RSSM / Dreamer → variants |
| [02 — JEPA / LeWorldModels](../../concepts/02-jepa-and-leworldmodels.md) | Predict in representation space |
| [03 — DINO-WM](../../concepts/03-dinowm.md) | Frozen DINO features as state |
| [04 — Agents with world models](../../concepts/04-agents-with-world-models.md) | WM-agents vs Stamped L4 tool-agents |
| [05 — Physical AI for industry](../../concepts/05-physical-ai-for-industry.md) | Twins, foundation models, industry |

### Reading packs

- [World models](../../compilations/world-models-reading-pack.md)
- [JEPA / LeWorldModels](../../compilations/jepa-leworldmodels-pack.md)
- [DINO-WM](../../compilations/dinowm-pack.md)
- [Agents with WMs](../../compilations/agents-world-models-pack.md)
- [Physical AI / industry](../../compilations/physical-ai-industry-pack.md)

### Strategy (Stamped context)

- [North star](../../strategy/north-star.md) — identity lock
- [Insights for Stamped](../../strategy/INSIGHTS_FOR_STAMPED.md) — what the research changes
- [PATHS FOR STAMPED](../../strategy/PATHS_FOR_STAMPED.md) — technical paths; Path B is the WM backend
- [PATHS V2](../../strategy/PATHS_FOR_STAMPED_V2.md) — founder field guide

### Maps

- [Stack hooks](../../maps/stamped-stack-hooks.md) — where FRONTIER attaches
- [Open questions](../../maps/open-questions.md)
- [Competitor signal](../../maps/competitor-signal-notes.md)
- [YC fit](../../maps/stamped-yc-fit.md)

### Papers and bibliographies

- [Paper notes index](../../papers/_INDEX.md)
- [World-models bibliography](../../bibliography/world-models.bib.md)
- [Physical-world / industrial AI bibliography](../../bibliography/physical-world-ai.bib.md)
- [YC RFS methods](../../yc-rfs/README.md)

### Product SSOT (read-only)

- [STAMPED_ARCHITECTURE.md](../../../technical/STAMPED_ARCHITECTURE.md)
- [L3 intelligence core](../../../technical/layers/l3/L3-intelligence-core.md)
- [Client-facing CORE vs FRONTIER citations](../../../technical/research/stamped-research-and-ml-citations.md)
- [ADR-014 — TS foundation models stay shadow](../../../decisions/011-015/ADR-014-ts-foundation-model-role.md)
- [ADR-015 — L3 dual-lane Lab](../../../decisions/016-020/ADR-015-l3-dual-lane-lab-detections.md)
- [ADR-016 — attribution shadows](../../../decisions/016-020/ADR-016-attribution-shadow-challengers.md)

## Hard rules

- World models and agents **do not write M&V or bill ₹**
- Do not promote FRONTIER into CORE without an ADR
- Do not rewrite identity away from plants / Industrial Intelligence
- This pack **links**; it does not fork claims
