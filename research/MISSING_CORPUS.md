---
type: Research Note
title: "Missing stamped-external corpus"
description: "Relative links in the world-models pack that resolve only inside stamped-external."
lane: FRONTIER
status: stub
timestamp: "2026-08-15T00:00:00Z"
---

# Missing corpus

The pack at [packs/world-models-for-stamped/](packs/world-models-for-stamped/) was written for a monorepo (`stamped-external`). Those paths are not in this sandbox. Do not invent their contents. Treat any claim that depends on them as `[UNVERIFIED]`.

Kill criteria copied into the pack itself (Mode 1 vs residual SPC; no bill rupees; no CORE edits) are the only gates this repo applies.

## Missing from `research/` (pack-relative `../../`)

| Pack link | Expected path from this repo |
|-----------|------------------------------|
| `strategy/north-star.md` | `research/strategy/north-star.md` |
| `strategy/INSIGHTS_FOR_STAMPED.md` | `research/strategy/INSIGHTS_FOR_STAMPED.md` |
| `strategy/PATHS_FOR_STAMPED.md` | `research/strategy/PATHS_FOR_STAMPED.md` |
| `strategy/PATHS_FOR_STAMPED_V2.md` | `research/strategy/PATHS_FOR_STAMPED_V2.md` |
| `CATALOG.yml` (source) | present here as a **new** catalog, not the source SSOT |
| `README.md` (research root) | not the source research README |
| `concepts/00-world-models-primer.md` | `research/concepts/00-world-models-primer.md` |
| `concepts/01-world-models-survey-digest.md` | `research/concepts/01-world-models-survey-digest.md` |
| `concepts/02-jepa-and-leworldmodels.md` | `research/concepts/02-jepa-and-leworldmodels.md` |
| `concepts/03-dinowm.md` | `research/concepts/03-dinowm.md` |
| `concepts/04-agents-with-world-models.md` | `research/concepts/04-agents-with-world-models.md` |
| `concepts/05-physical-ai-for-industry.md` | `research/concepts/05-physical-ai-for-industry.md` |
| `concepts/08-data-for-the-real-world.md` | `research/concepts/08-data-for-the-real-world.md` |
| `compilations/world-models-reading-pack.md` | `research/compilations/world-models-reading-pack.md` |
| `compilations/jepa-leworldmodels-pack.md` | `research/compilations/jepa-leworldmodels-pack.md` |
| `compilations/dinowm-pack.md` | `research/compilations/dinowm-pack.md` |
| `compilations/agents-world-models-pack.md` | `research/compilations/agents-world-models-pack.md` |
| `compilations/physical-ai-industry-pack.md` | `research/compilations/physical-ai-industry-pack.md` |
| `maps/stamped-stack-hooks.md` | `research/maps/stamped-stack-hooks.md` |
| `maps/open-questions.md` | `research/maps/open-questions.md` |
| `maps/competitor-signal-notes.md` | `research/maps/competitor-signal-notes.md` |
| `maps/stamped-yc-fit.md` | `research/maps/stamped-yc-fit.md` |
| `papers/_INDEX.md` | `research/papers/_INDEX.md` |
| `papers/hafner-planet-2019.md` | `research/papers/hafner-planet-2019.md` |
| `papers/hafner-dreamer-2020.md` | `research/papers/hafner-dreamer-2020.md` |
| `papers/timesfm-google-2024.md` | `research/papers/timesfm-google-2024.md` |
| `papers/chronos-amazon-2024.md` | `research/papers/chronos-amazon-2024.md` |
| `bibliography/world-models.bib.md` | `research/bibliography/world-models.bib.md` |
| `bibliography/physical-world-ai.bib.md` | `research/bibliography/physical-world-ai.bib.md` |
| `yc-rfs/README.md` | `research/yc-rfs/README.md` |

## Missing from product / decisions (pack-relative `../../../`)

| Pack link | Expected monorepo path |
|-----------|------------------------|
| `technical/STAMPED_ARCHITECTURE.md` | product SSOT |
| `technical/layers/l3/L3-intelligence-core.md` | L3 CORE |
| `technical/research/stamped-research-and-ml-citations.md` | client-facing citations |
| `decisions/011-015/ADR-013-counterfactual-savings-ledger.md` | ADR-013 |
| `decisions/011-015/ADR-014-ts-foundation-model-role.md` | ADR-014 |
| `decisions/016-020/ADR-015-l3-dual-lane-lab-detections.md` | ADR-015 |
| `decisions/016-020/ADR-016-attribution-shadow-challengers.md` | ADR-016 |
| `handoff/l6/l6-counterfactual-display-stub.md` | L6 stub |
| `consumers/stamped-l3-eval/` | Lab eval corpus / UI |

## Present in this repo

- Pack files under `research/packs/world-models-for-stamped/`
- This stub
- Local `research/CATALOG.yml` (sandbox index, not the source catalog)
- Dated notes under `research/maps/wm-experiment-*.md`
