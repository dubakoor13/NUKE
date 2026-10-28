# Stage 44 — verified OpenMC v0.15.3 fission-source feedback mechanism

**Status: genuinely executed numerical physics, successfully verified.** [Actions run 40759925867](https://github.com/dubakoor13/NUKE/actions/runs/40759925867) passed **8 adversarial mathematical and 2 mandatory genuine-original-HDF5 acceptance tests**. No new fuel, uranium enrichment, reactor geometry, control drum, hydrogen state or other physical design changes were made.

## Finding: different inactive generation counts alter source-producing k feedback, not merely reported statistics

Verified against the **original upstream OpenMC C++ v0.15.3** `openmc-dev/openmc` Git tag, *not just a NUKE repository copy*:

- [Official `src/eigenvalue.cpp` v0.15.3](https://github.com/openmc-dev/openmc/blob/v0.15.3/src/eigenvalue.cpp), Git blob SHA **`a2120a006d5758df3bd831f8b813ad2338e8b9d6`**: `calculate_average_keff()` sets `simulation::keff` to the **current generation k** in inactive generations, but the **running mean of active-generation k** when active.
- [Official `src/physics.cpp` v0.15.3](https://github.com/openmc-dev/openmc/blob/v0.15.3/src/physics.cpp), Git blob SHA **`41509af97be90e8853bd9bb6be0d252312069735`**: neutron fission production uses `nu_t = p.wgt() / simulation::keff * weight * nu_fission / total`. Thus at unchanged particle state, the expected number of banked neutrons depends **inversely on the feedback `keff`**.
- The C++ reader independently downloaded both official files at the exact upstream release tag, recomputed canonical Git blob hashes, tested for those numerical operations, and rejected any modified source.

**OpenMC physically draws a next-generation source from the fission bank.** When active and inactive running-`k` policies differ, the same bank and RNG stream at the prior step need not produce the same next bank. This is a model/algorithm mechanism, not automatic evidence of nonconvergence or a defect. Earlier Stage42 proved user tally scoring does not cause the divergence; Stage43 identified the first actual saved fission-bank divergence at batch 23 and verified additional checkpoint writes do not change the histories.

## Original HDF5 shows when the source feedback first differs

The audit reopened **32 genuine native neutron/photon eigenvalue HDF5 checkpoints** from previously executed [Stage43 original 56-StatePoint archive](https://github.com/dubakoor13/NUKE/actions/runs/40757722214/artifacts/12847760477): broad/compact source starts × 20/40 inactive × batches 19,20,21,22,23,24,60,100, seed 13007, one requested OpenMP thread, 3,000 source neutrons per generation, 100 total generations.

There is **exact source-bank agreement through batch 22**. Both 20- and 40-inactive histories have the **same native `k21` and `k22`**, but at the end of batch22 they select *different feedback values for the following transport generation*:

- 20-inactive run is active and uses `(k21 + k22)/2`.
- 40-inactive run is still inactive and uses `k22`.

Since `nu_t ∝ 1 / keff_feedback`, the relative expected fission-site production **20-inactive versus 40-inactive** is `k22 / ((k21+k22)/2)`.

| Original physical neutron transport observable | Broad initial source | Compact initial source |
|---|---:|---:|
| Genuine HDF5 `k21` | 0.012533344926384056 | 0.011306548497355299 |
| Genuine HDF5 `k22` | 0.013597194591145669 | 0.011179702400800642 |
| Feedback k for next generation: 20 inactive | 0.013065269758764863 | 0.011243125449077971 |
| Feedback k for next generation: 40 inactive | 0.013597194591145669 | 0.011179702400800642 |
| **Expected relative neutron fission production** `20/40 - 1` | **+4.0712885551%** | **−0.5641051375%** |
| Exact original fission-source bank equality through batch22 | YES | YES |
| First source-bank difference | **Batch23** | **Batch23** |

At the first changed source bank, broad and compact modes had their first distinct original structured source sites at one-based entries **4** and **184**, respectively (Stage43). The expected-production ratio is **not** the percentage of bank sites that differ; stochastic offspring sampling and bank synchronization determine the positions.

## Qualification and next actual core-physics work

The same released OpenMC C++ code exhibits a numerical feedback mechanism matching the timing and direction of the first observed source-bank divergence. This **strongly explains the source-path sensitivity** without assuming tally scoring or multiple OpenMP threads are the cause. An exact single-particle stochastic ancestry *causal proof* would require a separate isolated C++ intervention; no such intervention has been run.

Thus same seed plus different inactive-batch counts is **not a valid common-random-numbers paired experiment** for heating or source convergence here. The earlier nominal 20-vs-40 source TV cannot be interpreted as an isolated *source-convergence bias* estimate.

The **reactor remains a cold, highly subcritical (~k=0.012) NERVA-derived surrogate**. These original neutron/photon HDF5 runs do NOT establish stationary source, independent genealogical uncertainty, historical NASA NERVA geometry, temperature-resolved evaluated neutron scattering, coupled hot-core material feedback or physical reactor thermal W/cm³.

**Priority next:** stop spending most of the program on near-duplicate Monte Carlo burn-in experiments; independently verify representative historical NERVA physical geometry/material and temperature/XS coverage, then implement a physically consistent thermal-feedback validation case without artificial criticality tuning or fake thermal power.

## Reproducibility artifacts

- [Successful original physics GitHub Actions 40759925867](https://github.com/dubakoor13/NUKE/actions/runs/40759925867), **10/10 tests** including 32 genuine original HDF5 statepoint checks.
- [Complete verified original HDF5 + pinned upstream C++ feedback JSON, artifact 12846663913](https://github.com/dubakoor13/NUKE/actions/runs/40759925867/artifacts/12846663913).
- [Source archive: 56 original native HDF5 from Stage43](https://github.com/dubakoor13/NUKE/actions/runs/40757722214/artifacts/12847760477). Artifacts expire around **2026-11-18**; retain source provenance/evaluated data hashes alongside them.
