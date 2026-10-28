# Stage 41 — verified single-OpenMP-thread genuine OpenMC source-path and heating controls

**Physics milestone, verified 2026-10-28 UTC / 2026-10-29 IST.**

This is native **OpenMC C++ v0.15.3 neutron/photon eigenvalue transport** with evaluated nuclear-data index and unchanged cold three-ring/19-fuel-element demonstrator, **not a validated operating hot NASA NERVA reactor**. The purpose is to isolate whether Stage38–40's large source changes when increasing inactive batches were merely due to using two OpenMP threads.

## Executed independent genuine transport and provenance

- [Original six 1-OpenMP-thread OpenMC C++ simulations](https://github.com/dubakoor13/NUKE/actions/runs/40753528970/artifacts/12845287445), 12 actual HDF5 StatePoints, two source-shape initializations (legacy-broad and central-compact), each with three actual executions: original20, identical repeat20, changed40, same seed label 13007. All runs: 3,000 particles per generation, 100 total batches, coupled neutron/photon/electron/positron energy heating, original native HDF5 checkpoints 060 and 100. `OMP_NUM_THREADS=1`, `--cpus=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. **Execution-environment request** is verified in workflow; the C++ runtime did not independently report a measured thread count to the analysis.
- The above first workflow's six original simulations and physical HDF5 audit **succeeded**, but a subsequent integration test contained a JSON field-name typo (`per_disjoint_active_window` rather than `physical_active_window`), so its overall CI conclusion was failure. This is **test plumbing**, not a neutron-transport or energy-conservation error. No physics result was discarded.
- [Successful Stage41 actual-original-HDF5 verification run 40755194172](https://github.com/dubakoor13/NUKE/actions/runs/40755194172), after fixing the test assertion, **downloaded those EXACT six completed original C++ transport binaries without re-executing**; passed **7/7 numerical/adversarial tests and 2/2 genuine 12-StatePoint HDF5 tests**, with original XML SHA, evaluated nuclear data fingerprint, original RNG seed, native source, 100 batch identity, 3D neutron/photon heating and exact whole-mesh/material/particle energy closure.
- [Accepted complete machine-readable original physics observations](https://github.com/dubakoor13/NUKE/actions/runs/40755194172/artifacts/12843889727), actual numerical JSON and original HDF5 path fingerprints. Do not replace the six original archived StatePoints with a fabricated mock.

## Measured original one-thread source transport physics

| Original OpenMC observable | Legacy broad | Central compact |
|---|---:|---:|
| 20-inactive original vs 20-inactive identical replay HDF5 source bank and entropy | **Bitwise identical** | **Bitwise identical** |
| First substantive (>1e-8) `k_generation` difference, 20 vs 40 inactive | **Generation 23** | **Generation 23** |
| First native Shannon entropy divergence, 20 vs 40 inactive | Generation 23 | Generation 23 |
| At generation 60, changed-inactive source-bank TV, 4×4×10 | 0.658000 | 0.647000 |
| At generation 100, changed-inactive source-bank TV, 4×4×10 | **0.4946667** | **0.6070000** |
| Native 61–100 original/replay 3D heating shape TV | **0** | **0** |
| Native 61–100 changed-inactive heating shape TV, 12×12×10 | **0.12730148** | **0.13635788** |
| Native 61–100 original/replay heat (eV per MC source) | 861196.902505 | 858389.325519 |
| Native 61–100 changed40 heat (eV per MC source) | 879236.628093 | 877933.377356 |

**Mean final source bank TV under a changed inactive threshold = 0.5508333333**; the same-settings replay baseline = **0.000000**. Mean late 3D coupled heating TV under changed inactive threshold = **0.1318296795**; same-settings baseline = **0.000000** to original stored precision.

These are **native executed Monte Carlo observations**, not physical reactor watts/temperature. Original 3D mesh/material/neutron/photon+electron/positron deposited-energy sums conserve for each of the six native histories. The same native first-active source mechanism is seen with ONE requested OpenMP thread and previously with TWO requested OpenMP threads (Stage40: 0.5756667 source TV / 0.1344351 heating TV, four source/seed cases). **Therefore multi-thread scheduling is NOT necessary for the observed source divergence.** This does not prove that active tally scoring consumes random numbers or isolate the exact source of the divergence.

## Relevant original C++ numerical mechanism to investigate, without modifying transport physics

In the repository's `src/simulation.cpp`, first active batch is determined by
`simulation::current_batch == settings::n_inactive + 1` and then each user tally is set active. In the original experiment, this is batch **21** vs **41**. The source history first differs materially at generation **23** in both one-thread initial-source cases. This makes tally activation and its downstream interaction with source-bank sampling an actionable C++ source-review target, **not a verified causal diagnosis**. Specifically test native particle RNG stream counters, source bank sampling, and tally callbacks across batches 20–24 in minimal deterministic source reproductions before altering the OpenMC solver.

## What is and is not complete

**Completed:** physics-level single-thread control run; six actual original OpenMC native coupled eigenvalue simulations; per-run source/entropy/k and deposited-energy partition verification; 7 mathematical tests and 2 mandatory genuine native-HDF5 tests; acceptance CI; preserved all original scientific binary data.

**Not qualified:** a stationary source distribution, independent fission genealogy, evaluated high-temperature NERVA cross sections, historic core geometry, coupled real fuel thermal feedback, physical watts, or nuclear reactor validity. The demonstrator's `k_eff≈0.012` is far below a validated NERVA reactor state. Do not convert these source-normalized heating units to W/cm³.

**Next engineering priority:** inspect and instrument OpenMC's active-tally transition RNG / fission-bank behavior at generations 20–24, first at 1 thread with unchanged underlying core. Then move decisively to historical NASA core physics reference, temperature-dependent evaluated neutron data and thermal-material feedback; adding more UI or generic status-report code is lower priority.

Original HDF5/results GitHub Actions artifacts are time-limited (listed expiration around **2026-11-18**).
