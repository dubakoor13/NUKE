# Stage 43 — verified first native neutron fission-source bank divergence

**Accepted source physics:** genuine official OpenMC C++ v0.15.3 neutron+photon eigenvalue Monte Carlo; fixed cold, highly subcritical NERVA-derived three-ring surrogate (`k≈0.012`), evaluated nuclear data, original geometry/materials and initial fission source. This is NOT a physical hot NASA NERVA core model or watt-level reactor power result.

## Actual original physical runs and acceptance

[Successful Stage43 numerical physics workflow 40757722214](https://github.com/dubakoor13/NUKE/actions/runs/40757722214) ran four NEW original coupled C++ OpenMC neutron/photon histories with **zero user scoring tallies**. Source mode: legacy broad or central compact; inactive 20 or 40; seed 13007, **3,000 starting neutrons per generation**, 100 total batches, **one requested OpenMP thread**.

Each of the four source histories saved the actual full 3,000-site native HDF5 fission bank at **batches 19,20,21,22,23,24,25,26,27,28,29,30,60,100**, giving **56 original C++ HDF5 files**. Native original file type, eigenvalue/photon mode, true source bank, generation count/seed, original active realizations, entropy/`k_generation` prefix invariants and evaluated-data/material/geometry input SHA were verified. Inactive checkpoint writing requires **both** the diagnostic no-user-tally flag and explicitly enabled inactive checkpoints, leaving the normal production runner's safeguards unchanged.

The four original Stage43 histories were **physically reproduced** against the four previously executed Stage42 source-only OpenMC histories at original source checkpoints **60 and 100**: full native fission-bank arrays and source-entropy histories exact; original `k` trajectories agreed to <1e-14. Thus the extra 19–30 HDF5 instrumentation **does not explain the source divergence**.

**Five adversarial numerical tests and two mandatory REAL 56-HDF5 tests PASSED.** No fabricated source or synthetic HDF5 is accepted as reactor evidence.

## Key C++ eigenvalue neutron source result

| Native completed OpenMC batch | Legacy broad: 20 vs 40 inactive | Central compact: 20 vs 40 inactive |
| --- | --- | --- |
| 19 | Exact same full 3000-site fission bank, entropy and k | Exact same full bank, entropy and k |
| 20 | Exact same | Exact same |
| 21 | Exact same | Exact same |
| 22 | **Exact same** | **Exact same** |
| **23** | **First bank, entropy and meaningful k difference** | **First bank, entropy and meaningful k difference** |
| 24 | Differences persist | Differences persist |
| 25–30 | Differences persist | Differences persist |
| 60 and 100 | Different 20/40 source trajectories; original Stage42 replay controls exact | Same |

At **batch 23**:

| Original native observable | Broad source | Compact source |
| --- | ---: | ---: |
| First source-bank element with changed original structured source site (1-based) | **4** | **184** |
| Max absolute `k_generation` difference observed through batch 23 | **0.0003625430879797796** | **0.00002810638154856354** |
| Original entropy histories between inactive variants identical through batch 23? | **No** | **No** |
| Original source banks at batch 22 identical? | **Yes** | **Yes** |

At batch 24, the max cumulative generation-`k` discrepancy rose to **0.001034** (broad) and **0.002028** (compact). The first changed entry in the full bank was index 1 for broad at batch24 and index 2 for compact.

## Interpretation

Stages 39–41 established that exact same-configuration repeats reproduce native source banks; changing the inactive threshold causes substantial divergence even on a single requested OpenMP thread. Stage42 established that **removing ALL user scoring tallies leaves the source-bank/entropy/k histories exactly unchanged at any fixed inactive threshold**, disproving the idea that scoring-tally activation is necessary to cause the divergence.

**Stage43 localizes the first native fission-source divergence precisely to batch 23 for BOTH initial source modes, after exact equality through batch 22.** The original C++ `src/simulation.cpp` marks the first active batch at `n_inactive+1` (batch 21 or 41), so source divergence is delayed **two completed batches** relative to the first active batch in the 20-inactive case. This timing is reproducible but **does not yet prove which OpenMC C++ function causes it**.

Next genuine core-numerics investigation: focus on OpenMC `initialize_batch`, `finalize_batch`, `calculate_generation_keff`, `calculate_average_keff`, global source-bank normalization/synchronization and per-particle RNG stream derivation across batches **20–24**. Trace or audit original C++ bookkeeping; do not change reactor enrichment, geometry or control angle to make `k_eff` higher. Do not confuse source-bank equality, entropy and generation-`k` with a stationary eigenvalue source or statistically calibrated uncertainty.

Original source-only output **cannot score deposited energy**, estimate watts or qualify a thermally coupled hot reactor. Historic NASA core geometry/temperature-resolved evaluated data and real reactor thermal feedback remain larger separate objectives.

## Original scientific machine-readable evidence

- [Stage43 all 56 original neutron/photon StatePoint HDF5, source input XML, nuclear data provenance](https://github.com/dubakoor13/NUKE/actions/runs/40757722214/artifacts/12847760477).
- [Stage43 full per-generation original bank SHA256/element mismatch, entropy and k observation JSON](https://github.com/dubakoor13/NUKE/actions/runs/40757722214/artifacts/12847446944).
- [Original Stage42 source-only HDF5 replay control](https://github.com/dubakoor13/NUKE/actions/runs/40756403944/artifacts/12846765965).

These GitHub Actions artifacts have limited retention until approximately **2026-11-18**; save the original binaries with evaluated data fingerprints for independent reproducibility.
