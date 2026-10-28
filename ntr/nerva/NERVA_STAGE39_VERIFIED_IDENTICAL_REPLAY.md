# Stage 39 — original OpenMC C++ identical-setting replay, verified

**Verified physics:** [successful original neutron/photon OpenMC run 40751331777](https://github.com/dubakoor13/NUKE/actions/runs/40751331777) (11/11 mathematical and 2/2 mandatory 24-native-HDF5 tests). Four genuinely NEW 100-generation, 20-inactive, 3,000-particles/generation simulations exactly repeated original Stage37 broad/compact initial source and seeds 13007/23011. Evaluated neutron/photon nuclear-data index, full OpenMC `geometry.xml`, `materials.xml`, `tallies.xml`, **`settings.xml`** SHA256, temperatures, fuel, tallies and photon transport were identical for the independently executed reference/replay pair.

## Actual numerical reproducibility and important precision distinction

The full **original 3,000-source-site HDF5 `source_bank` arrays** at batches **030, 060, 100** were **bit-for-bit identical across reference and replay in all four cases**. Native **source entropy arrays were also bit-for-bit identical at all checkpoints**.

By contrast, `k_generation` values differed in the least significant floating-point digits, max absolute deviation 8.67×10^-18 to 1.73×10^-17, with first unequal representation at generation 1–2. The algorithm deliberately reports **strict full-trajectory bitwise equality as FALSE** because it includes `k_generation`; this is **not evidence** of a changed physical source bank. The replicated source trajectories are numerically identical by the stronger direct source-bank and entropy evidence.

| Original exact-setting replay | Broad seed 13007 | Broad seed 23011 | Compact seed 13007 | Compact seed 23011 |
| --- | ---: | ---: | ---: | ---: |
| Every saved 3,000-source-site native bank bitwise identical | YES | YES | YES | YES |
| Exact entropy history | YES | YES | YES | YES |
| Maximum absolute k-generation difference | 8.67e-18 | 1.39e-17 | 1.73e-17 | 8.67e-18 |
| Batch-100 source histogram TV | 0 | 0 | 0 | 0 |
| Deposited-energy totals per three original active windows | Same to float precision | Same | Same | Same |

Original neutron/photon/electron/positron `heating` scores vs spatial/material partitions conserved. The observed full 12×12×10 heating shape TV per window in identical replay was on the order of **1e-18 to 1e-17**, numerical roundoff rather than a physical source-shape change. Exact native field bitwise identity is a different requirement from physical/reproducible energy to floating-point summation precision.

## Consequence for Stage 38 burn-in experiment

Four original Stage38 20-vs-40 inactive-history comparisons at otherwise unchanged settings showed larger genuine source differences: the first significant `k_generation` discrepancy >1e-8 and the first source entropy discrepancy were both at **generation 23 for all four modes/seeds**. Stage38 final source-bank 4×4×10 total variation averaged **0.5756667**, and late spatial heat TV **0.1344351**. Stage39's **unchanged-control** exact source-bank replay contradicts attributing that large variation merely to unavoidable noise in repeated 20-inactive runs.

This evidence **supports sensitivity of source tracks to changed active/inactive tally scheduling under the examined OpenMC executable setup**, but causal attribution still requires a three-way audit of the exact same source labels and all original HDF5, careful control of execution threading, and source offspring order. A minute `k` rounding difference alone must never be labeled a neutron ancestry change.

## Reproducible binary provenance

- [Stage37 original genuine C++ OpenMC HDF5](https://github.com/dubakoor13/NUKE/actions/runs/40747291683/artifacts/12841948627).
- [Stage39 four exact-setting replays: 12 new native HDF5, XML and nuclear-data hashes](https://github.com/dubakoor13/NUKE/actions/runs/40751331777/artifacts/12844066614).
- [Stage39 complete source-bank, entropy, k and three-window 3D heating JSON](https://github.com/dubakoor13/NUKE/actions/runs/40751331777/artifacts/12844126525).
- [Stage38 real 40-inactive transport HDF5](https://github.com/dubakoor13/NUKE/actions/runs/40749064255/artifacts/12843212697).

Artifacts have time-limited retention until approximately 2026-11-18. The cold far-subcritical three-ring core (k~0.012) is not a physically validated historical hot NERVA reactor; no reactor watts, source stationarity or independent-neutron confidence coverage is claimed.

**Next physics gate:** an audit comparing THREE actual physical executions per source/seed: original 20-inactive, unchanged 20-inactive replay, and 40-inactive variant, with tolerance-aware k, **exact native source-bank** and source-entropy checks, and spatial heating/energy closure.
