# Stage 40 — verified three-arm coupled OpenMC native source-trajectory control

**Date:** 2026-10-28 UTC / 2026-10-29 India. **Physics:** fixed cold NERVA-like 19-fuel-element surrogate, `k≈0.012`; official OpenMC C++ v0.15.3 neutron+photon eigensolver, 3,000 source neutrons/generation, 100 batches, same physical material/geometry/evaluated neutron/photon data. **Historical hot NERVA and reactor watts NOT qualified.**

## Three separately executed original Monte Carlo arms

1. **A — original 20-inactive:** [Stage37 genuine original OpenMC HDF5 archive](https://github.com/dubakoor13/NUKE/actions/runs/40747291683/artifacts/12841948627).
2. **B — unchanged 20-inactive:** [Stage39 physically identical replay original OpenMC HDF5 archive](https://github.com/dubakoor13/NUKE/actions/runs/40751331777/artifacts/12844066614). Source box, seed, materials, `settings.xml`, `geometry.xml`, `tallies.xml`, `materials.xml` SHA256, nuclear data index and source-population configuration identical to arm A.
3. **C — 40-inactive instead:** [Stage38 genuinely executed OpenMC HDF5 archive](https://github.com/dubakoor13/NUKE/actions/runs/40749064255/artifacts/12843212697). The same physical reactor and source parameters, but a different active/inactive tally schedule.

Four independent model/source/seed *configurations* are examined in all three arms: legacy-broad/central-compact sources × seed labels 13007/23011. For each arm and configuration the full native source at batches 060 and 100 is required, and the pair of saved cumulative native heating moments gives a genuine last active window **61–100**. The Stage39 reference/replay auditor additionally validates originals at batch 030. The Stage38 reference/40-inactive auditor checks native k/entropy, transport provenance, scorer identities and exact material/neutron/photon/electron/positron energy sums.

**[Successful Stage40 verification workflow 40752516393](https://github.com/dubakoor13/NUKE/actions/runs/40752516393): 11 numerical/adversarial plus 2 actual 24-native-HDF5 tests ALL PASSED.** Original sources / same-source replay / changed-inactive native HDF5 were reopened via the official pinned OpenMC v0.15.3 Python StatePoint reader. No newly fabricated neutron data or new transport was substituted for original executed physics.

## Direct results — original native source genealogy

| Observable | A vs B — same 20 inactive | A vs C — 20→40 inactive |
| --- | ---: | ---: |
| Full native 3,000-source-site bank equality at 60/100 in all 4 configs | **Exact identical** | **Different** |
| Original generation source entropy array | **Exact identical** | Different |
| Maximum k-generation difference in unchanged replay | ≤1.7347e-17 | Substantially larger |
| Four-case mean final 4×4×10 fission-source histogram TV at 100 | **0.000000** | **0.5756666667** |
| Four-case mean same-window 61–100 12×12×10 deposited-heating shape TV | **9.8402e-18** | **0.1344351486** |
| First meaningful k difference >1e-8 and source entropy difference | **None** | **Generation 23, all 4 configs** |
| Original neutron/photon+electron/positron material/mesh heating sum closure | Passed | Passed |

Per-case actual late-window physical source-normalized deposited energy and source variation:

| Original source mode / seed | A vs C source bank TV at 100 | A vs C late 3D heat TV | A/B late heating (eV/MC start) | C late heating (eV/MC start) |
| --- | ---: | ---: | ---: | ---: |
| Broad / 13007 | 0.4946667 | 0.1273015 | 861196.90 | 879236.63 |
| Broad / 23011 | 0.5603333 | 0.1714684 | 853130.70 | 839557.78 |
| Compact / 13007 | 0.6070000 | 0.1363579 | 858389.33 | 877933.38 |
| Compact / 23011 | 0.6406667 | 0.1026128 | 865645.68 | 855191.48 |

**Interpretation:** A and B demonstrate highly repeatable physical neutron source tracks and deposited-energy fields in an identical-configuration control, despite harmless last-digit floating-point changes in `k_generation`. Altering only the inactive-generation tally schedule is **associated** with a much larger source genealogy/entropy divergence at generation 23, and nontrivial 3D heating changes. **Numerical equality of original seed labels is not sufficient to infer any cross-run pairing** until source arrays are inspected.

The likely next task is to investigate **OpenMC RNG consumption/source bank handling around the first active batch**, assess CPU threading sensitivity with a single-thread controlled run, and distinguish a reproducible inactive-setting-dependent algorithm effect from stochastic scheduling differences. With only 4 source configurations, the exact internal causal mechanism is not proven; neither source stationary distribution nor genealogical independence is statistically certified.

## Reproducible accepted machine-readable evidence

[Stage40 original three-arm complete JSON result artifact 12843413749](https://github.com/dubakoor13/NUKE/actions/runs/40752516393/artifacts/12843413749) includes per-source/seed full HDF5/entropy/native-k identity checks and original heating values. Original archives from Stages37/38/39 should be preserved before their current approximately **2026-11-18** expiration. No assumption of hot historical NASA NERVA reference fidelity or physical reactor W/cm³ is made.

**Status:** numerical reproducibility control verified; true source stationarity, historical NASA core geometry, temperature-resolved data, and coupled hot reactor heating remain material physics goals.
