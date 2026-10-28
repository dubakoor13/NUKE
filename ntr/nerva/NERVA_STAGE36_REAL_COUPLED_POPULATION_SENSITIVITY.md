# Stage 36 — genuine OpenMC neutron-source population sensitivity (fixed cold core)

**Scope:** transport and numerical verification only. No reactor geometry, enrichment, core temperature, nuclear data, control angle, historical NERVA calibration, power normalization or cooling model is modified.

## Prior original-transport evidence

- [Stage 33 verified original OpenMC C++ v0.15.3 neutron+photon run](https://github.com/dubakoor13/NUKE/actions/runs/40741032756): four independently seeded 3,000-source-particle/generation histories (13007, 23011, 33013, 43019), 65 total batches, 20 inactive, 12 genuine HDF5 source snapshots at batch 30, 45 and 65. Observed same-seed source-site histogram mean TV (30→65) **0.6015**; source convergence **not** established.
- [Stage 34 accepted disjoint native heating windows](https://github.com/dubakoor13/NUKE/actions/runs/40742640556): mean source-normalized heat in active windows [21,30]/[31,45]/[46,65] was **908558.8314 / 850473.5912 / 886599.7145 eV per MC starting neutron**. The heating windows share correlated fission genealogy; their moments are not IID statistics.
- [Stage 35 accepted native 3D energy distribution](https://github.com/dubakoor13/NUKE/actions/runs/40743595418): original fine-mesh 12×12×10 first-vs-last heating spatial-fraction TV **0.2248836445788241**, reduced to **0.16460848172497022** when integrated axially. Neither result establishes hot-core behavior.

## New physical experiment

The Stage36 workflow uses the **same official evaluated-data OpenMC v0.15.3 C++ neutron/photon transport container**, but executes **four new original transport histories with 6,000 particles per generation**, identical four seed *labels*, 65 batches (20 inactive), the same 3-ring cold-surrogate model, source initialization, nuclear data, neutron/photon physics, and 12×12×10 heating mesh. Saves the genuine original HDF5 bank/tally checkpoints at **30, 45, and 65**, plus transport XML and SHA256 provenance.

It re-runs the previously physically verified four-seed Stage31 coupled neutron/photon heating audit for the 6,000-population results before comparing with archived exact original 3,000-population HDF5. This requires **24 genuine HDF5 source bank snapshots** and **8 independently executed source histories**, not a numerical model of what extra particles might produce.

For each of the eight actual histories:

1. Verify source-bank HDF5, seed, photon transport, particles/generation, batch index, active tallies, full cumulative `k_generation` and entropy history prefixes, source-bank SHA256, and agreement with original all-particle heating tally ledger.
2. Recover all three physically nonoverlapping heating windows from native raw `sum` and `sum_sq` (active batches 21–30, 31–45, 46–65), and check conservation for 3D mesh, all materials, and neutron/photon/electron/positron contributions.
3. Check **identical geometry, materials, tally XML SHA256 and neutron/photon evaluated nuclear-data index** across the 3,000 and 6,000 experiments. Only `particles_per_generation` may differ in the physical configuration. A changed temperature or control angle is rejected.
4. Observe the source-bank spatial TV from 30→65 and all-six-between-seed final-bank source TV at **4×4×10**; measure exact repeated site positions and inverse-Simpson diversity (NOT an independent-history count).
5. Observe mean deposited heating eV/source over the three windows, fine-mesh first-to-last **3D heating shape TV**, final-field cross-population seed-average shape TV, and between-seed shape variation. Conservatively aggregate heat on **12×12×10 → 6×6×10 → 3×3×10 → 1×1×10**, verify energy closure and monotonic non-increase of total variation with coarsening.
6. Preserve compressed original population×seed×window physical source/heating arrays as NPZ and independently interpretable summary as JSON.

**Interpretation limits:** same integer seed with a different population does **not** create a mathematically justified common-random-numbers paired experiment; populations consume RNG sequences differently. Source sites and active generations remain genealogically correlated; 4 seeds/population cannot establish calibrated coverage or fission-source stationarity, and noise can worsen even if nominal population rises. Correlated total-energy/mesh differences should not be converted to root-N confidence bands without a validated dependence model.

**Reactor qualification:** the model remains a cold, extremely subcritical conceptual NERVA-derived surrogate (prior computed `k_eff≈0.012`). Absolute power, fuel temperatures, real NASA NERVA configuration and reactor-core reactivity are not qualified. The output is eV per MC starting neutron or normalized histogram fraction, **never physical watts**.

## Acceptance

The Stage36 GitHub Actions workflow requires **11 independent mathematics/adversarial tests** and **3 mandatory tests reading real original 3,000/6,000 OpenMC StatePoints**. It fails closed if the 6,000-population experiment, original 3,000 archive, native HDF5, source checksum, materials, XML or nuclear data is absent or mismatched. It archives *actual* new 6,000 run original HDF5 and a separate JSON + NPZ population comparison, retaining the 3,000 source by its original Stage33 artifact ID. No Stage36 real numerical output is claimed before successful workflow acceptance.

## Further fundamental physics gates

Demonstrate dependence on source initialization and inactivity duration; extend generations to examine source stationarity, genealogical clustering and spatial tally bias; move from conceptual cold-surrogate geometry to a documented NASA-reference core with evaluated high-temperature data and verified thermal-material feedback. These are separate from simply increasing Monte Carlo particles/generation.


## Accepted original coupled-transport results — 2026-10-28

**Successful genuine [Stage36 GitHub Actions run 40744903368](https://github.com/dubakoor13/NUKE/actions/runs/40744903368)** on the PR's original analysis workflow. Four NEW original coupled neutron/photon OpenMC C++ simulations ran at **6,000 particles per generation**, were checked against the four archived 3,000-particle runs and generated **24 verified original HDF5 statepoints** (12 per population). The workflow completed **11/11 mathematical/adversarial tests**, **2/2 original four-seed coupled material/particle heating conservation tests**, and **3/3 real 24-HDF5 population tests**, including reproducing independent Stage33/35 3,000-particle results. Geometry/material/tally XML and evaluated neutron/photon data indices matched across populations.

### Actual Monte Carlo observations

| Observable | 3,000 particles/gen | 6,000 particles/gen |
| --- | ---: | ---: |
| Observed mean within-seed source-bank TV, batch 30→65, 4×4×10 | 0.601500 | 0.438167 |
| Mean fine 3D heating-shape TV, window 21–30 vs 46–65, 12×12×10 | 0.2248836446 | 0.1674812125 |
| Mean integrated axial heating TV, same disjoint windows, 1×1×10 | 0.1646084817 | 0.1228793978 |
| Early deposited energy, 21–30, eV per MC source (4-seed mean) | 908558.831 | 839238.131 |
| Middle deposited energy, 31–45, eV per MC source (4-seed mean) | 850473.591 | 856146.884 |
| Late deposited energy, 46–65, eV per MC source (4-seed mean) | 886599.714 | 854853.510 |
| Late per-seed deposited energy sample SD, eV/source | 25341.994 | 10418.531 |
| Average final bank fraction of EXACT distinct source-site positions | 0.027250 | 0.026167 |
| Final bank inverse-Simpson distinct-position diversity index (mean over 4 seeds) | 29.8293 | 56.3403 |
| Final bank largest repeated-position weight fraction (mean over 4 seeds) | 0.061750 | 0.027792 |

**Cross-population source final-seed-average spatial TV = 0.2786666667**, a nontrivial difference in the normalized 4×4×10 fission-source histogram.

**Cross-population late-window heating-fraction total variation (4-seed mean spatial fields):** fine 12×12×10 **0.0847103893**, coarsened 6×6×10 **0.0701093217**, 3×3×10 **0.0648015695**, integrated axial 1×1×10 **0.0613942419**. TV decreases under conservative mesh coarsening, as it must.

The observed source-bank 30→65 mean TV decreased by about **27.2%**, and the fine heating-window shape TV by about **25.5%**, on doubling population. These numerical observations **do not measure or prove convergence rates**, and are not calibrated uncertainty reductions.

**Critical physical warning:** Only about 2.6–2.7% of saved source sites have unique *exact coordinates* in the final resampled source banks. This does NOT imply 2.6% independent neutron histories: repeated locations are an observable signature of resampling/cloning, but fission descendants may be related even when coordinates differ. The diversity-index values above **are not independent fission-history counts**. All statistical and historical hot-core qualification flags remain FALSE.

### Original machine-readable physics evidence

- 6,000-particle **12 actual OpenMC HDF5**, full inputs/cross-section audit and transport provenance: artifact [12840923642](https://github.com/dubakoor13/NUKE/actions/runs/40744903368/artifacts/12840923642).
- Verified JSON comparison and compressed NPZ holding full 2×4×3 source-bank histograms and deposited-heating fields: artifact [12841632050](https://github.com/dubakoor13/NUKE/actions/runs/40744903368/artifacts/12841632050).
- Original 3,000 reference HDF5 archive: artifact [12839092884](https://github.com/dubakoor13/NUKE/actions/runs/40741032756/artifacts/12839092884).

All Stage36 artifacts have limited GitHub retention (listed expiration **2026-11-18**); retain the HDF5 original binaries and original evaluated data index, not just reduced JSON, for a reproducible benchmark.

### Interpretation and next numerical priority

This experiment supports a **population-size-sensitive spatial source and heating model**. It does not establish stationarity or physical accuracy of a hot NERVA core: the cold low-`k_eff` conceptual surrogate has substantial source resampling/cloning, and different populations show different source distributions and energy fields. **Next implement real source-initialization and active/inactive-generation sensitivity** with longer independent neutron histories, plus source ancestry analysis. Historical NASA geometry and temperature-qualified evaluated nuclear data must be established before any physically interpreted hot-core heat flux.
