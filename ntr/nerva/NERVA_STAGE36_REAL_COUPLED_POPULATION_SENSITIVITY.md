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
