# NUKE Stage 26 — genuine OpenMC root-connected geometry evidence

**Date:** 2026-10-26  
**Parent:** Stage 25 [PR #138](https://github.com/dubakoor13/NUKE/pull/138).  
**Source:** original four-seed OpenMC 0.15.3 **cold-surrogate** statepoints and `summary.h5` from [Actions run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638). This work does not run new transport or modify design parameters.

## Scientific gap

Stage 25's `_summary_materials_and_cells` reconstructs material records and enumerates every HDF5 `geometry/cells` group with `fill_type=material`. Such a list, even when HDF5/XML identity is verified, does **not by itself** establish that each cell belongs to a chain reachable from OpenMC's root universe.

An unconnected cell or universe may be serialized without participating in the root-connected geometry. Equating **declared materials**, **native cell definitions**, **root-connected cell definitions**, **nonempty physical regions**, and **actual neutron histories that visited those regions** is scientifically incorrect. These have distinct evidence requirements.

## Implemented

`ntr/nerva/real_native_root_topology.py` reads *native binary summary HDF5* for the same original 4 source seeds. It does not recreate a Python geometry from a source file. It:

1. Extracts native material, cell, universe, lattice IDs and fill references, including lattice outer universe.
2. Checks that references resolve, cell membership is unambiguous, there is exactly one root not used as a fill, and no reachable universe/lattice cycles.
3. Traverses the root reference graph, distinguishing all HDF5 material-filled cell definitions from those included by a root path.
4. Rejects Stage 25's claim if any material-filled native cell definition was actually non-root, or if root-connected material IDs/count differ.
5. Compares a canonical HDF5 connectivity fingerprint across four independent RNG seeds to catch different topology being incorrectly pooled as one experiment.
6. Emits a JSON record with exact root ID, connected universes/lattices/cells/materials, orphan cell definitions, and native HDF5 SHA.
7. Preserves the scientific **BLOCKED** status of hot criticality, neutron-history visitation, spatial heat deposition and NASA historic fidelity.

This is an **independent native topology** audit, not a proof of cell volume or validated geometry dimensions. The same cell definition can appear at multiple lattice instances; its ID is counted once. A reference path to a cell is not proof that the corresponding space is nonempty.

## Checks

`validate_real_native_root_topology.py` contains **7 isolated parser tests** using deliberately artificial HDF5 reference graphs: nested universe, lattice, orphan cell, nonunique root, missing material, missing lattice universe, reference cycle, and topology hash behavior (seven total tests). Those fixtures are for algorithm verification **only**, not OpenMC transport evidence.

It also contains **2 actual-data checks** requiring the genuine archived OpenMC HDF5 data:

- Four genuine summaries must have a consistent root reference topology and agree with Stage 25's material reachability/count statements.
- Passing a topology audit must *never* mark reactor operating qualification, real physical visitation or local nuclear energy deposition as demonstrated.

The [native root CI workflow](../../.github/workflows/nerva-native-root-topology.yml) fails when the original artifact is missing, and does not create synthetic replacements.

## Status and follow-on physics gates

**Before Stage 26 real-source CI succeeds**, the following Stage 25 counts remain observations for native cell definitions rather than independently proven root-connected definitions: 60 HDF5 material-filled cells/seed; material IDs 19–22; four materially involved types. Do not label this stage passed until original archived HDF5 has been inspected by the new code.

Even after a pass, the following remain **unqualified**: independent geometry volume/instancing, genuine neutron source stationarity, evaluated hot neutron/photon materials and scattering data, nonzero material-separated local deposited power, thermal power normalization, matched temperatures/pressure states, and NASA experimental benchmarks.

**P0 reminder:** Original actions artifact is described as expiring **2026-11-13**. Preserve the actual original bytes/checksums and the true evaluated neutron isotope HDF5 binaries. A geometry graph audit cannot substitute for missing nuclear data.

This stage addresses an evidence *interpretation* error; it is not a reactor performance upgrade and does not justify decrementing the broader 60–70% subjective historical-fidelity gap by itself.


## Executed CI evidence — 2026-10-26

**Original-data run [40300596727](https://github.com/dubakoor13/NUKE/actions/runs/40300596727): SUCCESS.** Original archived ZIP artifact 12492499186 was downloaded and its artifact digest was reported as `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`.

- **7/7 synthetic native-graph parser tests passed**, including malformed/ambiguous references, unassigned cell and cycle.
- **2/2 genuine original OpenMC HDF5 tests passed** on native summaries for seeds 13007, 23011, 33013, 43019.
- All four native HDF5 root graphs report material IDs **19–22**, **60 distinct root-connected material cell definitions per seed**, and no material-filled cell definition outside root connectivity.
- The four source graphs' canonical connectivity fingerprints agree. The JSON record is retained in the Actions run as `nuke-stage26-original-native-root-topology-report`.
- This **upgrades Stage 25's HDF5 material inventory statement to verified native root connectivity**, not to experimentally occupied volume or neutron visitation. Real hot-core transport, evaluated nuclear data, source stationarity, local deposited heating and historical fidelity remain **BLOCKED**.

This scientific audit's pass should not be counted as a hot reactor physics validation or proof of criticality.
