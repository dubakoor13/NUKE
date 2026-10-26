# NUKE Stage 30 — genuine coupled neutron–photon OpenMC heating transport

**Date:** 2026-10-26  
**Parent:** Stage 29 PR #143, photon evaluated-data coverage verified on the actual 0.15.3 NNDC image  
**Execution:** Dedicated [GitHub Actions workflow](https://github.com/dubakoor13/NUKE/blob/feature/nerva-real-coupled-photon-heating-stage30-20261026/.github/workflows/nerva-stage30-coupled-transport.yml)  
**Target:** advance from paper/XML/HDF5 data prerequisites to a **new genuine coupled neutron+photon core Monte Carlo run**, without modifying reactor design or claiming NASA performance.

## Physics

The current model is a **conceptual 3-ring cold NERVA-derived surrogate**. The official OpenMC 0.15.3 executable is invoked with `--photon-transport` and the actual NNDC evaluated library. No reactor criticality/control/enrichment tuning.

A single reproducible numerical-evidence run is requested with: seed 13007, 3000 starting neutrons per generation, 65 batches, 20 inactive, default carbon/zirconium/uranium conceptual fuel and model geometry, hydrogen 293.6059129982851 K, material-resolved full-domain `heating`, neutron- and photon-ParticleFilter `heating`, and mesh-resolved `heating` over a 12×12×10 Cartesian mesh. This does not model operating-temperature NERVA.

The native `StatePoint` is reopened by the official Python reader **after OpenMC has executed**. `real_coupled_photon_heating_audit.py`:

1. Checks `statepoint.65.h5` native source SHA against the actual transport provenance written by NUKE.
2. Requires native `photon_transport=True`, eigenvalue run mode, exact seed and real active-batch realizations.
3. Requires Stage 29 preflight for every modeled photon element and original evaluated photon HDF5 SHA-256 manifest.
4. Inspects **the native mesh-registered tally**, **material-filter tally**, and **one particle-filter tally** with `heating` score. The mesh/particle/material domains are compared by mean energy per Monte Carlo source, preserving all bins.
5. Reports actual neutron and photon deposition separately; retains material-by-material heat scores (including zero/uninstantiated materials). Identifies any observed nonzero photon deposition **without treating it as operating reactor power**.
6. Computes source-normalized full-domain mean residuals and identifies any mismatched estimators or spatial coverage. DOES NOT invent covariance or add individual mesh-bin SEMs to obtain total uncertainty. Native particle score conditional SEMs and original bin-level SEMs are preserved.
7. Explicitly marks spatial fission source stationarity, physical data-release qualification, historical reactor temperature/geometry matching, neutron/gamma thermal deposition NASA fidelity and source-to-watts conversion **FALSE**.

`validate_stage30_coupled_heating_arithmetic.py` provides **7 synthetic numerical integrity tests only**. It does not substitute test doubles for actual transport and does not certify gamma deposition. The GitHub Actions container stage is the real physics-evidence step.

## Required interpretation and remaining work

The Stage 30 low-statistics Monte Carlo result, if executed successfully, is a physical neutron/photon interaction snapshot of the **cold far-subcritical surrogate**. It is not proof of a source-stationary critical NERVA core; elevated-source-gamma deposition does not establish the real full-power profile.

**Do not normalize this to reactor W without separately documented physical power/source-rate.** The true next milestones remain evaluated temperature-dependent fuel/moderator neutron data, independent fission-source stationarity, statistically sound mesh convergence and uncertainty, physically complete material/geometry-to-coolant mapping, and independent NASA benchmark agreement.

**At file creation:** source code and executable CI workflow have been committed; actual CI source transport completion and numerical deposition values must be confirmed from the relevant run logs before declaring this stage validated.


## Important actual executed finding: two-species heating was incomplete

The **first actual** official OpenMC coupled-transport run [40315185396](https://github.com/dubakoor13/NUKE/actions/runs/40315185396) ran successfully, including 7/7 original arithmetic regression tests, and produced native `statepoint.65.h5` SHA-256 `83328638e716bbc04760495aa9e742c1b0e72df1b6aed5a75d30b08fdde2186d` (Actions evidence artifact ID **12646387269**, expires 2026-11-16).

Original native score observations (eV per simulated starting neutron):
- Mesh total `heating`: **889544.3360467955**.
- All material-filter bin `heating`: **889544.3360467954**, closure to roundoff.
- Two-species `nerva_particle_heating` (`neutron` + `photon`): **858736.8111715256**, a deficit of **3.4632927923728796%**, despite the scorer reporting matching estimator names.
- `neutron` heating: **858703.0924028475**; *direct photon event* heating **33.718768678086654**.
- Genuine native combined k-effective: **0.01208 ± 0.00021** (Monte Carlo nominal, source stationarity not demonstrated).

This is NOT full material/particle score closure. Reviewing **actual repository `src/tallies/tally_scoring.cpp`** showed why: `score_analog_tally_ce()` specifically scores HEATING on **electrons and positrons** although their transport is local (flux zero). A `ParticleFilter(["neutron","photon"])` necessarily omits their local energy deposition. The omitted energy likely accounts for the ~3.46% mean discrepancy; it must be verified by a *new actual* run, not merely assumed.

**Corrective Stage30 patch:** update `nerva_particle_heating` to include `["neutron","photon","electron","positron"]`, audit each species separately and define the electron/positron bins as explicit charged secondary local deposition. The revised workflow demands the mesh, material and four-particle means close on the actual native statepoint and fails if they do not. The revised run must have its own explicit confirmed numerical result. A raw direct photon event fraction alone substantially undercounts gamma-mediated electron/positron energy transfer.

**Scientific interpretation:** even after proper accounting of photon-related charged secondaries, this is the same cold, highly subcritical surrogate; evaluated-data hot-state fidelity, physical reactor watts and historical NASA reactor validation remain **UNQUALIFIED**.
