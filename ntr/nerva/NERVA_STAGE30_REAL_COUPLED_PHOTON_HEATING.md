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
