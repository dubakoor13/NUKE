# NUKE / NERVA — MASTER CORE PHYSICS, NUMERICS, EVIDENCE AND VALIDATION TODO

**As-of:** 2026-10-25  
**Repository:** https://github.com/dubakoor13/NUKE  
**Working snapshot:** feature/nerva-real-transport-evidence-gates-stage21-20261025 (Stage 21, stacked on PR #132)  
**Relationship to main:** IMPORTANT — much of the Stage 6–20 FX2-TH chain is in OPEN, STACKED PULL REQUESTS, NOT MERGED INTO main.  
**Primary objective:** an evidence-traceable, numerically robust and independently physically benchmarked NERVA-type nuclear thermal rocket **reactor-core** model.  
**Priority requested:** reactor-core physics, numerics, validation and conservation FIRST. Engine cycle, optimization, UI, and repository housekeeping are secondary unless they block verified core physics.  
**Document purpose:** persistent, detailed engineering handoff and executable work queue. Not a declaration of reactor qualification or percent-complete certification.

> **Critical assessment:** NUKE has substantial real source code for OpenMC modeling, transport diagnostics, spatial multigroup diffusion, delayed-neutron kinetics, eigen/adjoint mathematics, temperature-table interpolation, radial/axial fuel conduction, and hydrogen enthalpy. What is NOT yet established is a single **physically qualified, hot NERVA core** that combines a verified historical configuration, evaluated temperature-/composition-dependent nuclear data, independently converged genuine OpenMC neutron/photon transport, correctly normalized spatial nuclear heat, justified fuel-to-neutronics mapping, and pressure-/density-consistent hydrogen thermohydraulics, then agrees with independent NASA measurements. Numerical regression success alone does not close these gaps.

---

## 0. How to use and update this file

1. Treat each work item with an ID (such as DATA-01, TRAN-04, HEAT-07) as a standalone task that requires an evidence artifact and explicit acceptance decision.
2. Keep the **implemented**, **numerically verified**, **transport demonstrated**, and **historically validated** statuses separate.
3. Do not declare an item complete merely because a Python file, function name, workflow or synthetic unit test exists.
4. Prefer implementing and reviewing ONE physical interface or governing-equation gap at a time, without substituting artificial physics for missing evaluated inputs.
5. Record every subsequent actual run with exact commit, branch, input-file SHA-256, evaluated nuclear-library manifest, solver version, seeds, settings, outputs, statistical diagnostics, independent comparison and limitations.
6. Update task checkboxes only when the listed acceptance evidence exists. A completed coding subtask does not imply its physical-validation subtask is complete.
7. This is the current plan for the **NERVA core**. Existing full-engine/tie-tube projects are catalogued as dependencies, not incorrectly classified as unimplemented.

### Evidence maturity vocabulary

| Code | Meaning | Acceptable evidence | Not sufficient |
|---|---|---|---|
| I | Implemented | Source exists at a named commit and import/run interface can be inspected | A stated intention or a PR title |
| N | Numerically verified | Independent analytic/dense/reference tests, convergence, units and conservation for exercised cases | A green workflow without checking what was tested |
| T | Transport demonstrated | Real OpenMC executable + evaluated data ran on a specified core model, statepoints, active-batch/seed convergence, uncertainty and artifact provenance retained | A mocked OpenMC API, synthetic arrays or merely generated XML |
| P | Physically compared | Same reactor/material/test conditions compared to independent trusted measurements with approved accuracy and uncertainty | Matching one tuned number or a textbook derivation |
| H | Historical NERVA fidelity demonstrated | Fully traceable named hardware/configuration and multiple NASA points, no hidden fitted unknowns, independent benchmarks | A generic NERVA-derived model with plausible geometry |

**Current global maturity is mixed:** many components have I/N; the core hot-model claim has NOT met all T/P/H gates. A defensible aggregate percentage is **not presently measurable** without a fixed requirements list and accepted evidence per gate. The older status document's 30–40% completion / 60–70% remaining estimate is an explicitly subjective *2026-10-21* planning estimate, not a newly audited percentage. Avoid presenting it as an objective current figure.

### Scope / non-goals for the next work block

**IN SCOPE:** transport and nuclear-data provenance, flux-spectrum/eigen/adjoint and precursors, group condensation, neutron/photon heating, reactor power normalization, 3-D/2-D coupling geometry, fuel/coolant/reflector material feedback, conduction/enthalpy and strict numerical validation.

**LATER / NOT A BLOCKER:** polishing dashboards, speculative LANTR/methane variants, thrust/Isp UI, unrelated propulsion repositories, whole-engine cycle optimization, GUI, original FORTRAN archaeology unless it provides essential independent physics reference, and C++ acceleration before correctness is established.

---

## 1. At-a-glance subsystem status — what we HAVE vs what is NEEDED

| Subsystem | HAVE today (source-code level) | Verified scope | Main missing core-physics requirement | Priority |
|---|---|---|---|---|
| OpenMC codebase | Full OpenMC source fork under openmc/ and NERVA modeling scripts | Software/tooling exists; upstream OpenMC tests are distinct | Reproducibly execute and archive a **specific** NERVA-like core with evaluated cross sections and credible statistical convergence | P0 |
| Core geometry | Hexagonal fuel lattice, 19-channel element, tie/support elements, graphite filler, Be reflector and control drums in NERVA-derived demonstrator | Geometry/export/consistency checks exist | Qualified dimensions, as-built materials, element/rod inventories, thermal expansion, drum absorber geometry and full cross-section map for one chosen historical target | P0 |
| Materials/nuclear data | OpenMC materials, source/temperature/kerma/thermal-scattering audit modules, source integrity checks | Auditing/guards and surrogate configurations | Manifest of actual evaluated isotope HDF5/temperatures, bound scattering, fission spectra, thermal kernels and data provenance for **every root-reachable material** | P0 |
| Real criticality and transport QA | run_openmc_transport and real-core convergence/neutron-balance/replica-statistics modules; **real 4-seed cold surrogate statepoints archived in Actions run 39933618638** | Genuine OpenMC 0.15.3 four-seed neutron-only HDF5 binaries verified; source stationarity and hot model NOT qualified | Independently converged source and matched historical/hot operating configuration, not another mere cold smoke run | P0 |
| Multigroup diffusion | Stage 2/11 Cartesian and Stage 12 finite-volume cylindrical r-z | Synthetic analytic cylindrical buckling, neutron balance and operator checks | Transport-derived, state-dependent multigroup coefficients on matching geometry; numerical and physical comparison with OpenMC | P0–P1 |
| Prompt/delayed neutron source | Stage 6/7/15 full delayed-family birth operators | Synthetic equilibrium and family spectrum/adjoint checks | Qualified fission-yield, delayed fractions and spectra from appropriate evaluated data for actual fuel isotopes | P1 |
| Spatial transient kinetics | Stage 7 backward Euler, Stage 13 adaptive BE, Stage 14 second-order L-stable SDIRK2 | Independent synthetic dense matrix exponential, stiff/time-step tests | Independent transport kinetics benchmarks; consistent physical source normalization / dynamic feedback; control worth | P1 |
| Generalized eigen/adjoint | Stage 12/15 forward, adjoint, beta-effective calculations | Synthetic dense generalized eigenvalue comparisons | Compare k, worth, generation time, beta-eff and shapes to OpenMC or independent transport; uncertainty | P1 |
| Adjoint sensitivities | Stage 16 d(k)/d(parameter) and d(rho)/d(parameter) | Perturbed dense eigenvalue finite differences | Physical dXS/dT, density feedback, real Doppler and moderator/reflector reactivity coefficients | P1 |
| Temperature tables | Stage 17 spatial multigroup coefficient/spectrum table interpolation | Synthetic piecewise-linear table derivative tests; fail-closed extrapolation guards | Actual evaluated and condensed tables; temperatures/densities/phase coverage; source spectral consistency and qualification | P0 |
| Spatial neutron-temperature coupling | Stage 8/18/19/20 Picard and small monolithic benchmark | Synthetic independent solver equality, residual/conservation tests | Consistent hot-core input state, true nuclear heating, feedback for ALL relevant materials, dynamic thermal-fluid coupling | P0–P1 |
| Neutron/photon deposited power | Main heat_partition and Stage 9/10 OpenMC remapping/statepoint evidence modules | Synthetic conservation, source-rate/units/mesh/statistics checks | Genuine separately scored real OpenMC heating from matched core, normalized source power and mapped without volume/material double counts | P0 |
| Fuel thermal physics | Screening fuel_conduction, resolved_transient and Stage 19 annular radial/axial FV | Log cylindrical resistance, axial conduction, sparse first-law tests | Actual fuel-element/coating/graphite geometry, k(T), cp(T), contact conductance, axial/tie radiative and boundary conditions, thermal stresses | P1 |
| Hydrogen thermal physics | NASA-7 frozen/equilibrium H2/H modules, CoolProp backend, existing cryogenic tie modules, Stage 20 variable enthalpy | Pure molecular NASA-7 h(T) in 200–3500 K numerically checked; separate CoolProp models exist | Unified h(P,T), rho(P,T), phase, real-fluid conditions, flow momentum/pressure, dissociation at applicable states | P1 |
| Coolant network | Channel / tie tube / header / Euler / pressure-driven solvers under main | Separate reduced-order conservative fluid regressions | Match actual core flow layout, connect into Stage 20 while solving pressure, density, mass/momentum, energy at consistent state | P1–P2 |
| Power and full NTP engine | Nozzle, pumps, turbine bleed, LH2 cycle and thrust/Isp paths under main | Reduced-order and synthetic thermo/cycle checks | Independent core thermal-power closure and measured station boundary conditions before engine predictions are credible | P2 |
| Nuclear uncertainty | Replica/mesh/source/tally/covariance diagnostics under main, Stage 10 independent seeds | Instrumentation and numerical statistics code | Real repeated-run evidence and full propagated nuclear/material/thermal uncertainty; stable confidence intervals | P0–P1 |
| Historical validation | NASA references, NERVA fidelity matrix, Emrich 2023 crosswalk, status handoff | Source catalog/index available | Lock one actual XE/NRX/1972 variant; digitize NASA source tables + uncertainty; benchmark multiple points | P0 |
| Merge/release integration | Many separate open PRs and dedicated GitHub workflows | Stage-specific successful workflows | Resolve stacked PR dependencies/duplicate implementations, merge in tested order and rerun all cross-module checks from a single commit | P0 integration |

**IMPORTANT BRANCH DISTINCTION:** “HAVE” in this table includes code on the **Stage 20 feature branch**. Do not claim every FX2-TH module is already on main.

---

## 2. Repository inventory — existing baseline on main

This inventory is grounded in inspected paths on the 2026-10-24 repository tree; a file's presence is NOT evidence of historical accuracy or successful real OpenMC execution.

### 2.1 OpenMC and model construction

- openmc/ — imported/upstream Monte Carlo transport source tree, including its own tests and data. The bundled upstream test/reference HDF5 files are **not** demonstrated NERVA hot-core statepoints.
- ntr/nerva/build_model.py; model.py; geometry.py; materials.py; config.py; layout.py; periphery.py; fuel_sector_geometry.py — NERVA-derived fuel/tie lattice, regions, materials, reflector/drum and fuel-sector builders.
- ntr/nerva/tallies.py; calculate_volumes.py; neutron_energy_groups.py — tally/filter definitions, volume handling and energy-group support.
- ntr/nerva/run_openmc_transport.py; run_full_openmc_analysis.py; postprocess_statepoint.py; postprocess_instance_tallies.py; run_analysis.py — transport execution and extraction pipelines.
- ntr/nerva/mesh_refinement.py; mesh_refinement_statistics.py; openmc_replica_convergence.py; openmc_replica_ensemble.py; source_bank_convergence.py; source_bank_trajectory.py; temporal_convergence.py; statepoint_quality.py — convergence-related algorithms and diagnostics.
- ntr/nerva/core_statepoint_diagnostics.py; core_neutron_balance.py; core_neutron_economy_validation.py; real_core_keff_autocorrelation.py; real_core_keff_history_qualification.py; core_neutron_leakage_history.py — transport evidence and neutron balance diagnostics.

### 2.2 Nuclear data, source quality and uncertainty

Existing main modules include:
- check_nuclear_data.py; nuclear_data.py; audit_nuclear_data_temperatures.py; audit_bound_thermal_scattering.py; audit_kerma_coverage.py.
- core_material_temperature_qualification.py; executed_neutron_temperature_energy_grid.py; executed_neutron_reaction_xs_integrity.py; executed_material_kerma_qualification.py; real_core_material_transport_support.py.
- physical_scattering_pair_qualification.py; core_prompt_delayed_fission_validation.py; core_fission_yield_energy.py; evaluated_fission_neutron_product_yield.py; core_fission_emission_native_uncertainty.py.
- core_reaction_history.py; core_reaction_ensemble.py; core_reaction_estimator_agreement.py; core_reaction_partition.py; core_mt901_reaction_exposure.py; real_core_neutron_energy_closure.py.
- fixed_source_core_physics.py; fixed_source_spatial_response.py; fixed_source_fission_branching.py; neutron_economy_uncertainty.py; axial_covariance.py.
- source_energy_audit.py; source_phase_space_audit.py; source_direction_physics_audit.py; source_history_statistics.py; source_population_sensitivity.py; source_clone_moment_closure.py; source_resolution_audit.py; real_neutron_tail_native_uncertainty.py and real_neutron_tail_replica_statistics.py.

These are real diagnostics, AND actual NERVA-derived **cold surrogate** OpenMC execution evidence was retrieved from GitHub Actions run [39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), artifact ID 12492499186. It is genuine neutron-only HDF5 execution, but source convergence, criticality, hot material coverage and local heating are all UNQUALIFIED. Task TRAN-01 remains open for the selected, physically traceable/reference operating core with independent source convergence.

### 2.3 Main-branch thermal, fluids and performance code already present

- ntr/nerva/heat_partition.py; heating_data_qualification.py; postprocess_statepoint.py — heating partitions, tally-level data qualification and power accounting.
- fuel_conduction.py; fuel_sector_geometry.py; solve_element_channels.py; resolved_transient.py; thermal_flow_coupling.py — fuel conduction and multi-channel energy.
- thermal.py; solve_thermal.py; openmc_transient.py; resolved_openmc_feedback.py; openmc_cycle_coupling.py; thermal_cycle_feedback.py — transient and OpenMC thermal interfaces.
- implicit_hydrogen.py; nasa_hydrogen.py; hydrogen_properties.py; compressible_channel.py — hydrogen enthalpy/thermo/EOS-adjacent physics.
- cryogenic_tie_enthalpy.py; cryogenic_tie_hydraulic_coupling.py; pressure_driven_tie.py; inertial_tie.py; isochoric_tie.py; distributed_tie_euler.py; dynamic_header_tie.py; dual_header_tie.py; dynamic_parallel_header.py — existing cryogenic, compressible/tie and header fluid solvers.
- lh2_feed_cycle.py; hot_lh2_cycle.py; regenerative_cooling.py; turbine_bleed_cycle.py; gas_feed_network.py; engine_performance.py; nasa_cea_rocket.py; nerva_engine.py — downstream supporting reduced-order plant models.

**Do not rewrite these from scratch** before evaluating whether their physics can be connected, benchmarked and maintained.

### 2.4 Documents already available in the repository

At repo root:
- William J. Emrich Jr. - Principles of Nuclear Rocket Propulsion-Butterworth-Heinemann (2023).pdf.
- NTP_nerva.pdf.
- aaa neuronic code.pdf.
- nucke engine simul.pdf.
- NERVA-CORE-BUNDLE.zip (must be inventoried; the ZIP is not evidence its entire contents were faithfully implemented).

In ntr/nerva/:
- EMRICH_2023_CORE_PHYSICS_CROSSWALK.md — verified chapter-heading/index crosswalk; **not** proof that book equations/data tables were all extracted correctly.
- NASA_FIDELITY_MATRIX.md — separates design targets from test hardware and lists NASA validation gaps.
- NERVA_STATUS_HANDOFF.md — substantial prior handoff, but dated 2026-10-21, before recent open stages; its percentage is subjective.
- OPENMC_TRANSIENT_COUPLING.md; NERVA_ENGINE.md; REFERENCES.md; RN_S_0114_SOURCE_AUDIT.md; NERVA_BUNDLE_ADJOINT_DELAYED_AUDIT.md.
- Stage 6–20 specific FX2TH_STAGE*_AUDIT.md and FAMILY_CONTRACT.md on the Stage 20 feature branch.

**Historical source interpretation:** 1972 NERVA flight engine design, XE-Prime experimentally tested hardware, NRX and SNRE/NESS or later alternatives are **not** one interchangeable reactor design. Select one target and keep comparisons separate.

---

## 3. FX2-TH / Stage 1–20 inventory and exact GitHub integration state

### 3.1 Early branches / alternative approaches

| Stage or PR | Main work | Status needing attention |
|---|---|---|
| #111 Stage 1 | Basic delayed-neutron point kinetics and flux-shape numerical checks | Open independent branch based on main; compare with Stage 6/7 family system before merging |
| #112 Stage 2 | Multigroup 2-D diffusion, adjoint and conservation checks | Open independent branch based on main; inspect duplicated operators |
| #113 Stage 3 | Full space-time diffusion with local delayed precursors | Open branch based on #112 |
| #114 Stage 4 | Two-rate improved-quasistatic-style evolving 2-D flux and precursors | Open branch based on #113; not equivalent to fully reconstructed historical FX2-TH |
| #115 | Audit of NERVA-CORE-BUNDLE.zip without extraction | Open; audit-only |
| #116 | NERVA bundle axisymmetric diffusion, r-z boundary/annular source correction | Open independent branch based on main; reconcile with Stage 12 |
| #117 | Bundle adjoint/delayed audit | Open; base of the MAIN Stage 6–20 stacked branch |
| #109 | An alternative implicit radial/axial conjugate FV core thermal implementation | Open separate branch based on main; compare with Stage 19/20 before choosing one implementation |

**Action:** these are NOT automatically merged into the Stage 6–20 feature branch merely because they share scientific subject matter. Avoid duplicate core solvers and inconsistent source contracts.

### 3.2 Main Stage 6–20 chain — open PRs, not merged to main

| Stage | PR | Core source / intent | Evidence level as of 2026-10-24 | Work still dependent on real data |
|---|---|---|---|---|
| 6 | [#118](https://github.com/dubakoor13/NUKE/pull/118) | delayed_family_contract.py; delayed-family fission and source contracts | I/N synthetic | Evaluated delayed-neutron nuclear data |
| 7 | [#119](https://github.com/dubakoor13/NUKE/pull/119) | fx2th_family_spacetime.py; spatial multigroup neutron/precursor kinetics | I/N synthetic | Transport-derived coefficients and kinetics benchmarks |
| 8 | [#120](https://github.com/dubakoor13/NUKE/pull/120) | fx2th_thermal_energy.py; fx2th_thermal_feedback.py; conservative two-node heat | I/N synthetic | Normalized physical heating and real solid/H₂ data |
| 9 | [#121](https://github.com/dubakoor13/NUKE/pull/121) | fx2th_transport_heat_projection.py; normalized spatial power remapping | I/N synthetic | Real, material-resolved OpenMC heating/tallies and overlap |
| 10 | [#122](https://github.com/dubakoor13/NUKE/pull/122) | fx2th_openmc_statepoint_heating.py; StatePoint provenance, statistical/seed checks | I/N mock/test-double | Real NERVA statepoint evidence and source convergence |
| 11 | [#123](https://github.com/dubakoor13/NUKE/pull/123) | fx2th_cartesian_family_bridge.py; Cartesian diffusion/family interface | I/N synthetic | Transport-matched group coefficients and 3-D mapping |
| 12 | [#124](https://github.com/dubakoor13/NUKE/pull/124) | fx2th_rz_family_diffusion.py; conservative axisymmetric r-z multigroup FV | I/N synthetic | Real core geometry, XS and 2-D/3-D transport comparison |
| 13 | [#125](https://github.com/dubakoor13/NUKE/pull/125) | fx2th_adaptive_core_kinetics.py; adaptive first-order implicit kinetics | I/N synthetic | Real time constants/stiff transient measurements |
| 14 | [#126](https://github.com/dubakoor13/NUKE/pull/126) | fx2th_sdirk2_neutron_kinetics.py; second-order stiff adaptive integration | I/N synthetic | Cross-check physical kinetics and convergence |
| 15 | [#127](https://github.com/dubakoor13/NUKE/pull/127) | fx2th_full_family_fission_eigen.py; prompt + independent delayed spectra; adjoint | I/N synthetic | Real family emission spectra, beta-eff, generation-time evidence |
| 16 | [#128](https://github.com/dubakoor13/NUKE/pull/128) | fx2th_rz_adjoint_sensitivity.py; analytic d(k)/d(parameter) | I/N synthetic | Actual material derivatives (especially dXS/dT and density) |
| 17 | [#129](https://github.com/dubakoor13/NUKE/pull/129) | fx2th_rz_temperature_tables.py; cellwise T-tabulated multigroup coefficients | I/N synthetic | Qualified tabulated nuclear data on matching geometry |
| 18 | [#130](https://github.com/dubakoor13/NUKE/pull/130) | fx2th_monolithic_rz_core.py; small-system nonlinear neutron/precursor/thermal BE | I/N synthetic | Real full heat source, H₂ pressure/density and fuel feedback |
| 19 | [#131](https://github.com/dubakoor13/NUKE/pull/131) | fx2th_conjugate_fuel_transport.py; fx2th_resolved_rz_feedback.py | I/N synthetic | Verified actual hex-fuel geometry, channel flow and deposition |
| 20 | [#132](https://github.com/dubakoor13/NUKE/pull/132) | fx2th_enthalpy_conjugate_core.py; fx2th_enthalpy_rz_feedback.py | I/N synthetic; **56/56** Stage 20 dedicated tests passed and Stage 19 compatibility workflow passed | Real-gas H₂, pressure/mass/momentum, fuel materials and OpenMC heating |

**Stack topology:** #117 -> #118 -> #119 -> #120 -> #121 -> #122 -> #123 -> #124 -> #125 -> #126 -> #127 -> #128 -> #129 -> #130 -> #131 -> #132. The arrow means the next PR depends on the previous branch. Stage 1–4 and #109/#116 are **additional separate paths**, not part of this exact arrow chain.

### 3.3 Verified recent CI, and exactly what it means

At Stage 20 commit 3d9f4f29612a756571b655bec9051ea368ccc150:
- Dedicated Stage 20 workflow run [40107100124](https://github.com/dubakoor13/NUKE/actions/runs/40107100124): **success**; 17 Stage 19 conduction tests + 18 Stage 20 H₂ enthalpy tests + 11 Stage 19 spatial neutron-feedback regressions + 10 Stage 20 H₂/neutron-feedback tests = **56/56**.
- Stage 19 compatibility workflow run [40107100232](https://github.com/dubakoor13/NUKE/actions/runs/40107100232): **success**; previous Stage 18 + Stage 19 core/thermal tests.
- The broader NERVA model-check run was still in progress at the last observed check. **Do not report that as passed** without checking its final run conclusion.
- These tests use synthetic nuclear coefficient tables, synthetic geometry/material properties and explicitly provided synthetic deposited heating. The tests prove exercised **discrete equations, units/guards and conservation**, not a physical NERVA reactor state or measured core power.
- Source tree note: the Stage 20 Git tree did not contain NERVA statepoint HDF5 files. **CORRECTED 2026-10-25:** Real OpenMC HDF5 statepoints **are** preserved as an external GitHub Actions artifact from successful run [39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), artifact ID 12492499186. SHA256 of the original ZIP: `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`. Do not conflate absence from Git tree with no real executed simulation. That cold subcritical surrogate still FAILS physical hot NERVA qualification.

### 3.3A CORRECTED real four-seed cold OpenMC evidence (2026-10-25)

**Real transport is already executed, but HOT NERVA PHYSICS is not validated.** A successful GitHub Actions [run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638) generated four original binary OpenMC 0.15.3 neutron eigenvalue statepoints (seeds 13007, 23011, 33013, 43019; 120 batches, 30 inactive, 3000 particles/generation, 90 active realizations). The archived ZIP artifact **12492499186**, name `nerva-real-openmc-four-seed-core-evidence`, has SHA256 `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`, is 7.1 MiB compressed and GitHub reports expiration **2026-11-13**. Original HDF5 bytes were inspected against reported k and statepoint digests. The mean reported `k_eff=0.01169802857789661` (formal SEM `7.69260125465282e-05`) establishes a strongly subcritical, unqualified **COLD SURROGATE**: independent spatial source convergence **FALSE**, nuclear data only ~293.605913 K (1/4 active materials with explicit temperature), local neutron `heating-local=0` and full reactor power normalization **BLOCKED**. [Stage 21 evidence audit](ntr/nerva/NERVA_STAGE21_REAL_OPENMC_EVIDENCE_AUDIT.md) records the exact provenance and the distinction between authentic real execution and physical validation.

- [x] **EVIDENCE-01 [P0]** Locate actual archived real cold OpenMC four-seed neutron-only source files, download original ZIP and independently inspect statepoint HDF5 native estimators and archive SHA.
- [ ] **EVIDENCE-02 [P0]** Retain an independently controlled immutable copy with long-term URLs/digests BEFORE the source Actions artifact expires on 2026-11-13; temporary CI artifacts are not a permanent archive.
- [ ] **EVIDENCE-03 [P0]** Rebuild and qualify an actual source-stationary reference transport ensemble; the present spatial-source cloning / instability report blocks physical k validation.
- [ ] **EVIDENCE-04 [P0]** Supply verified evaluated hot fuel/reflector/coolant data, density/phase/s(a,b), and actual source physics; one-temperature cold library cannot qualify NERVA operation.
- [ ] **EVIDENCE-05 [P0]** Produce nonzero statistically resolved, disjoint genuine neutron+photon material heating with justified source/power normalization; reject the archived zero `heating-local` as a physical hot-core heating input.
- [ ] **EVIDENCE-06 [P0]** Select one historical NASA test configuration and independently benchmark actual transport results; do not tune the cold surrogate to k=1 and call it validation.

The Stage 21 read-only gate implements *binary HDF5 identity, cross-seed XML equivalence, ensemble k recomputation and explicit blocked-physics reporting*; it does NOT close EVIDENCE-02–06. **No real hot operating point has been qualified.**

### 3.4 Immediate merge/integration TODO

- [ ] **MERGE-01 [P0]** Snapshot branches, exact PR bases/heads and CI conclusions into a dependency manifest. Acceptance: every expected source and audit file is reachable from a proposed integration commit.
- [ ] **MERGE-02 [P0]** Resolve duplication: Stage 2 vs 11 vs 12 vs #116; Stage 8 vs #109 vs 18–20; Stage 4 IQS versus Stage 13/14 modern transient solvers. Acceptance: one primary implementation per discrete operator, alternatives retained only with explicit validation role.
- [ ] **MERGE-03 [P0]** Review upstream #117 and Stage 6–20 against main changes; preserve main's existing OpenMC real-data qualification modules and source provenance checks. Acceptance: no direct edits or cherry-picks silently discard other branches.
- [ ] **MERGE-04 [P0]** Merge with a single reviewed, ordered dependency chain or consolidate through a clearly scoped integration PR. Acceptance: working import paths, no stale source IDs, all dedicated workflows green at integration HEAD.
- [ ] **MERGE-05 [P0]** Run full NERVA regression + relevant upstream OpenMC tests at the final integrated revision. Acceptance: evidence recorded for the exact merged commit, not separate parent branches only.
- [ ] **MERGE-06 [P1]** Add canonical runnable example (SYNTHETIC vs REAL explicitly selectable, no automatic fallback) for Stage 20 integrated equations. Acceptance: failure is explicit when evaluated nuclear data/actual deposition are absent.
- [ ] **MERGE-07 [P1]** Refresh README.md / ntr/nerva/README.md / NERVA_STATUS_HANDOFF.md from observed evidence; link THIS TODO. Acceptance: no assertion that unmerged PR code exists on main.

---

## 4. P0 work queue — indispensable physics evidence and data

### 4.1 Lock the actual reactor target and reference traceability

- [ ] **REF-01 [P0] — Choose ONE primary historical test/configuration.** Select a named NRX/XE-Prime/other specific public NERVA test configuration or explicitly label it a NERVA-derived surrogate; identify exact hardware revision, operating point and intended observable set. **Need:** source documents with test date, hardware dimensions, fuel/reflector/tie structure, coolant state, control positions, power and uncertainties. **Acceptance:** versioned configuration identity and no mixing experimental data with 1972 design requirements.
- [ ] **REF-02 [P0] — Create a NASA primary-source evidence index.** For every geometry/material/measurement input record NASA NTRS citation, document title/revision, PDF SHA, **actual** PDF viewer page, figure/table/equation number, extracted value, units, reading uncertainty and extraction/peer-review status. **Acceptance:** every calibrated or validated quantity traceable; no inferred page numbers reported as inspected facts.
- [ ] **REF-03 [P0] — Digitize validation measurements.** Reactor k or critical control state where available; heater/fuel/channel temperature, axial power shape, pressure/flow, power, startup transient/control worth and other accessible measured channels. **Acceptance:** machine-readable measurement file with time/station metadata and uncertainty. Where data do not exist, flag **not available**; do not invent.
- [ ] **REF-04 [P0] — Inventory provided source assets.** Hash/inspect NTP_nerva.pdf, Emrich 2023, NERVA-CORE-BUNDLE.zip, the remaining root PDFs, existing docs and additional NASA reports; extract diagrams, governing-equation definitions and references without changing source. **Acceptance:** source-audit table distinguishing original code, extracted text, OCR uncertainty and independently reconstructed equations.
- [ ] **REF-05 [P0] — Define acceptance tolerances BEFORE tuning.** Establish independent calibration set vs hold-out validation set, observables, uncertainty bounds, mesh/timestep and statistical acceptance. **Acceptance:** frozen machine-readable benchmark manifest with approved pass/fail methodology; never fit a parameter to a datum and call that same datum independent validation.
- [ ] **REF-06 [P1] — Complete Emrich physics crosswalk at equation level.** Use checked-in Emrich 2023 edition, real page images and source equation inspection for diffusion, transport, kinetics, thermal-fluid, material properties, burnup and experiments. **Acceptance:** page-and-equation verified references; do not copy alternate-edition equation numbers.

### 4.2 Evaluated cross sections and thermal/neutronic material state

- [ ] **DATA-01 [P0] — Establish evaluated nuclear data manifest.** Enumerate all root-reachable isotopes and materials including fuel, moderator, hydrogen coolant, graphite, ZrC, ZrH, Be, absorber, structure and coatings as actually used. Resolve libraries, versions, temperatures, thermal scattering kernels and reaction support. **Acceptance:** file-by-file cryptographic hashes and audit of real neutron/photon data opened by OpenMC; no placeholder cross-section filenames.
- [ ] **DATA-02 [P0] — Material/temperature validity audit.** For every nuclear material region, capture actual temperature, density, nuclide density, thermochemical/phase identity, interpolation policy and available evaluated-temperature range. **Acceptance:** no missing temperature kernels, unsupported isotope pairings, or unannounced fallback/extrapolation; use existing core_material_temperature_qualification and executed_neutron_temperature_energy_grid.
- [ ] **DATA-03 [P0] — Build real multigroup coefficient source.** Independently compute/condense D, Sigma_a, Sigma_s(g_in,g_out), nuSigma_f, chi_prompt and chi_delayed with normalization conventions from executed transport nuclear data on the SAME geometry and material state as the Stage 12/17 solver. **Acceptance:** group boundaries and collapse weights saved; transport-reaction-balance comparison with uncertainty; no invented energy-group coefficients.
- [ ] **DATA-04 [P0] — Temperature-group table coverage.** Construct actual selected-group tables over fuel, moderator, reflector, absorber and (where applicable) coolant T ranges. **Acceptance:** Stage 17 accepts them without extrapolation, preserves evaluated manifest, and comparison against held-out evaluated temperatures establishes table-interpolation error.
- [ ] **DATA-05 [P0] — Density dependence distinct from temperature.** Nuclear response requires state dependent on both temperature and density/composition, not fuel temperature alone. **Acceptance:** separate thermal-expansion, H₂ density/phase/moderation and reflector/material density paths, with independent volume consistency and units.
- [ ] **DATA-06 [P1] — Fission prompt/delayed isotope evidence.** Verify actual family beta, lambda, spectra and emitted energy vs isotope/incident group where supported. **Acceptance:** independent Stage 15 fission matrix birth accounting and isotope/yield audits against source nuclear data; explicit approximation where data unavailable.
- [ ] **DATA-07 [P1] — Evaluate derivative validity.** Construct dXS/dT, dXS/drho, d(chi)/dT and associated uncertainty from qualified tables or evaluated processing, not guessed Doppler constants. **Acceptance:** Stage 16 adjoint d(k)/dT, d(k)/drho matches independently rerun transport/eigen solves over a small valid perturbation; knot nonsmoothness handled.
- [ ] **DATA-08 [P1] — Photon and fission-energy data.** Verify energy release and transportable heating conventions including neutron/photon energy and escaping energy, for the actual materials. **Acceptance:** explicit selected estimator and energy-production/escape ledger; no double count between prompt/delayed and transport heating.
- [ ] **DATA-09 [P1] — Nuclear-data uncertainty.** Obtain defensible reaction XS, scattering, fission and delayed-data uncertainty/covariance when available. **Acceptance:** state whether uncertainty is available vs unknown; propagate appropriate correlation rather than assume all errors independent.
- [ ] **DATA-10 [P1] — Data-interface reproducibility.** Persist physical table files in a versioned external artifact manifest; verify hashes of ACTUAL bytes rather than accept only a declared hash token. **Acceptance:** stale/edited data rejected, and table-to-OpenMC material links are independently checked.

### 4.3 Genuine neutron and photon transport on selected core

- [ ] **TRAN-01 [P0] — Qualify a real transport reference, building on an ALREADY EXECUTED cold surrogate.** Four actual cold NERVA-derived surrogate statepoints from 2026-10-23 are archived and their exact hashes verified (see Stage 21). The selected historical target, stationary fission source, physical criticality and hot data are NOT met. **Acceptance:** run a source-convergence-qualified, geometrically traceable reference with evaluated data, preserve genuine statepoints/seed evidence, and report the original cold surrogate separately as unqualified.
- [ ] **TRAN-02 [P0] — Independent source convergence (KNOWN BLOCKER in genuine cold surrogate).** The actual 2026-10-23 four-seed run reported source cloning and no spatial source stationarity despite formally stable k. Use separately initialized seeds/source banks, suitable inactive generations and source shape/entropy/moment convergence checks, not only k stabilization. **Acceptance:** prescribed independent evidence using existing source_bank_* / replica modules; report any persistent source bias.
- [ ] **TRAN-03 [P0] — Statistical k and tally convergence.** Retain active-batch histories, autocorrelation and appropriate effective sample size, independent ensembles and region-specific relative uncertainties for physically relevant tallies. **Acceptance:** reproducible uncertainty diagnostics, statistically stable interpretation and approved stopping.
- [ ] **TRAN-04 [P0] — Energy/reaction/neutron economy closure.** Audit absorption/capture/fission/scattering, leakage, emitted neutrons and fission source weights with explicitly chosen estimator definition. **Acceptance:** closed neutron population/economy within measurement/statistical uncertainty and physically consistent filter scopes.
- [ ] **TRAN-05 [P0] — Nuclear data and physics score checks.** Exercise actual thermal scattering, photon transport, heating/KERMA score coverage, reaction types and phase/energy domains; no silent missed nuclide. **Acceptance:** data-quality audit and tally score correctness at the specific statepoint.
- [ ] **TRAN-06 [P0] — Mesh/spatial convergence.** Refine geometry discretization and mesh/tally resolution, distinguish tally voxel error from actual material geometry error. **Acceptance:** observed convergence of k, axial/radial flux and heating with statistical intervals and documented numerical extrapolation when justified.
- [ ] **TRAN-07 [P1] — Independent hot material snapshots.** Repeat real OpenMC runs for at least a few defensibly specified, source-traceable fuel/moderator/coolant/reflector state combinations. **Acceptance:** independent seed/statistical convergence and uncertainty at every hot/cold state.
- [ ] **TRAN-08 [P1] — Control-drum / absorber physics.** Compare drum rotation, worth and flux deformation on real geometries with named reference inputs. **Acceptance:** transport-based worth with uncertainty and independent perturbation response; do not claim NERVA drum calibration from the demonstrator's arbitrary angle convention.
- [ ] **TRAN-09 [P1] — Transport leakage and angular inadequacy of diffusion.** Determine where Stage 12 diffusion has unacceptable errors (vacuum boundaries, streaming/gaps, reflector/drum faces). **Acceptance:** position- and group-resolved diffusion vs transport discrepancies, with explicit diffusion validity domain.
- [ ] **TRAN-10 [P1] — Save a qualified transport benchmark package.** Store input/outputs, exact settings, nuclear-library hashes, cross sections, confidence analysis, convergence plots, rate estimates and acceptance results in a stable artifact manifest. **Acceptance:** another machine/person can reproduce against the SAME software revision.

### 4.4 Physically valid neutron/photon power from transport

- [ ] **HEAT-01 [P0] — Obtain REAL heating tallies.** Generate material-separated neutron/photon heating (and any required local-kerma components) using a documented score/particle-transport convention on the matched core. **Acceptance:** actual statepoint & score/filter evidence, not generated test doubles.
- [ ] **HEAT-02 [P0] — Nuclear source-rate / MW normalization.** Establish exactly what the raw score means (typically energy per source particle) and how externally specified absolute source rate or independently validated thermal power scales it into watts. **Acceptance:** unit-checked eV/source × J/eV × source/s, with no silent fission-eigenflux-to-watts inference or circular calibration.
- [ ] **HEAT-03 [P0] — Disjoint material energy partitions.** Prove a scored deposition unit belongs exactly once to fuel, coolant, tie, reflector, absorber, structure or explicitly unresolved/escaped reservoir. **Acceptance:** total and spatial first-law closure within statistical uncertainty, no double-counted cell/material filters.
- [ ] **HEAT-04 [P0] — Mesh-to-thermal geometry projection.** Build a true overlap/incidence map from actual transport materials/cells/axial segments to representative fuel rings, coolant segments and tie units. **Acceptance:** weights nonnegative, sum rules correct, total W conserved, material identity and volume units traceable.
- [ ] **HEAT-05 [P0] — Heating uncertainty and correlations.** Quantify source variation, within-run tally noise, covariance and spatial correlations before thermal projection; use independent replica ensembles when appropriate. **Acceptance:** confidence bounds on each physically important fuel/channel/tie peak and whole-core power, without arbitrary independence assumptions.
- [ ] **HEAT-06 [P1] — Radial fuel-ring deposition within elements.** Current representative annuli are thermal surrogates. Determine actual nonuniform ring/azimuthal nuclear deposition and fuel coatings from transport or independently justified subcell reconstruction. **Acceptance:** mapped ringwise watts with source evidence and a quantified geometric/modeling error.
- [ ] **HEAT-07 [P1] — Photons, escapes and delayed-energy treatment.** Check neutron energy slowing, capture gammas, photon transport, escaping leakage, energy deposition in H₂ and unmodeled categories. **Acceptance:** internally consistent selected heating estimator/particle mode and no energy-generation/deposition accounting conflicts.
- [ ] **HEAT-08 [P1] — Connect Stage 9/10 genuine data to Stage 19/20.** Replace the synthetic callback with provenance-validated transport heating, preserving end-of-step energy ledger and uncertainty. **Acceptance:** no fallback to fabricated power; independent final field and aggregate balance checks.
- [ ] **HEAT-09 [P1] — Time-varying deposited power.** Determine physical power amplitude evolution/shape feedback separately from normalized transport shape, including delayed-neutron power contribution as appropriate. **Acceptance:** no arbitrary source normalization hidden in a transient or repeated physical advancement during iterations.
- [ ] **HEAT-10 [P1] — Peak vs integral quality.** Test whether calculated peak fuel and H₂ deposition are stable under mesh refinement/statistical uncertainty. **Acceptance:** transport-supported confidence intervals for maxima, not only global integrated W.

---

## 5. P1 work queue — core neutron equations, numerics and resolved feedback

### 5.1 Multigroup diffusion, forward/adjoint and kinetics

- [ ] **NEUT-01 [P1] — Verify actual transport-to-FV condensation.** Map the chosen 3-D OpenMC groupwise flux and reaction rates into Stage 12 (r-z) integrated loss and fission operators. **Acceptance:** independent groupwise reaction-rate and leakage source closure on matching cells; explicit treatment of cylindrical averaging and channels not axisymmetric.
- [ ] **NEUT-02 [P1] — Quantify the 2-D cylindrical approximation.** Compare 3-D heterogeneous core/reflector/tie/control geometry vs representative axisymmetrized model. **Acceptance:** bias in k, local flux and axial/radial power documented and bounded; no use in regions where approximation fails silently.
- [ ] **NEUT-03 [P1] — Real fast/thermal multigroup benchmark.** Use at least one physically documented group structure for the actual fuel/moderator, including reflector spectrum and thermal scattering. **Acceptance:** transport-derived group constants and error budget; synthetic two-group benchmarks retained only as mathematical unit tests.
- [ ] **NEUT-04 [P1] — Validate forward and adjoint eigenvectors on real states.** Compare Stage 15 k and adjoint importance-derived measures with suitable independent transport references or independent numerical methods. **Acceptance:** no unvalidated use of a diffusion adjoint as a transport adjoint in drum or channel sensitivity.
- [ ] **NEUT-05 [P1] — Physical prompt lifetime and delayed beta-eff.** Source and verify neutron generation-time definitions, precursor yields and family energies, including normalization. **Acceptance:** independent benchmark results and uncertainty for the chosen isotope/mixture and spectrum, not a synthetic normalization gauge.
- [ ] **NEUT-06 [P1] — Temperature/density worths.** Feed qualified dD/dT, dSigma/dT, spectral and material-density slopes into Stage 16. **Acceptance:** adjoint sensitivity and finite-perturbation transport agree over controlled perturbations, both sign and magnitude with uncertainty.
- [ ] **NEUT-07 [P1] — Real transient reference case.** Solve a bounded, physically motivated reactivity/source/thermal perturbation with full delayed families. **Acceptance:** time-step and mesh convergence and independent integration/reference; no unsafe extrapolation of a toy kinetics transient to a real start-up.
- [ ] **NEUT-08 [P1] — Reconcile transient algorithms.** Maintain first-order BE, adaptive BE and second-order SDIRK2 as clear alternatives; Stage 18/19/20 coupled thermal Picard path remains first-order. **Acceptance:** consistency of source units and time advancement, chosen production solver justified by stiffness/accuracy and positivity.
- [ ] **NEUT-09 [P1] — Nonlinear coupling error budget.** Vary time step, nuclear temperature iteration tolerance, fuel-ring mesh, coolant axial mesh, nuclear XS temperature grid and source tolerance independently. **Acceptance:** distinguish solver convergence from model/physical error; nonconvergence fails closed.
- [ ] **NEUT-10 [P1] — High-fidelity regionwise feedback.** Keep fuel, coolant H₂, tie moderator, reflector and absorber temperatures/densities separate rather than using one volume-weighted fuel temperature to represent everything. **Acceptance:** nuclear material state from actual region-resolved thermophysical properties and isotope composition.
- [ ] **NEUT-11 [P1] — Near-critical numerical safeguards.** Analyze sensitivity and conditioning of criticality/eigenmode under nearly degenerate roots and abrupt material transitions. **Acceptance:** forward/adjoint residual and spectral-gap checks, no false derivative acceptance.
- [ ] **NEUT-12 [P2] — Efficient integrated transient operators.** Profile dense Newton and full-state LU vs sparse Schur/iterative methods on nontrivial meshes. **Acceptance:** memory/scaling benchmark and deterministic validation prior to C++ migration.

### 5.2 Actual fuel solid geometry and heat transfer

- [ ] **FUEL-01 [P0–P1] — Reconstruct selected reference fuel element.** Exact coolant channel layout, graphite/fuel ligament, ZrC coating, hex flats, tie contact, filler and axial/support geometry with uncertainties. **Acceptance:** 3-D CAD/mesh or precise parameterized geometry whose surfaces/volumes match primary references.
- [ ] **FUEL-02 [P1] — Physics-consistent micro-to-macro mapping.** Replace unqualified representative annulus where needed with actual hex-channel sectors and mapped volume/area/conductance. **Acceptance:** matching solid masses, heat-capacity volumes, heat paths, nuclear tally volumes, channel counts and energy budgets.
- [ ] **FUEL-03 [P1] — Temperature-dependent thermal properties.** Provide k(T), cp(T), rho(T), coating properties, anisotropy if material demands, and uncertainty. **Acceptance:** source tables with phase/temperature validity and independent property/energy benchmarks.
- [ ] **FUEL-04 [P1] — Coating and contact resistances.** Model graphite/ZrC/structure/heat-transfer film layers with correct cylindrical/real-channel area and temperature dependence. **Acceptance:** independent analytical layered-cylinder or geometry-appropriate conduction solution, continuity of heat flux.
- [ ] **FUEL-05 [P1] — Nonuniform nuclear volumetric heat.** Couple actual material-resolved W/m3 or W/cm3 from OpenMC to ring/sector/axial thermal elements. **Acceptance:** exact units and conservation per material/region, uncertainty on peak fuel temperatures.
- [ ] **FUEL-06 [P1] — Axial/radial conduction and end conditions.** Replace adiabatic outer/axial default only when hardware demands conduction to tie, reflector, support, vessel or radiation boundary. **Acceptance:** complete boundary inventory and nonduplicated thermal fluxes.
- [ ] **FUEL-07 [P1] — Thermal expansion and nuclear geometry.** Update material densities/volumes consistently with temperature without creating nuclear inventory. **Acceptance:** conserved isotope atoms/solid mass and independent deformation/geometry consistency; no unqualified physical expansion coefficients.
- [ ] **FUEL-08 [P2] — Temperature gradients/stress margins.** Couple defensible material failure limits, thermal stresses and coating gradients only after verified materials and power. **Acceptance:** stress/strain model validated separately, limits tied to sources, not arbitrary safety numbers.
- [ ] **FUEL-09 [P1] — Contact with tie/support elements.** Heat/conductance and nuclear moderator heterogeneity from the specific support structure, not generic all-fuel channels. **Acceptance:** independent heat/energy accounting and neutron feedback with tied interface geometry.
- [ ] **FUEL-10 [P1] — Peak-temperature numerical convergence.** Refine fuel rings, axial bins, angular sectors and wall-resistance mesh; compare with independent conduction reference. **Acceptance:** mesh-sensitive maximum temperature error separately tracked from total power.

### 5.3 Hydrogen thermodynamics, mass/momentum and two-phase boundaries

- [ ] **H2-01 [P1] — Integrate state-consistent h(P,T), cp(P,T), rho(P,T).** Reuse existing CoolProp real-fluid backend where its property domain applies; keep NASA-7 gas-phase model for its documented domain. **Acceptance:** same EOS used in all storage/transport terms, validity/error handling and unit checks, independent property comparison.
- [ ] **H2-02 [P1] — Physical cryogenic inlet.** Stage 20 NASA-7 H₂ strictly begins at 200 K; it cannot handle a liquid-H₂ inlet. Bridge real cryogenic/two-phase supply through separately verified CoolProp/enthalpy path. **Acceptance:** valid phase region, no silent NASA-7 extrapolation or instantaneous artificial gas transition.
- [ ] **H2-03 [P1] — Mass conservation and density evolution.** Current Stage 19/20 uses fixed coolant mass per segment and prescribed flow; add coupled continuity and EOS without over-constraining pressure/mass. **Acceptance:** conservative storage/in/out mass ledger under varying pressure and flow.
- [ ] **H2-04 [P1] — Momentum/pressure loss.** Coupled mass/momentum/total-energy solver for actual channels, variable cross-section and wall friction with compressibility/acceleration. **Acceptance:** benchmarked channel pressure profile and propagation speeds; no detached prescribed mdot where impossible.
- [ ] **H2-05 [P1] — Actual H₂/H chemistry if temperatures warrant.** Determine whether frozen, equilibrium or finite-rate dissociation is physically applicable in hot core/nozzle/return passages. **Acceptance:** consistent enthalpy/species mass and reaction energy across spatial cells; compare to accepted chemistry reference in declared domain.
- [ ] **H2-06 [P1] — Real heat-transfer coefficient UA.** Use validated Reynolds/Prandtl, channel surface roughness, gas property, ZrC coating/conduction and conjugate wall models to calculate UA, with uncertainty. **Acceptance:** no tuned unknown UA represented as empirical NERVA truth; independent correlations/hardware evidence.
- [ ] **H2-07 [P1] — Local transport regimes.** Check laminar, transitional, turbulent, choked, flash/two-phase or flow reversal regimes as physically relevant, and enforce correlation validity. **Acceptance:** explicit bounds/mode switching or rejected states, not an unqualified global correlation.
- [ ] **H2-08 [P1] — Branch and tie flow distribution.** Match fuel channels vs tie/support loops and common plenums, including counterflow where documented. **Acceptance:** conserved total mass/enthalpy, physically consistent pressure matching and hydraulic network junction conditions.
- [ ] **H2-09 [P1] — Dynamic pressure-reactivity coupling.** Propagate H₂ density/moderation changes into evaluated neutron tables at the current material state, separate from solid fuel temperature. **Acceptance:** independently checked density worth and correct fluid/nuclear spatial region identity.
- [ ] **H2-10 [P1] — Thermal transient convergence/robustness.** Compare Godunov/upwind + implicit energy against analytic Riemann/thermal transport/steady channel references where valid, with mesh and temporal refinement. **Acceptance:** no hidden artificial heat generation or unstable pressure/enthalpy interpolation.
- [ ] **H2-11 [P2] — Real-fluid boundaries of applicable models.** Explicitly gate CoolProp, NASA-7 frozen gas and equilibrium H₂/H modes by pressure/temperature/phase; prevent mixed inconsistent enthalpy reference states. **Acceptance:** validated mode-switch tests and continuity ledger.

---

## 6. Physical coupling and whole-core closure — the central unfinished integration

These are the most important **cross-module tasks**, not new isolated solvers.

- [ ] **COUPLE-01 [P0] — One canonical spatial geometry/mapping manifest.** Cross-reference every OpenMC material/cell instance, Stage 12 r-z diffusion cell, Stage 19/20 equivalent fuel-ring/channel, tie element and actual physical volume. **Acceptance:** material/region IDs and volumes match; explicit rules for non-axisymmetric control/reflector sectors; no unmatched physical energy.
- [ ] **COUPLE-02 [P0] — One nuclear heat source-of-truth.** Real OpenMC statepoint and its normalization/uncertainty feed Stage 20 as tagged watts, not synthetic callbacks. **Acceptance:** no fake heating enabled in REAL mode, full nuclear energy accounting and source provenance preserved to output.
- [ ] **COUPLE-03 [P0] — One actual temperature-/density-dependent multigroup source-of-truth.** Stage 17 reads data derived from exactly the materials/energy grid in the executed OpenMC case; Stage 20 updates correct spatial material state. **Acceptance:** source/material/temperature/density fingerprints match, or step is rejected.
- [ ] **COUPLE-04 [P1] — Full nonlinear thermal-neutronic feedback with real-data gates.** On each converged physical timestep, update flux, delayed families, normalized spatial heating, fuel/tie/hydrogen temperatures and relevant densities. **Acceptance:** no repeated physical step within iteration, converged neutronic + thermal + mass/energy residuals, never silent coefficient clipping.
- [ ] **COUPLE-05 [P1] — Compare Picard vs monolithic.** Stage 18 has a small dense Newton benchmark and Stage 19/20 a resolved Picard flow. **Acceptance:** independently match results on shared supported meshes and physics; identify failure/stability envelope and scalable solver path.
- [ ] **COUPLE-06 [P1] — Couple enthalpy to momentum/pressure.** Stage 20 currently retains externally fixed coolant cell masses and mdot. **Acceptance:** pressure/density/enthalpy and advective flux from ONE consistent conservative fluid step, including inlet/outlet boundaries.
- [ ] **COUPLE-07 [P1] — Verify total source/neutron/energy conservation in one run.** Source-rate, fission neutron birth, leakage, neutron/precursor populations, deposition and coolant energy must each reconcile under physically consistent units. **Acceptance:** independent end-to-end conservation ledgers, not just individual module checks.
- [ ] **COUPLE-08 [P1] — Reactor power versus coolant enthalpy.** Do not infer power directly from normalized eigenflux. **Acceptance:** if power is prescribed, label it external; if predicting absolute power, show physical amplitude equations and source/energy calibration with initial conditions and feedback.
- [ ] **COUPLE-09 [P1] — Error/uncertainty budget across coupled chain.** Partition nuclear statistics, XS/energy group approximations, power projection, fuel material, H₂ EOS, geometry, iteration and mesh errors. **Acceptance:** defensible uncertainty envelope on k, peak temperature, coolant outlet state and power.
- [ ] **COUPLE-10 [P1] — State transitions beyond one timestep.** Run controlled transient histories with step rejection/recovery, state snapshots and exact chronological provenance. **Acceptance:** no repeated timestep advancement in Picard, no stale material or source, stable conservation across many steps, reproducible restarts.
- [ ] **COUPLE-11 [P1] — Control-state and boundary coupling.** Where historical hardware permits, integrate control drum state, source and coolant boundary responses rather than arbitrary prescribed scalar reactivity. **Acceptance:** fieldwise transport-supported response and independent benchmark.
- [ ] **COUPLE-12 [P2] — Performance/engineering output only AFTER core qualification.** Feed independently verified coolant outlet enthalpy/pressure/mass flow to the existing nozzle/cycle solver. **Acceptance:** force/nozzle predictions use thermal and mass states that actually close; no unrelated tuning to target thrust/Isp.

---

## 7. Verification, validation and reproducibility task matrix

### 7.1 Numerical tests we already have (do not delete)

| Test family | Existing verification concept | What it DOES prove | What it does NOT prove |
|---|---|---|---|
| Stage 7/12 | Analytical cylindrical Bessel/buckling, annular volume, spatial neutron balance | Correct exercised FV geometry and multigroup algebra | Actual cross sections/core physical geometry |
| Stage 7/13/14 | Independent full neutron/precursor matrix exponential, adaptive reduction, SDIRK2 convergence | Exercised transient time integrators and stiff mathematical reference | Observed NERVA reactor kinetics |
| Stage 15 | Dense generalized fission eigen/adjoint, independent delayed-family emission | Correct full prompt/delayed operator and adjoint pairing | Evaluated isotope delayed-group data |
| Stage 16 | Finite-difference k/ρ sensitivities against independent dense eigensolves | Correct first-order adjoint calculus for synthetic coefficients | Physical Doppler or density reactivity coefficient |
| Stage 17 | Tabular interpolation, analytic temperature slopes, full-family source budget | Numerically safe supplied coefficient interpolation | Physical correctness of supplied coefficients |
| Stage 18 | Monolithic BE/Picard agreement and thermal first law | Correct exercised coupled nonlinear equations | Hot reactor feedback fidelity |
| Stage 19 | Annulus log conduction, harmonic axial faces, upwind H₂ enthalpy for constant cp | Resolved FV energy conservation | As-built fuel channels or variable real-fluid physics |
| Stage 20 | NASA-7 h(T), analytic/Brent inversion, nonlinear Jacobian, Stage 19 reduction, coupled Picard | Nonlinear frozen molecular-H₂ enthalpy and full-state discrete consistency | Actual pressure/density evolution or hot NERVA benchmark |
| Stage 9/10 | Tally projection and StatePoint test-double/seed guards | Conservativity/strict data contract and statistics algorithms | Genuine NERVA executable OpenMC evidence |

### 7.2 Mandatory future test suite classes

- [ ] **TEST-01 [P0] — True independent OpenMC benchmark.** At least one real matching statepoint and independent seed ensemble with supported evaluated data. Output artifacts explicitly labeled REAL TRANSPORT, not synthetic.
- [ ] **TEST-02 [P0] — Physical geometry/source/material validation.** Every accepted benchmark configured from locked historical input values and source pages, without silently tuned parameters.
- [ ] **TEST-03 [P0] — Neutron + photon heating estimator closure.** Real tally uncertainty, source-rate, direct H₂ heating and material split conserved against independent energy release/escape accounting.
- [ ] **TEST-04 [P1] — Multigroup transport vs diffusion.** k, angular leakage, group flux, reflector/moderator spectra and spatial power compared with uncertainty and validated approximation limits.
- [ ] **TEST-05 [P1] — Temperature/density sweep.** Actual coefficient tables reproduced by source transport and independent held-out statepoint calculation; temperature/spectrum and density sensitivities.
- [ ] **TEST-06 [P1] — Independent core thermal benchmark.** Known geometry/material or public experiments with heat transfer and H₂ flow; compare peak fuel/coat temperature, outlet h and pressure loss.
- [ ] **TEST-07 [P1] — Multi-step transient benchmark.** Compare accepted thermal/kinetic time integration against independent solution; includes source and material feedback, mass/energy storage and reactivity.
- [ ] **TEST-08 [P1] — Manufactured-solution FV tests.** Nonuniform r/z/ring/axial mesh, variable coefficients, real boundary sources, positive/negative fluxes and physically consistent conservative errors.
- [ ] **TEST-09 [P1] — Parameter perturbation regression.** Confirm prompt/delayed neutron production, spectra normalization, absorber worth and beta_eff across real transported states; quantify numerical sensitivity uncertainty.
- [ ] **TEST-10 [P1] — Stiff/positivity/nonconvergence.** Guard abrupt XS table boundaries, too-large timesteps, solver stalls and out-of-range H₂ temperatures; explicit failure without invented fallback.
- [ ] **TEST-11 [P1] — Material conservation under thermal expansion and EOS updates.** No creation of isotope atoms, H₂ mass, fission energy, or fuel volume by interpolation.
- [ ] **TEST-12 [P1] — Statistical reproducibility and seed separation.** Publish source bank, run settings, independent-seed counts, k/tally uncertainty, and comparison scripts.
- [ ] **TEST-13 [P1] — Restart/checkpoint determinism.** A multi-step run from checkpoint reproduces accepted state and all conservation ledgers within documented tolerance.
- [ ] **TEST-14 [P1] — Cross-branch merged regression.** Every original and updated test run at ONE merged Git SHA; no passing independent branches with broken combined imports.
- [ ] **TEST-15 [P2] — Runtime/performance scaling.** Multiple mesh sizes and solver tolerances on supported Windows/Linux environments; profile memory before optimizing or porting to C++.
- [ ] **TEST-16 [P1] — Historical hold-out cases.** Compare against multiple unreused NASA records, report per-observable discrepancy and uncertainty, do not retroactively redefine pass targets.
- [ ] **TEST-17 [P1] — Uncertainty propagation.** Transport statistical, nuclear-data, material and thermofluid parameter uncertainties propagated consistently; identify dominant contributors.
- [ ] **TEST-18 [P1] — Source provenance assertions.** Every output has provenance scope (synthetic/executed/physically benchmarked) and is rejected as REAL if any mandatory upstream artifact is synthetic or missing.

### 7.3 Reproducible executable validation commands already on Stage 20

Execute from NUKE repo root with its appropriate Python + NumPy/SciPy dependencies:

    python -m ntr.nerva.validate_fx2th_rz_family_diffusion
    python -m ntr.nerva.validate_fx2th_full_family_fission_eigen
    python -m ntr.nerva.validate_fx2th_rz_adjoint_sensitivity
    python -m ntr.nerva.validate_fx2th_rz_temperature_tables
    python -m ntr.nerva.validate_fx2th_monolithic_rz_core
    python -m ntr.nerva.validate_fx2th_conjugate_fuel_transport
    python -m ntr.nerva.validate_fx2th_resolved_rz_feedback
    python -m ntr.nerva.validate_fx2th_enthalpy_conjugate_core
    python -m ntr.nerva.validate_fx2th_enthalpy_rz_feedback

The dedicated run logs for #132 at the Stage 20 snapshot are evidence of passing synthetic numerical tests. The above list is a **test execution plan**, not a claim that all nine commands were separately rerun for THIS document.

Genuine physical runs must additionally use the repository's existing run_openmc_transport / run_real_core_replicas / run_full_openmc_analysis and qualification pipeline with an ACTUAL appropriate evaluated data library. Never silently point to upstream OpenMC test statepoints or generated fake NPZs.

---

## 8. Inputs, references and external artifacts NEEDED — exactly what to gather

### 8.1 Historical documents and digitized primary data

- [ ] Named baseline reactor/test configuration with revision; exact cross-section/assembly drawings including scale/datum.
- [ ] Fuel composition and loading, material densities, fuel-element/channel count, tie/support arrangement, coating thicknesses and material identity for selected hardware.
- [ ] Reflector, control drum/absorber and pressure-vessel geometry/material construction with control-angle conventions.
- [ ] Test instrumentation positions, run power, H₂ flow, inlet/outlet pressure/temperature, peak channel/fuel measurements and uncertainty, where public.
- [ ] Original NASA axial/radial power maps, criticality/rod-worth information or reactor kinetics traces, with source revision and context.
- [ ] Temperature-dependent solid property curves and allowed material domains for fuel graphite/coatings/tie/reflector, with source provenance.
- [ ] Distinguish references that support **design targets**, **experimental measurements**, **surrogate assumptions**, and **modern physics methods**.
- [ ] Source-doc extraction audit for every supplied PDF; unknown/missing portions explicitly marked as unavailable.

### 8.2 Nuclear/transport data and runnable artifacts

- [ ] Evaluated neutron data (all nuclides actually present) and proper temperature grids / S(a,b) thermal scattering; photon data where needed.
- [ ] Versioned OpenMC runtime/container/compiler environment and actual executable provenance.
- [ ] Actual selected NERVA geometry/material/settings/tally inputs and a consistent physical material/temperature/density manifest.
- [ ] Multiple real independent-seed OpenMC statepoints with fission source convergence diagnostics, active-batch histories and full tally covariance/uncertainty limitations.
- [ ] Energy-group definition, real group-collapse methodology and resulting 2-D/3-D nuclear coefficient tables.
- [ ] Absolute source-rate/power normalization justification and scored fuel/H₂/tie/reflector/structural neutron+photon deposition.
- [ ] Tagged spatial mesh overlap/rebin and power uncertainty files, not only one global total-power number.
- [ ] Reference transport comparison independent of the new diffusion code.

### 8.3 Thermal/hydraulic property and geometric artifacts

- [ ] True fuel/tie/channel thermal meshing and material sector volume map to OpenMC.
- [ ] Temperature-dependent solid k, cp, density/expansion, coating/contact conductance.
- [ ] Validated wall transfer correlations and documented channel hydraulic diameters/surface condition and coolant thermal region.
- [ ] Real-fluid hydrogen properties from an independently traceable EOS/data source across actual inlet to outlet pressure-temperature/phase range.
- [ ] Dynamic channel mass, momentum and total-energy boundary network, including tie/counterflow and headers where relevant.
- [ ] Pressure drop, mass flow distribution and outlet enthalpy constraints at the same measured reactor operating point.
- [ ] Independently checked nonlinear solver residual, first-law and mass-balance reports.

### 8.4 Missing inputs that should NEVER be silently fabricated

- Historical reactor fission isotope inventory, delayed-family yields, control-drum worth or absolute neutron source if not actually documented.
- Absolute thermal MW derived solely from an eigenvector scaled to arbitrary normalization.
- Hot NERVA fuel XS interpolation beyond the evaluated nuclear library.
- Pressure/density/momentum response inferred solely from the Stage 20 NASA-7 frozen h(T) model.
- Actual hexagonal-channel geometry inferred by treating a representative equivalent annulus as an as-built channel.
- Test/qualification evidence inferred from a successful synthetic GitHub Actions run.

---

## 9. Ordered execution plan — concrete next deliverables

This sequence prioritizes **physical input evidence over accumulating more abstract solvers**. Every milestone should land in an individually inspectable PR and include a source/equation/test audit.

### Next 1 — consolidate and lock the requirements

- [ ] Resolve MERGE-01–MERGE-05; record what is already merged vs pending, and choose one canonical Stage 6–20 integration path.
- [ ] Choose REF-01 and complete the minimal REF-02 historical-source matrix.
- [ ] Publish a frozen reference-config manifest, component geometry assumptions and physical-data missing list.

**Exit gate:** no one can mistake a synthetic surrogate or branch-only feature for a published historical NERVA reactor model.

### Next 2 — real transport evidence BEFORE further hot-core claim

- [ ] DATA-01/02; verify library versions/isotope/material coverage.
- [ ] TRAN-01/02/03; run independent seeds on the locked actual/core-surrogate geometry with real OpenMC and retain artifacts.
- [ ] TRAN-04/05/06; quantify neutron balance/source convergence, mesh error, statistical uncertainty, heating score validity.

**Exit gate:** archived actual statepoints and reproducible transport input; transport observable means + confidence diagnostics recorded.

### Next 3 — real energy deposition and absolute source mapping

- [ ] HEAT-01/02/03; qualify scored neutron/photon energy and physical normalization.
- [ ] HEAT-04/05/06; map real tallies to actual fuel sectors/channel/tie and quantify bin uncertainty.
- [ ] COUPLE-01/02; replace synthetic W power path with fail-closed, provenance-preserving real-data path.

**Exit gate:** accepted material-disjoint deposited W, no double count, and source-to-thermal conservation with documented uncertainty.

### Next 4 — evaluated state-dependent multigroup neutronics

- [ ] DATA-03/04/05/06; source real XS(T,rho,composition) and spectra.
- [ ] NEUT-01/02/03/04; compare k, group flux/power and neutron balance with real transport.
- [ ] DATA-07; verify adjoint and finite-difference temperature/density coefficients.

**Exit gate:** Stage 17 coefficients are physically traceable to a real matching transport state, and diffusion's approximation error is quantified.

### Next 5 — actual core thermal and H₂ physics

- [ ] FUEL-01/02/03/04/05; replace thermal-equivalent annulus assumptions where physically material.
- [ ] H2-01/02/03/04/06; join validated h(P,T), density, mass/momentum and heat-transfer laws, including cryogenic boundary when applicable.
- [ ] COUPLE-03/04/05/06/07; close the actual resolved hot-core temperature/density-neutronics/heating state without fabricated material inputs.

**Exit gate:** all connected mass, neutron population and energy balances close; physical data sources and convergence are independent and auditable.

### Next 6 — independent historical validation and uncertainty

- [ ] REF-03/05 and TRAN-07/08/09; reproduce multiple untuned physical reference states.
- [ ] TEST-01–TEST-18 as relevant; report quantitative discrepancies vs pre-agreed tolerances.
- [ ] Separate any discrepancy due to nuclear data, geometry, heat-transfer closure, physical instrumentation or numerical error.

**Exit gate:** independent measurements, not merely a synthetic test suite, support the claimed NERVA configuration and domain.

### Later — whole engine and C++

- [ ] Only after core is demonstrably credible, connect credible core outlet h/P/mdot to verified feed, pumps, turbine, nozzle and Isp code.
- [ ] Profile and port performance-critical validated Python mathematical kernels to modern C++ as justified. Preserve physics contracts and reference results; do not treat translation as validation.
- [ ] Add more operating modes (other propellants, LANTR or other nuclear thermal concepts) as **separately validated configurations**, not as hidden substitutes for NERVA.

---

## 10. Strict definition of done — acceptance gates

**Do not report the NERVA reactor-core project as physically complete until all mandatory gates are satisfied.**

- [ ] **GATE A — Config identity.** One historically traceable target (or explicitly classified demonstrator), source-referenced geometry/material/control/measurement definitions, with independent validation hold-outs.
- [ ] **GATE B — Real nuclear data.** Actual evaluated nuclear and photon data manifest, materials and temperature/phase coverage, fission-yield/spectrum provenance and no unsupported XS extrapolation.
- [ ] **GATE C — True transport.** Reproducible OpenMC neutron/photon case on matched geometry, multi-seed source convergence, statistics, neutron balance and k/power spatial diagnostics.
- [ ] **GATE D — Multigroup validation.** Real transport-derived coefficients, source and flux/adjoint eigenmode comparisons, meaningful k/beta_eff/sensitivity error analysis.
- [ ] **GATE E — True heating.** Qualified scored energy deposition, source-rate normalization, disjoint material heat partitions, projection to thermal cells, statistical confidence and full first-law closure.
- [ ] **GATE F — Real thermal/hydraulics.** Qualified fuel geometry/material thermal properties, H₂ h(P,T)/density and mass/momentum, wall/film physics and appropriate cryogenic/hot regimes.
- [ ] **GATE G — Coupled transient numerics.** Mesh and time convergence of actual neutron–precursor–temperature–hydrogen feedback; no stale XS or synthetic energy, no repeated physical time advancement; mass/neutron/energy conservation.
- [ ] **GATE H — Independent historical comparison.** Multiple hold-out NASA test observables for the SAME article, with uncertainty and documented residuals under predeclared acceptance bounds.
- [ ] **GATE I — Single integrated revision.** Code integrated into main or a stable release branch, all relevant CI and independent real-data benchmarks passed on its exact commit, artifacts retained.
- [ ] **GATE J — Reproducibility / scientific claims.** Input/data/runtime/model hashes, source report, numeric outputs, quality gates and explicit valid operating domain. Anything not supported is flagged *unvalidated*.

### Completion must be split into different reportable percentages

If a percentage is ever requested, first define weighted criteria and report **separately**:
1. source-code implementation completeness;
2. independent numerical verification completeness;
3. real transport-data integration completeness;
4. external historical/physical validation completeness;
5. merged production/release readiness.

One overall “% done” without agreed weighting and the above evidence creates false confidence.

---

## 11. Engineering pitfalls / hidden blockers that MUST be monitored

| Risk | Why it matters | Prevention / evidence gate |
|---|---|---|
| “Green CI = real reactor validated” | Synthetic inputs can satisfy perfect mathematical energy closure while physics is wrong | Tag synthetic vs real at every output and require real transport artifact |
| Mixed NASA historical designs | 1972 design targets are not XE-Prime measured data | Lock reference variant, measurement/test source and held-out cases |
| Fake hot nuclear XS | A linearly interpolated synthetic table is not Doppler physics | Evaluated XS source, temperature/density coverage and transport comparison |
| Absolute power from arbitrary flux | Eigenflux scale has no standalone watts meaning | Trace physical source rate or independently specified external power |
| Volume normalization and mapping | Stage 12 macro r-z cells are NOT Stage 19 micro-annular fuel channels | Geometric overlap and explicit channel-count, mass/volume audit |
| Heating double count | Fuel, direct H₂, tie, photon and other energy can overlap | Disjoint score/material filters and independent energy balance |
| Incorrect neutron vs photon heating | A score may omit deposited photon energy or include escaping terms | Document score mode/physics support and source energy ledger |
| Source convergence ignored | k stability alone may conceal spatial source bias | Independent source-bank moments, active-batch and seed diagnostics |
| Temperature averages hide bad subregions | One homogenized T can lie inside table while rings exceed range | Stage 19 safeguard: check every fuel ring before XS sample |
| Constant H₂ cp silently applied | Thermodynamic outlet temperatures and energy transport wrong | Stage 20 h(T), independent variable-cp tests; document domain |
| NASA-7 for cryogenic LH₂ | NASA-7 frozen H₂ h(T) is gas-phase only 200–3500 K | Real-fluid property/model dispatch, fail-closed cryogenic range |
| Frozen molecular H₂ above dissociation relevance | Reaction/chemical energy can matter at high T | Determine equilibrium/frozen chemical mode from supported EOS/reference |
| Pressure ignored in coolant feedback | Mass, density, moderation and flow may evolve substantially | Coupled mass/momentum/EOS, compare pressure-flow observables |
| Wrong transient time order | Solving neutrons multiple times from newly advanced states inside Picard creates artificial time | Always advance from identical old physical states for trial iterations |
| Overlarge implicit step | Stiff stable method can still give inaccurate transient or nonphysical population | Stage 13/14 adaptive residual and positivity; time-step study |
| Diffusion near leakage/drums | Simple 2-D diffusion may miss angular streaming/boundary behavior | Explicit OpenMC comparisons and domain restrictions |
| Inconsistent eigen/adjoint normalization | Corrupts beta_eff, lifetime and sensitivities | Verified Stage 15 normalization and Stage 16 gauge invariance |
| Unverified physical derivatives | Correct analytic d(k)/dθ with wrong dXS/dT gives wrong feedback | Qualified DATA-07, dense perturbation and transport comparison |
| Parallel open PR chains | Duplicate code can be “complete” individually and broken in combined main | Ordered merge and integrated CI at single SHA |
| Legacy handoff percent assumed factual | Early subjective estimate predates Stage 6–20 | Track task ID evidence, no objective overall % without weights |
| Missing actual statepoint artifacts | Read-only checks cannot reproduce a simulated physical result without data | Artifact registry with OpenMC version, library hashes, seed/statepoint IDs |
| Upstream OpenMC test data mistaken for NERVA | openmc/tests HDF5 is regression data for a different geometry | Mandatory NERVA model/source geometry fingerprint in evidence gate |

---

## 12. Suggested output files / directory structure for future evidence

These are **proposed future artifacts**, not a claim that these paths exist:

    ntr/nerva/evidence/
      README.md
      reference_configuration.json
      nasa_primary_source_index.csv
      source_pdf_sha256.json
      historical_benchmarks.csv
      numerical_acceptance_policy.json
      nuclear_data_manifest.json
      executed_material_inventory.json
      transport_case_manifest.json
      transport_ensemble_statistics.json
      statepoint_artifact_index.json
      group_collapse_manifest.json
      multigroup_nuclear_tables_manifest.json
      transport_vs_diffusion_comparison.csv
      heating_units_and_partition_audit.json
      mesh_to_channel_mapping_manifest.json
      fuel_material_property_manifest.json
      hydrogen_thermo_and_eos_manifest.json
      coupled_transient_conservation.json
      independent_validation_comparison.csv
      total_uncertainty_budget.json
      main_branch_integrated_regression.json

Evidence should refer to immutable large binary transport/tally artifacts using location + SHA hashes. Avoid bloating GitHub with duplicate statepoint files; do not omit them from the reproducibility manifest simply because stored externally.

---

## 13. Standard iteration / handoff log template

Append a dated record to this document or a linked work log after each implemented stage:

| Field | Required value |
|---|---|
| Work item IDs | Exact IDs, e.g., HEAT-02 and COUPLE-02 |
| Baseline | Old commit/branch and scientific assumptions |
| Files changed | Paths and newly implemented governing equations |
| Input evidence | Physical source IDs; evaluated nuclear-data SHA; actual statepoint/run IDs; explicitly synthetic if synthetic |
| Independent reference | Analytic derivation, separate numerical library or different established code |
| Numerical tests | Commands, counts, pass/fail, mesh/time/source convergence |
| Physical comparison | Named experimental cases, observables, uncertainties, observed errors or “NOT YET PERFORMED” |
| Conservation | Neutron, mass, enthalpy, deposited/escaped energy residual definitions and actual observed values |
| Limitations | Temperature/pressure/phase range, geometry surrogate and unresolved source data |
| Code integration | PR URL, base/head branch, merge/review and relevant GitHub Actions conclusion |
| Updated TODO | Exact completed/pending task checkboxes and new dependencies |

### Last documented iteration: Stage 21 (2026-10-25)

- **Implementation:** molecular H₂ NASA-7 variable enthalpy, full fuel/conjugate thermal solve, opt-in connection to spatial temperature-dependent neutron and delayed-family feedback.
- **Evidence:** synthetic numerical suites at commit 3d9f4f29612a756571b655bec9051ea368ccc150; dedicated run 40107100124 passed 56/56; Stage 19 compatibility workflow run 40107100232 passed.
- **Unresolved:** actual NASA hardware-matched geometry and materials, evaluated XS/temperature/density tables, genuine converged OpenMC neutron/photon heating, absolute power normalization and full H₂ pressure/mass/momentum closure.
- **Integration:** PR #132 OPEN; stacked on #131 etc.; not main. No claim of physically validated NERVA core.

---


### Follow-up physical evidence audit (2026-10-25)

- **Real OpenMC execution identified and independently inspected:** four seed-specific HDF5 binaries in GitHub Actions artifact 12492499186, run 39933618638, original SHA `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`.
- **Real physics blocked:** source stationarity not established, `k_eff≈0.011698` far subcritical, hot nuclear-data coverage false, local neutron `heating-local=0`, prescribed power normalization unjustified and historical fidelity unvalidated.
- **Stage 21 code:** `real_transport_evidence_gate.py`, its real-artifact numerical tests, and CI download of original binary evidence. This is **T-level executed cold surrogate evidence** only, not P/H-level nuclear/thermal qualification.
- **Immediate action:** EVIDENCE-02 before 2026-11-13, followed by EVIDENCE-03/04/05 and real-history source checks; Stage 6–20 stack remains unmerged.

## 14. Most important next action

**Do not start with another abstract neutron integrator or additional synthetic dashboard.**

1. Lock a single historical reference configuration and create the primary-source/input manifest (**REF-01, REF-02**).
2. Audit evaluated nuclear-data inventory and run/retain a real reproducible OpenMC transport ensemble for that reference/surrogate (**DATA-01/02, TRAN-01/02/03**).
3. Extract real material-resolved neutron/photon heating with independently verified source-rate normalization and conservative mapping to fuel/H₂ thermal cells (**HEAT-01 through HEAT-05, COUPLE-01/02**).
4. Only then promote Stage 17–20 from *synthetic numerics* toward *quantitatively physical hot-core prediction*, coupling real temperature/density/material and pressure/enthalpy feedback while preserving mass/neutron/energy balances.

**Success means independent physical evidence for a declared NERVA core—not merely more code, more green synthetic tests, or a thrust/Isp number.**
