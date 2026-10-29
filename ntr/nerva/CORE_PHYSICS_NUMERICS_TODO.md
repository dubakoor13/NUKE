# NUKE / NERVA — Reactor-core physics and numerics: detailed remaining TODO

**Scope:** the user's "first one": the remaining **reactor-core physics and numerical implementation**, not full engine-cycle completion.  
**Assessment date:** 2026-10-29.  
**Code basis:** Stage57 branch `feature/nerva-stage57-transient-hydrogen-conservation-20261029`; upstream Stages 51–57 are stacked and may **not be merged into `main`**.  
**Estimated unfinished implementation:** **30–40% of the specifically scoped core physics/numerics work**, a qualitative planning estimate, NOT a measured percentage.  
**Separate physical-validation debt:** much larger; green synthetic unit tests cannot establish a NASA-qualified hot NERVA core.

## Status and ground rules

Use `[x]` only for code actually implemented *and tested in the identified revision*; use `[ ]` for work still requiring implementation, integration or acceptable evidence. A checked feature may be **verified numerically but not validated physically**.

**Existing components to reuse, not duplicate:**

- Stage51: `geometry.py`, `fuel_sector_geometry.py`: actual six clipped ZrC channel footprints and OpenMC CSG ownership; corrected fuel mass.
- Stage52: `fuel_sector_conduction_2d.py`: coarse 19-volume symmetric lateral-heat operator.
- Stage53: `fuel_cutcell_conduction_2d.py`: exact cut-cell areas, physical fuel-only face conductances, 19 fuel/ZrC interfaces, coupled implicit thermal step.
- Stage54: `fuel_cutcell_axial_transport.py`: axial solid conduction and per-channel H2 upwinding.
- Stage55: `fuel_cutcell_nonlinear_enthalpy.py`: bounded tabulated `cp(T)`, `k(T)`, film properties, nonlinear solver, enthalpy-based hydrogen transport.
- Stage56: `hydrogen_channel_hydraulics.py`: 19-channel ideal-H2 steady compressible pressure, density, velocity and friction.
- Stage57: `hydrogen_transient_fvm.py`: *separate* 19-channel conservative transient `rho`, `rho*u`, `rho*E` finite-volume flow solver with prescribed external wall heating.
- Existing OpenMC and source/thermal code: `model.py`, `run_openmc_transport.py`, `openmc_cycle_coupling.py`, `fx2th_conjugate_fuel_transport.py`, `thermal_flow_coupling.py`, plus existing neutronic diagnostics/kinetics.
- Stage57 focused CI: 76 Stage52–57 synthetic/regression tests passed; **this is not a coupled hot-core validation**.

**Mandatory implementation conventions**

1. **Do not create a parallel independent NERVA core solver** when extending an existing module suffices. Maintain one authoritative integration entry point.
2. Preserve original physical geometry/material identities while building numerical capability. Do **not** quietly tune enrichment, control drums, reflector or source normalization to make outputs look realistic.
3. The fluid energy state is conserved **total energy** in Stage57; the solid energy state is **integrated internal enthalpy / heat capacity** from Stage55. Do not advance both Stage55's H2 unknowns and Stage57's H2 unknowns for the same fluid during one timestep.
4. Every energy-flow term has one producer, one consumer and unit checks. Transported total energy must include kinetic energy; fluid pressure work is already in the Euler energy flux, not an extra wall-energy source.
5. Distinguish **[implemented]**, **[numerically verified]**, **[evaluated-data qualified]**, **[physically validated]** in tests and reports. Never substitute synthetic watts or surrogate gas properties for the latter two.
6. Test cases must report exact source code revision, geometry, property-data provenance, nuclear-data library and operating-domain limits.
7. Do not delete existing alternatives until the replacement agrees with their documented physical limits and regression tests.

## Priority / dependency map

| Priority | Track / intended next stage | Dependency | Deliverable / exit condition |
|---|---|---|---|
| **P0** | **C01 — Stage58 coupled fuel + H2** | Stage55 + Stage57 | One energy-conservative transient solid/H2 step; pressure/temperature feedback actually exercised |
| **P0** | **C02 — Transport-to-thermal heating** | OpenMC statepoint + geometry consistency | Genuine, provenance-bearing heating field maps conservatively into fuel, coating, coolant/ties |
| **P0** | **C03 — Hot neutron-data qualification** | transport preflight | All required isotopes/material temperatures covered by real evaluated neutron data; explicit rejection otherwise |
| **P0** | **C04 — Neutronic–thermal iteration** | C01–C03 | State-dependent source/heating, temperature and density iteration converges or fails explicitly |
| **P1** | C05 — Resolved solid geometry/contact | C01 | Coating/fuel interfaces and axial/radial conduction with mesh-error evidence |
| **P1** | C06 — Thermophysical real-H2 path | C01 | qualified pressure-dependent EOS and energy derivatives within a documented domain |
| **P1** | C07 — Neutron source and kinetics verification | C02–C04 | independent-seed source/heating uncertainty, consistent source normalization and kinetics limits |
| **P1** | C08 — Robust core numerics | C01–C07 | manufactured solutions, conservation, grid/time-step tests, failure handling |
| **P2** | C09 — Core-only integration/reproducibility | all tracks | one canonical runnable pipeline, pinned inputs, reference results and CI acceptance |
| **Separate** | C10 — Historical reactor-core validation | above + selected NASA baseline | quantitative benchmark matrix; **not counted as completed by the coding checklist** |

The table is **dependency order**, not evidence that any later stage has been implemented.

---

## C01 — Stage58: physically conservative transient fuel/H2 thermal coupling (P0, START HERE)

**Stage58 first numerical acceptance (2026-10-29):** [PR #172](https://github.com/dubakoor13/NUKE/pull/172) at commit `af3be1ce4b42d7cd35c8b66812bd50e3115caead`, [CI run 40848431073](https://github.com/dubakoor13/NUKE/actions/runs/40848431073) **PASS: 9 new tests and 76 previous Stage52–57 regressions**. Demonstration: 1,008 solid and 114 fluid conserved unknowns; 0.393069737 J wall energy; 9.0381e-12 J absolute global first-law residual. Shared gas state, source antisymmetry and acoustic subcycling verified numerically using **synthetic properties and heating**. Still pending: explicit pressure/flow-dependent film law, characteristic boundaries, independent mesh/timestep convergence, complete invalid-domain matrix, full OpenMC heating and physical validation. This is not an actual hot-core benchmark.

## C01 — Stage58: physically conservative transient fuel/H2 thermal coupling (P0, START HERE)

**Current real gap:** Stage55 solves a fully coupled *fuel plus H2 enthalpy* thermal system with fixed H2 zone masses; Stage57 separately solves *H2 compressible mass, momentum and total energy* with **externally prescribed wall watts**. They have not been made into a common time advance. Simply calling both would double-advance hydrogen energy and generally violate conservation.

### C01A. Define a shared physical interface

- [x] **C01.01** Specify the definitive state object: solid temperature per `(axial slab, fuel cut-cell)`; H2 conservative `(rho, rho*u, rho*E)` per `(axial slab, channel)`; optional coating thermal state; simulation time.
- [x] **C01.02** Establish one authoritative mapping between axial edges in Stage54/55 and Stage57. Reject mismatched lengths or number of cells instead of interpolating silently.
- [x] **C01.03** Reuse Stage53's *exact* fuel/ZrC contact arc area matrix, applied to all axial slabs; preserve the six outer-hex-clipped channel footprints.
- [ ] **C01.04** Standardize physical units: W/K interface conductance, W per fluid channel/slab, J stored, Pa, K, kg, m/s, m², m³; validate dimensions and 19 channel indices.
- [x] **C01.05** Ensure each fuel cut-cell wall contact contributes to exactly one H2 channel and each channel receives exactly the sum of contacting solid-wall transfers.
- [x] **C01.06** Expose `H2` thermodynamic quantities (T, rho, p, velocity, internal energy) by conversion from **Stage57's conserved state only**.
- [x] **C01.07** Separate Stage55's *solid-only* thermal residual/solve from its legacy coupled H2 enthalpy residual. Preserve the old solver as a tested constant-volume limiting case, not as a second active fluid integrator.
- [x] **C01.08** Supply the same wall heat `Q_wall[z,ch]` as a negative source to solid energy and a positive source to Stage57 total energy, with no extra `cp(T)*dT` coolant update.

### C01B. Integrate numerical timestep and feedback

- [x] **C01.09** Implement a clearly named coupled-step entry point, e.g. `advance_fuel_hydrogen_core(...)`, with explicit initial state, external deposited heat and boundary data.
- [x] **C01.10** Support at least one stable *partitioned* scheme: predictor wall transfer, transient-fluid advance, updated fluid T, solid implicit solve, iterate wall transfer until convergence.
- [ ] **C01.11** Ensure H2 pressure/density changes alter flow-state thermodynamics used for wall convection and fluid storage, not only displayed diagnostics.
- [x] **C01.12** Recalculate fuel conductivity and heat capacity from Stage55's bounded property tables at each nonlinear iteration; reject extrapolation.
- [x] **C01.13** Recalculate coating conductivity and fluid-side heat-transfer coefficient from valid material/fluid states, documenting the present approximate wall-transfer resistance.
- [x] **C01.14** Use Stage57's acoustic CFL condition and explicit subcycling as necessary, while retaining longer implicit solid thermal steps.
- [x] **C01.15** Ensure total prescribed fuel/neutron deposition to the solid is divided among substeps **once**; do not apply full-step watts repeatedly as energy.
- [x] **C01.16** Ensure wall heat transfer subcycles are conservative even when H2 timesteps differ from fuel timesteps: `sum(Q_fuel_wall*dt) + sum(Q_H2_wall*dt)=0` after summing all substeps.
- [ ] **C01.17** Provide convergence norms for solid T, coolant T, wall-heat integral, and global first-law balance; do not claim convergence from one variable alone.
- [x] **C01.18** Handle iteration failure deterministically: backtrack/halve `dt` or abort with diagnostic fields; do **not** accept unconverged temperatures/negative pressures.
- [ ] **C01.19** Include explicit physical flow-direction treatment at both channel ends. Stage57 wall/transmissive/simple-reservoir boundaries are not characteristic boundary conditions.
- [x] **C01.20** Enforce that solid, coating (if modeled), fluid and external fluxes are counted exactly once in the balance budget.
- [x] **C01.21** Provide the reverse control-volume budget: power deposited to solid minus solid storage equals wall transfer + any declared external flux; fluid storage + outlet total-energy flow minus inlet total-energy flow equals wall transfer and other declared fluid sources.

### C01C. Required Stage58 tests

- [x] **C01.22** *Isolated adiabatic equilibrium:* uniform solid/fluid T and zero flow/heating stay stationary for repeated timesteps.
- [x] **C01.23** *Sealed thermal equilibration:* hot solid/cold hydrogen approach equilibrium with **constant total** solid + fluid thermal/kinetic energy.
- [x] **C01.24** *Uniform distributed heating:* analytically prescribed fuel watts cause exactly the specified integrated energy rise without double counting.
- [ ] **C01.25** *Selective channel heating:* heat one contact region; only physically connected fluid passage receives direct wall energy, and lateral solid spreading follows the Stage53 operator.
- [x] **C01.26** *Pressure-feedback exercise:* wall heating changes hydrogen T and p in a sealed passage, and this changed state is used on the next wall-heat iteration.
- [x] **C01.27** *Open-flow energy accounting:* report H2 inlet/outlet **total** energy including enthalpy/kinetic terms, and compare to solid source and storage.
- [x] **C01.28** *Zero-wall-conductance limit:* recovers independent Stage55 **solid-only** conduction plus Stage57 standalone hydrogen dynamics.
- [ ] **C01.29** *Low-speed fixed-density limit:* reproduces Stage54/55 constant-property reference, within discretization and energy-variable differences.
- [ ] **C01.30** *Axial thermal pulse:* resolve downstream advection plus axial solid conduction and verify direction and conservation.
- [x] **C01.31** *Time-step/subcycling sensitivity:* compare `dt`, `dt/2`, `dt/4` using a fixed reference setup; report errors instead of pass/fail alone.
- [ ] **C01.32** *Wall-contact mapping:* sum Stage53 interface areas and wall-energy transfers independently before/after thermal-flow integration.
- [ ] **C01.33** *Invalid-domain tests:* reject out-of-range T, negative densities/pressures, exceeded Mach/CFL, failed nonlinear iteration, empty source, mismatched meshes.
- [x] **C01.34** Run Stage52–57 regression suites unchanged; ensure Stage58 new tests do not mask earlier failures.

**Stage58 definition of done:** a runnable *single* transient fuel-plus-19-channel hydrogen simulation, one authoritative H2 state, visible pressure/temperature feedback, demonstrated global/local energy closure, passing limiting-case and refinement tests. This is **numerical verification**, not a NASA-hot-core qualification.

---

## C02 — Genuine OpenMC deposited-heating coupling to the physical fuel volumes (P0)

**Stage68 whole-core four-particle heating reconciliation (2026-10-29):** [PR #182](https://github.com/dubakoor13/NUKE/pull/182), [focused CI #40865096620](https://github.com/dubakoor13/NUKE/actions/runs/40865096620) **PASS: 12 new + 178 previous tests (190 total)**. `openmc_full_core_energy_ledger.py` independently reads the existing whole-core `nerva_material_transport` `heating` tally over **nine material categories** and `nerva_particle_heating` `heating` tally over **four particle types: neutron, photon, electron, positron**. Checks native HDF5 source/tally manifest, all nine expected material IDs and four particle identities, `sum/sum_sq` realization moments, photon-enabled continuous-energy mode and source-rate provenance. Rejects significant independent whole-core energy discrepancy using a documented configurable statistical guard and the conservative sum of per-score standard errors (cross-tally covariance not supplied). Separately tracks fuel, hydrogen, charged-secondary deposition and **unresolved nonthermal structural materials**; none of the whole-core totals are incorrectly injected into ONE fuel element. **Synthetic numerical reference:** full material heating **9.613059804 W**, four-particle total **9.613059804 W**, residual **0 W**, charged secondaries **0.4806529902 W**, fuel **3.8452239216 W**, H2 **1.7623942974 W**, unmodeled non-fuel/H2 materials **4.005441585 W**, conservative difference stderr bound **0.08331318497 W**. This does **not** complete C02.10 (still requires genuine mapped heat + unresolved = justified whole-core deposition), nor C03/C10 (real neutron/photon transport, evaluated-data and physical validation); all Stage68 HDF5 tests were SYNTHETIC.



**Stage67 19-channel direct hydrogen deposition source mapping (2026-10-29):** [PR #181](https://github.com/dubakoor13/NUKE/pull/181), [focused CI #40863916063](https://github.com/dubakoor13/NUKE/actions/runs/40863916063) **PASS: 11 new Stage67 tests + 167 previous numerical tests (178 total)**. Updated `nerva_3d_fuel_hydrogen_heating` OpenMC tally to score ordered 19 `CellFilter` coolant cells, with `heating-local` in neutron-only transport and `heating` plus separate neutron/photon particle bins in photon-enabled transport. New `openmc_native_hydrogen_channel_heating.py` parses native HDF5 filter strides and true x-fastest mesh bins, confirms all 19 ordered channel cell IDs and a single physical fuel-element XY mesh, uses *exact circular bore × mesh rectangle* intersection areas, rejects heating scored where a channel does not exist, and conservatively maps source axial bins to `hydrogen_deposition_w[z,19]` accepted by Stage66. Unknown score covariance is bounded by summing marginal standard errors; fuel–hydrogen wall heat remains an INTERNAL ledger and is not added to nuclear direct deposition. Demonstration synthetic native-HDF5: channel **#9**, third axial segment **0.6408706536 W neutron + 0.2403264951 W photon = 0.8811971487 W**. Stage66 input of 10 µs yields **8.811971487e-6 J** direct H2 energy, zero wall transfer and **1.04e-14 J** global first-law residual. **Not physically validated:** source and evaluated nuclear-data provenance externally asserted, fuel-element reused lattice instance not independently identified by native `CellFilter`, and fixture HDF5 data are synthetic; actual OpenMC statepoints and reactor-wide material-deposition energy closure remain pending. Do not mark C02.07, C02.14, C03 or C10 complete based on this work.



**Stage66 independently conserved direct H2 deposited energy (2026-10-29):** [PR #180](https://github.com/dubakoor13/NUKE/pull/180), [focused CI #40862701904](https://github.com/dubakoor13/NUKE/actions/runs/40862701904) **PASS 9 new tests + 158 prior Stage52–65 regressions (167 total)**. The Stage58 coupled core now accepts a distinct **direct hydrogen nuclear source** `hydrogen_deposition_w[z,19]` alongside `fuel_deposition_w[z,fuel_cutcell]`. Hydrogen direct deposited watts are added to Stage57 conservative H2 total energy, **NOT** to the internal solid↔coolant wall-flux ledger; global first law uses both externally deposited fuel and H2 energy with boundary total-energy flux. Return values separately report per-axial/channel direct H2 joules and integral fuel/H2/wall energy. Synthetic example (5 µs): fuel direct nuclear deposition **0.000406094882806 J**; hydrogen direct deposition **0.209 J**; internal fuel-to-H2 wall energy **0.168397486158 J**; solid energy change **−0.167991391281 J**, H2 total-energy change **0.377397486158 J**; first-law residual **−6.14e−12 J**. Tests cover isolated heated channel, open/closed flow, subcycling, combined wall/volume deposition, unchanged zero-source limit and invalid negative input rejection. **This closes the core solver's missing separate coolant-energy *input*, not the real OpenMC-to-19-channel deposition extraction.** C02.07, C02.14 and C10 stay unchecked until actual identifiable coolant tallies and verified source normalization exist.



**Stage65 coupled neutron + transported-photon HEATING scoring (2026-10-29):** [PR #179](https://github.com/dubakoor13/NUKE/pull/179), [focused CI #40861755228](https://github.com/dubakoor13/NUKE/actions/runs/40861755228) **PASS: 10 new Stage65 tests + 148 preceding numerical regression tests**. Corrected the actual fuel OpenMC tally configuration in photon-enabled mode to score **`heating`** with `MeshFilter`, one fuel `MaterialFilter` and two `ParticleFilter` bins, neutron and photon. The new `openmc_native_coupled_fuel_heating.py` reads native `[mesh,material,particle]` filter strides using x-fastest source voxels, preserves neutron and photon deposited-energy maps separately, and bounds unknown neutron/photon tally covariance by the sum of marginal standard errors. Stage61 spatial remapping now conserves both components independently and sends their **sum exactly once** into Stage58. Synthetic example: **5.075695576512 W neutron + 1.884159721584 W photon = 6.959855298096 W**, energy remapping residual **−8.88e−16 W**, conservative uncertainty bound **0.0845949262752 W**. Mode-incompatible `heating-local`, missing particle bins, corrupted native filter stride or inconsistent combined-component maps are rejected. For neutron-only transport `heating-local` already assumes locally deposited secondary-photon energy; it cannot be combined with an independently transported-photon tally. All Stage65 CI HDF5 data are SYNTHETIC and source rate is caller-provided; no actual coupled transport data or qualified NASA reactor power. Genuine source/instance/data validation C02.14 and C03 remain pending. OpenMC physical score definitions: https://docs.openmc.org/en/stable/methods/energy_deposition.html



**Stage64 physical-voxel indexing correction (2026-10-29):** [PR #178](https://github.com/dubakoor13/NUKE/pull/178), [focused CI #40860355726](https://github.com/dubakoor13/NUKE/actions/runs/40860355726) **PASS: 5 new and 143 earlier Stage52–63 numerical tests**. Independent reading of upstream OpenMC C++ `src/mesh.cpp` `StructuredMesh::get_bin_from_indices()` proves the native tally's structured mesh bin mapping is `bin = i + nx*(j + ny*k)` (x fastest; **Fortran-order NumPy reshaping**). Stage60/61 had incorrectly read native tally results as **z fastest**, which preserved total heating while placing it in wrong fuel axial zones and physical cut-cells. Stage64 corrects HDF5 mean/stderr reshape and test fixture flattening, Stage61's source-area/XY overlap column indexing and 3D source-energy projections. The independent 3D **one-hot nuclear heating voxel** test enforces the source bin number and physically correct x/y/z location, not only net energy; example synthetic source maps **1.1215236438 W to axial slab #3 only**, 0 W elsewhere, with 0.0 W residual. **Still pending:** native statepoint generated by a real neutron-transport execution, individual fuel-element source identity, source normalization and physically validated neutron/photon deposition. Passing spatial geometry tests does not qualify a physical reactor.



**Stage63 native OpenMC format correction (2026-10-29):** [PR #177](https://github.com/dubakoor13/NUKE/pull/177), [focused CI #40859515004](https://github.com/dubakoor13/NUKE/actions/runs/40859515004), **PASS: 10 new source-format tests + 133 Stage52–62 regression tests**. Independent review of *actual upstream* `openmc-dev/openmc` `src/state_point.cpp` and `include/openmc/constants.h` revealed a material Stage62 error: genuine HDF5 `version` is a two-element integer array `[18,2]`, not scalar `18`. The old parser would reject valid real OpenMC statepoints while synthetic tests passed. Stage63 fixes version parsing to the format OpenMC's `check_filetype_version` expects (major 18, valid nonnegative minor), adds the `continuous-energy` transport-mode check and requires `current_batch`. Tally active-realization capacity is now checked against `current_batch - n_inactive`, NOT the configured `n_batches - n_inactive` (important for intermediate statepoints). The run metadata are retained. **These are still native-layout synthetic fixtures**; CI does not claim a real OpenMC-generated statepoint, physically identified individual fuel element, evaluated neutron cross-section coverage or qualified watt normalization. Complete C02.14/C03 only after independent real-data exercises and documentation.



**Stage62 HDF5 source-mode and tally evidence audit (2026-10-29):** [PR #176](https://github.com/dubakoor13/NUKE/pull/176), [focused CI #40858591607](https://github.com/dubakoor13/NUKE/actions/runs/40858591607), **PASS: 12 new tests + 121 previous tests**. Stage60 native parser now refuses heating-local tally data if the **actual statepoint's** `photon_transport` HDF5 flag is missing, malformed or enabled, even when a caller falsely claims neutron-only mode. Enforces native OpenMC statepoint file schema version 18, OpenMC version metadata, generation/inactive/particle counts, tally ID manifest, selected tally membership and `n_realizations <= n_batches - n_inactive`. Records native run metadata with derived nuclear heating and computed file SHA256. **This is synthetic native-layout HDF5 verification**; no real OpenMC-generated statepoint was evaluated, and an external nuclear-data SHA256 string is not independently verified. Reactor material/element identity, evaluated cross sections, source convergence, defensible physical source normalization and dual neutron-photon scoring **remain pending**. Do not mark C02.01, C02.03, C02.11, C02.14, C03 or C10 as physically complete.



**Stage61 exact spatial handoff (2026-10-29):** [PR #175](https://github.com/dubakoor13/NUKE/pull/175), [CI #40855157471](https://github.com/dubakoor13/NUKE/actions/runs/40855157471) **PASS: 11 Stage61 tests + 109 earlier numerical regression tests**. New `openmc_native_spatial_fuel_heating.py` preserves true source x/y/z spatial heating, computing source-to-thermal overlapping areas from the original hex and 19 exact circular coated-channel exclusions; uses the source-voxel fuel-only area instead of distributing all reactor power by whole-element volume. Explicitly **rejects default reactor-wide meshes** when asked to map them to one resolved hex. Verified per-voxel source-column conservation and cutcell target-row area closure; native marginal stderr bounds use the same positive overlap weights. Synthetic example: 79.2 eV/source × 10¹⁸ source/s = 12.68923894128 W mapped, residual 0 W, uncertainty **upper bound** 0.4287424672584 W. The caller still must assert isolated element geometry; an explicit native OpenMC cell/instance provenance audit has **NOT** been implemented and cannot be inferred from a material-only tally. All CI HDF5 data are synthetic, not real OpenMC transport. Neither a whole-reactor tally nor an unverified element-instance sum is a valid physical NERVA heat source.



**Stage60 HDF5 parser / uncertainty bridge (2026-10-29):** [PR #174](https://github.com/dubakoor13/NUKE/pull/174) at code commit `f93d82c7cfcf432789eada7980ff018841603ce3`, [focused CI #40853860684](https://github.com/dubakoor13/NUKE/actions/runs/40853860684) **PASS: 12 HDF5 tests and all 97 Stage52–59 numerical regressions**. Native-format statepoint filetype, eigenvalue run mode, tally score/name/filter/material identity and regular 3D mesh geometry are checked. Native `sum`/`sum_sq` realizations are converted to mean+standard error; x/y mesh means become axial fuel eV/source, file SHA256 is computed rather than merely accepted. Positive linear source→axial→fuel-volume weights propagate **conservative uncertainty upper bounds**, with within-batch cross-bin covariance explicitly unavailable. Missing source-rate evidence, wrong fuel material or unsupported photon-enabled cases are rejected. **All CI statepoints are synthetic native-layout HDF5 fixtures:** actual neutron transport HDF5 files, evaluated-data hashes, historical fuel power and real photon tally coverage have **not** been independently validated. Do not mark C02.03, C02.11, C02.14 or C03 as complete from this run.



**Stage59 partial numerical implementation (2026-10-29):** [PR #173](https://github.com/dubakoor13/NUKE/pull/173) at commit `8e4d3f7e1b5cd0cd98d92867a54520ec596e6df7`, [focused CI run 40850469318](https://github.com/dubakoor13/NUKE/actions/runs/40850469318): **12 new tests + all 85 previous Stage52–58 tests PASS**. Added `openmc_fuel_axial_heat_handoff.py` to accept *exported* axial neutron and photon heating scores **eV per source particle**; physical W requires caller-supplied source rate and source-rate basis. Axial-bin overlap and actual Stage53 fuel-volume partition conserve heat (example 22.430472876 W -> 22.430472876 W, 3.55e-15 W residual, 4.806529902 W nonfuel deposition separately unresolved). Map retains SHA256 labels, explicit transverse-uniform approximation and exact target geometry guards. **Missing:** native OpenMC HDF5 parser, actual hash verification, material-instance/source/tally identity, independent source normalization validation, true transverse score reconstruction and uncertainty propagation. This partial work is **not** verified hot-core heating.

## C02 — Genuine OpenMC deposited-heating coupling to the physical fuel volumes (P0)

**Existing:** OpenMC statepoint/tally tools and multiple real transport/source-investigation experiments exist; Stage51–57 geometric/thermal volumes are usable. **Not demonstrated as a fully qualified, reactor-power-driven coupled hot core.**

- [ ] **C02.01** Define a canonical source-data contract including OpenMC statepoint path, geometry/material hash, cross-section library metadata, batch/seed, estimator type and tally normalization.
- [x] **C02.02** Explicitly distinguish tally output *per starting/source neutron* from an independently justified physical power [W].
- [x] **C02.03** Extract separate neutron and photon deposited-heating components where the chosen scored physics requires both; report missing contributions rather than silently equating neutron-only heating to total reactor watts.
- [x] **C02.04** Preserve tally mean, standard deviation, covariance or correctly documented unavailable covariance through mapping.
- [ ] **C02.05** Map HDF5 fuel-element instance and axial heating into each physical Stage53 cut-cell and Stage54 axial slab using an explicit overlap/conservation operator.
- [x] **C02.06** Do not uniformly smear a coarse tally into fine local cells without labeling the assumed reconstruction and its unresolved spatial error.
- [ ] **C02.07** Include coating, tie-tube, structural and coolant energy-deposition channels only where distinct physical OpenMC material/cell tallies actually support them; retain an explicit `unresolved` bucket otherwise.
- [ ] **C02.08** Detect and reject identical geometry instance indices mistakenly counted twice, including the six formerly overlapping ZrC caps.
- [x] **C02.09** Confirm mesh/tally volume weighting and flux-source mapping respect the 132-cm axial length and actual fuel-only volumes.
- [ ] **C02.10** Check `sum(mapped heat) + unresolved = justified total deposited heat` within reported stochastic and numerical uncertainties; never force closure through arbitrary residual rescaling.
- [ ] **C02.11** Explicitly record external physical normalization inputs (actual reactor power/energy balance), with uncertainty and source. Invalid/absent inputs must produce a hard failure or *per-source-particle* diagnostic mode, not fake watts.
- [ ] **C02.12** Expose one typed heating interface directly consumable by Stage58; avoid five competing adapters that each have a different normalization convention.
- [ ] **C02.13** Test 1-bin, nonuniform 3D, and mixed cell/mesh instances with exactly integrable synthetic watt patterns.
- [ ] **C02.14** Test actual OpenMC HDF5 availability/provenance and rejected incomplete photon/evaluated-data cases independently from synthetic-only CI.

**Exit:** physically traceable and uncertainty-bearing *genuine transport* heat can reach the same fuel/H2 volumes without changing total deposition.

---

## C03 — Evaluated-data and actual hot neutron transport qualification (P0)

**Existing:** OpenMC hot-temperature preflight and explicit evaluated-data testing have advanced significantly (including actual interpolation experiments); cold-only nuclear libraries and unsupported temperature points must still be rejected. A successful hot neutron run is not automatically a validated NERVA reactor.

- [ ] **C03.01** Freeze a physically identifiable *reference reactor configuration* before declaring any meaningful core operating-point calculation; distinguish its literature evidence from development surrogates.
- [ ] **C03.02** Inventory **all** isotopes appearing in occupied transport materials, with density, nuclide composition and relevant neutron/photon reactions.
- [ ] **C03.03** Record provenance and SHA of evaluated neutron and secondary-photon datasets, plus OpenMC transport version and temperature interpolation method.
- [ ] **C03.04** For each hot material T, verify bracketing physically evaluated cross-section tables; reject out-of-range T and nearest-table substitution when not justified.
- [ ] **C03.05** Check thermal-scattering kernel compatibility for moderating materials and note temperature/data gaps instead of making up kernels.
- [ ] **C03.06** Confirm material temperatures used by actual OpenMC materials match the fluid/solid state being solved, including coating/structural materials.
- [ ] **C03.07** Run geometry-debug and material-identity audits at each significant operating configuration; preserve Stage51 no-cell-overlap checks.
- [ ] **C03.08** Run genuine **multi-seed** OpenMC simulations at supported temperatures; report `k_eff`, distribution, cross-section coverage, source histories and tally uncertainty. Do not tune enrichment or source to fabricate representative criticality.
- [ ] **C03.09** Distinguish strongly subcritical conceptual surrogate configurations from physically representative reactor reference states. Record inability to establish a qualified baseline as a blocker.
- [ ] **C03.10** Demonstrate robust active-source spatial convergence and isolate initialization/generation dependence; compare independent starting sources and repeated seeds.
- [ ] **C03.11** Verify native neutron/photon heat tallies and per-material/instance reconciliation independently from user-side aggregation.
- [ ] **C03.12** Re-run accepted transport cases with alternate particle counts/active generations and compare confidence intervals, not only raw means.
- [ ] **C03.13** Implement explicit transport acceptance results: `data_qualified`, `source_converged`, `heating_complete`, `geometry_valid`, `physical_baseline_supported`; each must be true before an asserted physically predictive hot-core result.

**Exit:** a reproducible transport case with honest physical/evaluated-data domain and defensible heat/statistics for the selected *reference*, not a source- or geometry-tuned imitation.

---

## C04 — Iterated neutronic–thermal feedback with independent convergence (P0)

- [ ] **C04.01** Define a single state vector transferred between OpenMC and Stage58: fuel/structural T, material density where justified, H2 pressure/T/density, and physical heating field.
- [ ] **C04.02** Use the same physical material/region identities and consistent axial/cross-sectional volume operators in both directions.
- [ ] **C04.03** Specify the supported coupling mode: quasi-static heat-shape iteration, outer-step transport refresh, or demonstrably valid transient kinetics; never imply Monte Carlo transport is instantaneously solved every acoustic substep.
- [ ] **C04.04** Rebuild or update OpenMC material state only within qualified evaluated-data temperature intervals, with explicit physical treatment of density and geometry change.
- [ ] **C04.05** Separate source normalization/reaction-rate change from thermal-shape change, so `power`, `source particles`, `k_eff` and total deposited heat are not conflated.
- [ ] **C04.06** Apply physical feedback: changed fuel temperature and hydrogen density modify the next **transport** calculation where consistent.
- [ ] **C04.07** Implement outer iteration with under-relaxation, bounds, trust/convergence checks and a documented failure mode; conserve heating at each accepted iterate.
- [ ] **C04.08** Check *at least* fuel temperature, H2 state, spatial heating, total deposited energy and transport statistical convergence before success.
- [ ] **C04.09** Include stochastic-error-aware stopping conditions so iteration does not chase random tally noise.
- [ ] **C04.10** Test fixed-zero feedback, synthetic linear feedback, and deliberately divergent feedback against analytical solutions.
- [ ] **C04.11** Record every outer-iteration state, material/T hash, source seed and mapping residual; preserve restart capability.
- [ ] **C04.12** Demonstrate at least one full supported-data transport→thermal→transport cycle with a quantified energy balance, without claiming a NASA-qualified output unless C10 is complete.

**Exit:** actual T/density neutronic feedback is exercised and shown to converge *with* quantified Monte Carlo uncertainty.

---

## C05 — Improve solid fuel/coating/tie-interface thermal physics (P1)

**Stage73 three-temperature angular fuel–ZrC–H2 physical coupling (2026-10-29):** [PR #187](https://github.com/dubakoor13/NUKE/pull/187), [focused CI #40879685872](https://github.com/dubakoor13/NUKE/actions/runs/40879685872), **PASS: 15 new + 240 previous Stage52–72 numerical regressions = 255 total**. New `coating_angular_interfaces.py` splits Stage53's *actual* outer coating–fuel cylinder arcs by six inner-hex facets, Cartesian fuel-cutcell edges and individual Stage72 angular-sector edges; independently splits the **inner** H2 bore arcs by the same angular bins and hex. Contact area sums reconcile *per cutcell/channel* with Stage53 and *per H2 channel* with Stage70 (synthetic CI reference maximum residual **2.1684e−19 m²** fuel, **0 m²** H2). Signed fuel→angular-coating and angular-coating→H2 wall watts are equal/opposite, with separate temperature-dependent fuel conductivity, coating radial resistance, explicit contact resistance and H2 film. The existing authoritative Stage58 `advance_coupled_core` now supports **opt-in `angular_sectors=N` with axial conduction enabled**, temperature `[z,19,angle]`, the Stage72 internally conserved nonlinear axial+azimuthal enthalpy operator, Stage57's one compressible-H2 solver, and independent fuel/coating/H2 energy source ledgers. Coating sources `[z,19]` are optionally split by physical angular volume as an **explicit uniform-volumetric assumption**; true measured `[z,19,angle]` sources can instead be supplied without redistribution. Full-loop tests cover independent three-temperature first law, insulating H2, one-sector nuclear coating heat, zero-source uniform equilibrium, physically exact interface area, unchanged legacy solver and invalid state rejection. Synthetic isothermal-wall reference: **215392.0449 W** fuel→coating, **12106.6976 W** coating→H2, assembled antisymmetric flux residuals **0 W** fuel and **1.82e−12 W** coolant. **Not finished:** C05.03 retains outstanding *radial through-thickness coating conduction DOFs* (the coating remains one radial enthalpy node per sector with explicit half-shell resistances), and no real OpenMC angular coating heating, independent physical source-rate normalization, validated high-T coatings, outer-hex/tie contacts or real NERVA operating point are claimed. C05.02 wall-normal finite-volume spatial convergence also remains open.



**Stage72 physically resolved angular ZrC conduction operator (2026-10-29):** [PR #186](https://github.com/dubakoor13/NUKE/pull/186), [focused passing CI #40877800931](https://github.com/dubakoor13/NUKE/actions/runs/40877800931), **PASS 12 new + Stage52–71 228 regressions = 240 total**. New `coating_angular_fvm.py` gives each of the **19 independent cooling-channel sleeves** separate angular thermal cells in addition to Stage71 axial cells: `T_coating[z,channel,angle]`. A hex-clipped annular sector has true volume obtained by integrating the local permitted outer radius over angle against all six oriented inner-hex halfplanes; the accumulated sector masses independently reconcile with Stage70 exact clipped coating volumes. Circumferential faces use the curved-wall resistance derived by radial integration `log(r_outer(theta_face)/r_bore)`, with periodic coupling ONLY between angular cells of the same physical annulus, **never between separate H2 bores**. Axial faces use Stage70 physical cross-sectional area and half-cell series resistances. Both conductances use positive temperature-dependent `k(T)`; the implicit nonlinear solver integrates bounded tabulated `cp(T)` exactly as enthalpy, with checked equal-and-opposite face fluxes, no negative masses and no extrapolation. Independent test suite compares a periodic Fourier eigenmode and analytical one-dimensional axial reduction, verifies uniform equilibrium, variable k/cp, spatial spreading, geometry mass, 19-channel isolation and global/per-node first law. Synthetic 2 axial zones × 19 passages × 16 angles: **0.050916020118 kg ZrC**, 608 circumferential + 304 axial faces, energy residual **6.06e−16 J** and face-cancellation residual **−8.38e−16 J**. **IMPORTANT NOT YET COMPLETE:** This is a standalone physical *internal* coating conduction operator. It is deliberately **not yet coupled to Stage58** because fuel-cutcell-to-angular coating contact arcs, inner coated bore-to-H2 angular heat flux, angular nuclear source distribution and physically exact state/flux mapping must be implemented before its unknowns can replace Stage70's `[z,19]` coating state. Thus `C05.03` remains open, as do OpenMC evaluated-data and historical hot-reactor validation. All material properties and heat sources in CI are synthetic.



**Stage71 conservative axial coating diffusion (2026-10-29):** [PR #185](https://github.com/dubakoor13/NUKE/pull/185), [focused successful CI #40871295127](https://github.com/dubakoor13/NUKE/actions/runs/40871295127), **PASS 12 new analytical/core tests + 216 Stage52–70 regressions = 228/228**. Added optional `axial_conduction_enabled=True` to the resolved Stage70 coating model, within the authoritative Stage58 three-temperature advance. Models *each of 19 physically separate coating annuli* along the physical axial mesh with exact annular cross-sectional area extracted from clipped Stage70 geometry, harmonic half-cell axial thermal resistance using bounded temperature-dependent ZrC `k(T)`, and implicit exact-`cp(T)` enthalpy storage; conservative interface fluxes cancel between adjacent nodes. The nonlinearly implicit solve uses an SPD frozen-k diffusion Jacobian, safeguarded line search and refuses extrapolation beyond declared material tables. Positive and negative axial interface transfer is separately recorded in joules; no fuel↔coating↔H2 double-counting or invented cross-conductance between distinct coolant bores. Manufactured references include two-slab exact implicit matrix, isolated heat pulse, zero mode, constant and variable material laws, negligible-k limit, one-slab reduction to Stage70, physical-source distribution, and Stage58 full-system first-law tests. **Synthetic three-slab illustration:** one channel cools/spreads from `[600,450,450] K` to `[596.802496,453.313383,450.079047] K`, axial interface energies `[2.396916682,0.055808074] J` and global source residual `1.95e-9 J`. **Still pending C05.03:** true azimuthal/circumferential conduction needs NEW angular coating unknowns in each annulus with correctly clipped contact geometry; direct conduction between different physical H2 bores would be incorrect. Coating properties and nuclear watts remain synthetic; real OpenMC statepoint material/instance mapping, verified heat generation, tie tubes and outer coating remain outstanding.



**Stage70 resolved channel-coating enthalpy and three-temperature coupling (2026-10-29):** [PR #184](https://github.com/dubakoor13/NUKE/pull/184), [focused CI #40869721386](https://github.com/dubakoor13/NUKE/actions/runs/40869721386) **PASS: 12 new and 204 previous numerical tests (216 total)**. New `resolved_coating_thermal.py` resolves all **19 axial ZrC-coated bores** as separate finite coating enthalpy states `T_coating[z,19]`. Coating volume/mass is computed using analytic hexagon/outer-disc minus hexagon/bore-disc intersections, including six peripheral clipped channel footprints, and contact geometry is audited against Stage53 cut-cell fuel arcs. Two different physical conductances are modeled: **fuel↔coating** (fuel centre-to-wall distance, outer coating half-shell, user-supplied area-specific contact resistance) and **coating↔H2** (inner coating half-shell plus Stage55 fluid film). Bounded tabulated cp(T) is integrated exactly; one Stage57 compressible H2 total-energy integrator remains authoritative. The **existing** `advance_coupled_core` now accepts optional resolved-coating state and direct `coating_deposition_w[z,19]`; when disabled it preserves prior Stage58 exactly. Local fuel/coating/H2 energy equations and global first law are checked separately, with antisymmetric internal transfers. Synthetic demonstration (5 µs): fuel deposited **0.013920 J**, coating deposited **0.000570 J**, H2 deposited **0.000950 J**, fuel→coating **0.816140 J**, coating→H2 **0.063047 J**, coating storage **0.753663 J**, combined residual **9.45e−12 J**, coating residual **−1.35e−12 J**. **Physical limits:** coating is a single radially lumped annulus per (axial,channel), not 2D tangential/axial conduction; outer element ZrC coating/ties and structural interfaces remain unmodeled, mass/thermal tables/contact R are synthetic, and no actual instance-identified OpenMC coating deposition has been extracted. **Do NOT mark C05.03 or C02.07 or historical validation complete solely from this stage.**



**Stage69 whole-core structural enthalpy substrate (2026-10-29):** [PR #183](https://github.com/dubakoor13/NUKE/pull/183), [first numerical CI #40867638953](https://github.com/dubakoor13/NUKE/actions/runs/40867638953): **PASS 14 new + all previous 190 Stage52–68 tests (204 total)**. New `structural_enthalpy_network.py` retains the 7 Stage68 nonfuel/non-H2 material identity buckets (ZrC, reflector, graphite, ZrH, Inconel, B4C, aluminum) as **whole-core** finite heat-capacity states with caller-provided positive material masses, bounded exact-integral `cp(T)` tables, symmetric nonnegative intermaterial W/K conductances, optional explicit reservoir heat loss, nonlinear implicit enthalpy time advance, per-material first-law residuals and antisymmetric intermaterial fluxes. The Stage68 nine-material whole-core source allocation is audited before the seven structural watts enter this subsystem; **fuel and hydrogen whole-core watts are retained outside its source vector**. 14 numerical tests include analytical constant-cp, two-node implicit conduction, nonlinear cp storage, reservoir export, repeated-step closure, genuine category identity and malformed-property rejection. **Synthetic HDF5 sources and synthetic masses, contacts, cp tables** only; these material aggregates currently **DO NOT** exchange heat with Stage58's ONE fuel element. **C05.03 (resolved coating storage/contact geometry), C05.09 (tie-tube/periphery exchange), C02.10 (real global source closure), C03 and C10 remain unchecked**. A physical coupled core requires resolved coating thickness/mass and validated fuel–coating–H2/tie/contact interfaces, not an arbitrary 7-node contact matrix.



- [ ] **C05.01** Decide and document the one reference cross-section representation used by OpenMC, Stage53 cut cells and 3D thermal extrusion.
- [ ] **C05.02** Replace Stage53's approximate half-cell fuel-to-wall distance with a demonstrably convergent embedded-boundary flux treatment or body-fitted reconstruction near circular holes.
- [ ] **C05.03** Model coating thermal storage and tangential/through-thickness conduction if its timescale materially affects the selected benchmark; alternatively document a justified zero-storage limit.
- [ ] **C05.04** Represent effective fuel↔coating contact resistance and uncertainty as explicit user-supplied properties, not hidden constants.
- [ ] **C05.05** Model outer-hex coating contact and external conduction appropriately; Stage53 currently assumes adiabatic exterior.
- [ ] **C05.06** Support axial-variable properties, optional anisotropy, and material-region changes without breaking symmetric positive diffusion and energy closure.
- [ ] **C05.07** Handle fuel/coating thermal expansion and density changes only where physical/model data support them; avoid inconsistent simultaneous geometry and mass changes.
- [ ] **C05.08** Preserve exact mass/volume for every source region after axial extrusion, and test fuel + coating + channels + tie geometry against transport geometry.
- [ ] **C05.09** Implement documented tie-tube and relevant periphery thermal exchange *if included in the chosen core reference*. Reuse the repository's existing tie models; avoid rebuilding them unconnected.
- [ ] **C05.10** Evaluate axial/radial mesh sensitivity of local peak temperature, wall heat flux, spatial heat-flow paths, not only total thermal energy.
- [ ] **C05.11** Test material discontinuities, zero-contact limit, high-contact limit, diffusion eigenmodes and 2D/3D manufactured temperature solutions.
- [ ] **C05.12** Flag unsupported damage, stress, corrosion or high-temperature material-failure predictions as not implemented unless separate physics and evidence exist.

**Exit:** thermal field and wall heat flux converge under mesh refinement with material interfaces represented at the fidelity demanded by the reference case.

---

## C06 — Hydrogen material physics and fluid numerical-domain closure (P1)

**Existing:** Stage56/57 ideal H2 transport; separate earlier repository CoolProp/tie-tube/cryogenic modules exist. Do not confuse these two paths.

- [ ] **C06.01** Choose one authoritative EOS path per operating state, with a documented *valid T–p–phase domain* and exact thermodynamic variables needed for the transient total-energy formulation.
- [ ] **C06.02** If using real-gas properties, implement `p(rho,e)`, `T(rho,e)`, derivatives/sound speed and transport properties from consistent thermodynamic potentials; do not replace `rho=p/RT` alone and keep incompatible ideal energy formulas.
- [ ] **C06.03** Add property-table bounds and explicit failure for unsupported phase, supercritical or dissociation regimes; do not silently extrapolate.
- [ ] **C06.04** Check `h=e+p/rho` and thermodynamic derivative consistency in the selected EOS; test reversible zero-heat compression/expansion in a qualified regime.
- [ ] **C06.05** Audit Stage57's positivity and temperature inversion across every RK stage, low flow, heating, pressure waves and variable cp.
- [ ] **C06.06** Implement and verify a well-posed subsonic characteristic inlet/outlet treatment appropriate to expected boundary states; preserve wall/simple-reservoir controls as tests only.
- [ ] **C06.07** Define verified behavior near flow reversal and subsonic/choked transitions. Stage57 presently *rejects* Mach outside its qualified subsonic bound.
- [ ] **C06.08** Use mass-flux-consistent Reynolds, viscosity, friction and convection relationships. Surface dependencies on actual pressure, temperature and flow.
- [ ] **C06.09** Verify friction work appears as internal-energy dissipation (not a duplicate loss of total energy); check pressure-work terms and open-boundary kinetic enthalpy.
- [ ] **C06.10** Compare constant-property ideal-gas solutions and existing Stage56 steady pressure profiles in their common validity limit.
- [ ] **C06.11** Resolve how per-channel flow and mass inventory are coupled at core inlet/outlet; do not prescribe mutually inconsistent `mdot`, pressure and reservoir states as independent boundary conditions.
- [ ] **C06.12** Add fully documented numerical stopping/rejection for Mach, temperature, density, EOS failures, acoustic CFL, conservation and nonlinear outer iteration.

**Exit:** consistent energy/EOS/pressure feedback in a documented domain, with no hidden fictitious thermodynamics.

---

## C07 — Neutronic source, heating statistics, and core kinetics interoperability (P1)

- [ ] **C07.01** Confirm actual reactor-core source/convergence diagnostics are still relevant after Stage51 geometric fixes and correct material temperature updates.
- [ ] **C07.02** Review and resolve reported real OpenMC fission-source reproducibility/sensitivity anomalies using native engine evidence, not user-side synthetic bank replays.
- [ ] **C07.03** Quantify independent-seed variability of both `k_eff` and spatial **heating**, separating Monte Carlo uncertainty from discretization/coupling error.
- [ ] **C07.04** Maintain stochastic tally covariance where supported and propagate uncertainty through the C02 source-to-temperature map.
- [ ] **C07.05** Make the existing r–z multigroup diffusion/precursor models explicitly distinguish transport-derived physically meaningful coefficients from analytic surrogates.
- [ ] **C07.06** Verify prompt/delayed neutron balance, source normalization, precursor conservation and spatial adjoint calculations under consistent units.
- [ ] **C07.07** Couple the kinetics *timescale* to thermal/H2 evolution only if a physically justified power normalization and cross-section/kinetics model exist.
- [ ] **C07.08** Demonstrate limits: fixed-power case, zero-feedback case, delayed-neutron-only test, controlled nonzero-feedback case, and unphysical-input rejection.
- [ ] **C07.09** Establish statistical error envelopes and independent benchmark calculations for any claimed power/temperature transient; synthetic kinetic constants are numerical demonstrations only.

**Exit:** heating/kinetics inputs have traceable physical meaning and their numerical uncertainty enters the thermal calculation.

---

## C08 — Numerical verification, convergence, robustness, and rejection policy (P1)

**Stage59 independent numerical observations (2026-10-29):** [focused CI 40850469318](https://github.com/dubakoor13/NUKE/actions/runs/40850469318) reports Stage58 sealed coupled thermal timestep errors versus 8 steps: **[0.00133920, 0.000589794, 0.000199376] K** for 1, 2, 4 steps; decreasing but not proof of a formal time-order study. With 20×24, 28×34, 40×46 fuel meshes, wall boundary conductance error versus zero-fuel-half-cell-resistance limit decreases **[1587.44, 1139.88, 848.36] W**. This is **one wall-flux trend**, not C08.02's full spatial peak-temperature convergence.

## C08 — Numerical verification, convergence, robustness, and rejection policy (P1)

- [ ] **C08.01** Define a minimal independent analytic/method-of-manufactured-solutions suite for diffusion, advection, interface exchange, compressibility and integrated enthalpy.
- [ ] **C08.02** Compare 2D fuel grids across at least three resolutions; report cell count, flux/temperature errors, peak-temperature convergence and numerical order where measurable.
- [ ] **C08.03** Compare multiple axial mesh sizes (fuel and H2 jointly), explicitly checking gradients, pressure waves and wall-flux shape, not only global inventory.
- [x] **C08.04** Compare `dt`, `dt/2`, `dt/4` on the combined Stage58 model and show expected temporal behavior away from nonsmooth events.
- [ ] **C08.05** Quantify splitting error from thermal–fluid subcycling and compare against a tighter-reference combined solution.
- [ ] **C08.06** Include a standard 1D ideal-gas Riemann/shock-tube benchmark with reference solution inside the Stage57 property/Mach domain, and explicitly reject tests outside that domain.
- [ ] **C08.07** Test traveling acoustic perturbations, reflective walls, steady uniform-flow preservation and finite-volume contact discontinuities under relevant regimes.
- [ ] **C08.08** Verify total mass and energy independently by two methods: control-volume differences and integrated *actual numerical boundary/source face fluxes*.
- [ ] **C08.09** Test solid–fluid interface antisymmetry to floating-point scale under the same transfer-energy accumulator.
- [ ] **C08.10** Test zero source, zero flow, hot/cold initialization, near-boundary property temperatures, extreme but admissible conductivity and weak/strong contact.
- [ ] **C08.11** Confirm unphysical states are rejected rather than silently clipped to valid-looking outputs; document when conservative recovery/retry is used.
- [ ] **C08.12** Verify tolerance changes do not falsely create passing cases. Relate residual tolerance to absolute energy scale, floating-point error and time discretization.
- [ ] **C08.13** Check sparse operator structure, symmetry for diffusion, invertibility/conditioning and solver behavior at larger 2D × axial grids.
- [ ] **C08.14** Profile computational cost separately from physical correctness and ensure no hidden dense matrix assembly at reactor-relevant grid counts.
- [ ] **C08.15** Run CI with deterministic synthetic tests for each new module, plus separate opt-in evaluated OpenMC regressions using pinned physics-data containers.
- [ ] **C08.16** Report side-by-side `implemented / regression-pass / real-transport-pass / physically benchmarked` gates for each subsystem.

**Exit:** reproducible numerical accuracy and stability evidence independent of just the equations' self-reported conservation residuals.

---

## C09 — One canonical core simulation and reproducible handoff (P2)

- [ ] **C09.01** Choose one canonical core-only driver, with the full chain `state -> OpenMC heating evidence -> physical fuel cut cells -> fuel/H2 transient -> state + diagnostics`.
- [ ] **C09.02** Keep legacy reduced models available with documented compatibility/limits, but do not expose five equivalent default solver entry points.
- [ ] **C09.03** Make all parameter sources explicit: selected baseline, material tables, nuclear data, solid/fluid boundary state and deposited-heating provenance.
- [ ] **C09.04** Validate schema shape, material identity, mesh size and physical units before expensive transport/thermal calculation.
- [ ] **C09.05** Emit canonical HDF5/JSON or equivalent results: state histories, exact integrated heat, boundary fluxes, conservation residuals, solver residuals and validity flags.
- [ ] **C09.06** Record actual code SHA, dependent PR/branch, runtime/compiler/OpenMC build, RNG seeds and nuclear-data hashes.
- [ ] **C09.07** Supply a deterministic zero/low-power synthetic smoke case requiring no external evaluated data, **clearly labeled synthetic**.
- [ ] **C09.08** Supply a separate evaluated-data run that refuses to run without qualified input libraries and defensible normalization.
- [ ] **C09.09** Ensure Stage51–57 stacked PRs and subsequent core work have a documented integration/merge sequence and no orphaned unreviewed physics code.
- [ ] **C09.10** Make a consolidated acceptance script run all real numerical gates, not just smoke tests, with per-gate PASS/FAIL and linked artifacts.
- [ ] **C09.11** Summarize changed physical assumptions between versions, expected changes in solution and non-regression scope.
- [x] **C09.12** Maintain **this single TODO file** as the canonical core-development checklist; update checkboxes with a commit/CI citation rather than creating redundant milestone reports.

**Exit:** another developer can reproduce the declared numerical core results and see exactly which physically validated claims are *not* supported.

---

## C10 — Separate physical verification/validation blockers (not solved by writing numerics alone)

These items belong in the core work plan but **must not be counted as implemented merely because all code checkboxes above are green**.

- [ ] **C10.01** Choose one explicit historical NASA NERVA core variant and document each geometry/material/power reference with exact source page/table, units and uncertainties.
- [ ] **C10.02** Audit whether the conceptual model's material loading, multiplication and transport/source behavior can represent that variant. Report discrepancy rather than tuning it invisibly.
- [ ] **C10.03** Qualify hot evaluated neutron cross-section temperature coverage for **all** active materials and any required thermal-scattering datasets.
- [ ] **C10.04** Supply traceable physical temperature-dependent fuel/ZrC properties and hydrogen EOS/transport tables, including temperature and pressure validity domains.
- [ ] **C10.05** Demonstrate transport criticality/source convergence/heating statistics and controlled neutron/photon energy deposition with real OpenMC data and uncertainty.
- [ ] **C10.06** Benchmark available core power profiles, criticality/reactivity, fuel/coolant temperatures and relevant pressure drops against documented *applicable* data.
- [ ] **C10.07** Separate reference uncertainty, model discrepancy, Monte Carlo sampling error and thermal/spatial/time-step numerical error.
- [ ] **C10.08** Reproduce independent benchmark results on a clean environment from pinned files/data and make unsupported historic operating claims fail closed.

**Exit:** a core model physically supported over a **declared** NERVA reference operating envelope; not full engine-cycle completion or hardware certification.

---

## Recommended execution order — no detours

1. **Stage58 / C01.01–C01.34:** couple Stage55 solid to Stage57 H2 *without double-counting energy*. Until this passes, avoid further unrelated subsystem expansion.
2. **C02 + C03:** make actual OpenMC heating and evaluated hot-neutron transport inputs trustworthy enough to supply power to Stage58.
3. **C04:** implement physically consistent neutronic–thermal feedback with true transport-data and uncertainty checks.
4. **C05 + C06:** improve fuel/coating interface fidelity and H2 EOS/BCs, driven by measured core-model sensitivity.
5. **C07 + C08:** improve source/kinetics interoperability, convergence and numerical acceptance.
6. **C09:** consolidate into one authoritative reproducible core pipeline.
7. **C10:** close separate physical/NASA validation before claiming a historically credible hot NERVA core.

## Core completion gates

The **implementation-only** core TODO is complete when all C01–C09 items required for the explicitly selected physical core operating envelope are implemented, checked and demonstrated by the canonical driver.

- [ ] **Gate G1 — Unified state:** Stage58 is one integrated solver, not two independently evolving hydrogen energies.
- [ ] **Gate G2 — Conservation:** mass, energy and channel-wall heat exchange balance over open/closed boundary cases.
- [ ] **Gate G3 — Physical source:** genuine OpenMC heating and explicitly justified physical normalization reach thermal volumes with a conserved budget.
- [ ] **Gate G4 — Hot data:** required evaluated neutron data and thermophysical tables cover the simulated operating state.
- [ ] **Gate G5 — Feedback:** temperature/density changes actually change transport and coolant pressure/flow, with robust convergence.
- [ ] **Gate G6 — Spatial/time convergence:** documented mesh and timestep trends for pressure, temperature, wall flux and deposited heating.
- [ ] **Gate G7 — Integrated execution:** one reproducible input executes the complete core physics path and writes auditable results.
- [ ] **Gate G8 — Honest evidence:** numerical verification, real Monte Carlo transport and historical validation are individually identified; none is inferred from the others.

**Outside this checklist:** full feed-cycle, pump/turbine shaft matching, nozzle performance and engine-level NASA comparisons. Those are part of the **separate overall NERVA** roadmap, not the estimated 30–40% outstanding core implementation.
