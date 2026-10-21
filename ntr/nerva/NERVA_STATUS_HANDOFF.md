# NERVA model — physics/numerics status and handoff

**Status date:** 2026-10-21  
**Repository:** [dubakoor13/NUKE](https://github.com/dubakoor13/NUKE)  
**Snapshot assessed:** `main` at `b45339058b1b846e2b12250f0747c562775f1911` (merged [PR #38](https://github.com/dubakoor13/NUKE/pull/38)).  
**Latest checked NERVA GitHub Actions:** [run 39564249337](https://github.com/dubakoor13/NUKE/actions/runs/39564249337) — **121 successful steps; zero failed**.

> **Assessment for the stated target — a physics-faithful, NASA-benchmarked NERVA engine replica:** approximately **30–40% complete; 60–70% pending**. This is a *subjective engineering planning range*, **not** measured percent complete, not a formal NASA certification level, and not a code-test coverage statistic. There is no accepted historical requirements/benchmark matrix from which an objective percent can yet be computed. The reusable numerical framework is more advanced than the historical-fidelity and integrated-cycle validation.

## 1. What the current verification actually establishes

The repository contains a substantial set of Python/NERVA reduced-order, CoolProp hydrogen, OpenMC-model/preprocessing, and transient thermal-hydraulic components. The current green CI proves that the **implemented equations and interfaces passed their coded regressions** on the exercised setups.

For the recently added transient tests, the nuclear deposition arrays and operating conditions are **synthetic inputs**. Local/global algebraic mass and energy closure does **not** by itself validate NASA tie-tube geometry, fluid-state/phase-change correlations, criticality, reactor performance, or experimentally observed thrust and specific impulse.

**Keep these four meanings separate:**

- **Implemented:** executable software and mathematical formulation exist.
- **Numerically verified:** designated synthetic/unit/regression conditions passed; conservation and convergence checks passed where explicitly exercised.
- **Transport validated:** an actual OpenMC neutron transport case has run with suitable nuclear cross-section data, thermal-power normalization, statistical diagnostics, and provenance. **Not established by the cited CI run.**
- **NASA historically validated:** the *specific* NERVA configuration reproduces relevant NASA data with documented geometric, material, reactor, thermal-hydraulic, pump/turbine, and nozzle conditions and uncertainty. **Not established.**

The repository has an OpenMC transport runner, `run_openmc_transport.py`, with cross-section preflight and optional executable invocation; that capability must not be confused with proof that historical model transport runs or NASA comparisons were completed.

## 2. Component-by-component assessment

| Subsystem | Current condition | Validation/fidelity gap |
|---|---|---|
| Steady NERVA nozzle/performance (`nerva_engine.py`, `engine_performance.py`) | Reduced-order choked-flow, thermodynamic and thrust/Isp paths implemented | Full real-hydrogen reactor-outlet chemistry, losses, operating-point calibration and measured-data comparison |
| LH2 feed, steady pump/turbine-bleed and parallel regenerative network (`lh2_feed_cycle.py`, `turbine_bleed_cycle.py`, `regenerative_cooling.py`) | Implemented; existing CoolProp/synthetic cycle regressions | True time-dependent pump maps/head, shaft inertia, turbine flow, nozzle/core mass demand and controller dynamics |
| Fuel-element and tie thermal/source mapping (`resolved_transient.py`, `thermal_flow_coupling.py`, `openmc_cycle_coupling.py`) | Resolved source partitions and reduced conjugate energy mapping implemented | Hardware-matched solid geometry/materials, detailed heat transfer, unified transient coupling and source provenance |
| Cryogenic tie H2 EOS and two-phase numerical models | Isobaric screening, rigid-cell internal-energy, pressure-driven and finite-inertance alternatives implemented | Phase slip, non-equilibrium boiling/flash, CHF, validity of heat-transfer and pressure-loss correlations |
| Distributed tie axial transport (`distributed_tie_euler.py`) | 1D homogeneous-equilibrium, CoolProp/Rusanov **mass, momentum and kinetic-inclusive total-energy** solver; synthetic conservation tests | Higher-fidelity wave/two-fluid physics, real tie dimensions, characteristic junctions and hardware verification |
| Hydraulic tie-to-engine boundaries (`distributed_tie_engine_boundary.py`) | Independent shaft-closed steady reservoir states supplied to Euler | A one-way boundary is not a fully converged dynamic engine |
| Tie + downstream header (`dynamic_header_tie.py`) | Shared mass/total-energy flux and evolving downstream EOS pressure/enthalpy; transient feedback tested | Real header geometry, junction momentum, transient engine discharge model |
| Live nozzle/reflector branch (`dynamic_parallel_header.py`) | Flow/enthalpy re-solved against current common-header pressure with existing hydraulic leg equations | The branch itself is quasi-steady and its wall heat is prescribed |
| Upstream + downstream header system (`dual_header_tie.py`) | Both H2 inventories and shared tie-Euler fluxes conservative; upstream EOS feeds both paths; synthetic CI passes | **Pump-delivered mass flow, pump stagnation enthalpy, and engine/core outlet demand are still fixed at the independently solved steady shaft-cycle reference** |
| OpenMC NERVA model/transport tools (`model.py`, `geometry.py`, `run_openmc_transport.py` and associated tools) | Model-generation and transport/preflight code exists | Demonstrate source-specific actual transport, cross-section/reaction coverage, mesh/tally convergence, eigenvalue/heating uncertainty, power coupling to thermal state |
| Historical NASA geometry, operating points, materials and qualification | References/presets and partial comparisons exist | Systematically extract, transcribe and trace model input and individual NASA test/analysis datasets for one defined NERVA variant |
| End-to-end reactor/engine uncertainty and experimental validation | **Not demonstrated** | Verified transport-to-thermal-to-cycle convergence and quantitative NASA benchmark agreement |

**Most important outstanding distinction:** the **two-manifold mass and energy inventory problem** has been implemented and numerically checked; **the changing pump/turbine/core/nozzle operating point has not been solved**. Neither the latest green CI nor the repository name implies a NASA-standard full engine.

## 3. Work still pending — prioritized

### P0 — close the engineering evidence gap

1. **Freeze an identifiable NASA reference configuration.** Specify the NERVA variant, relevant document revisions, geometry, fuel/moderator/tie tube layout, materials, hydrogen state points, reactor power and operating point. For every input, record source document/page/table/figure, units, conversion and uncertainty. Avoid mixing incompatible historical variants into one supposedly exact replica.
2. **Run actual OpenMC transport against documented nuclear data.** Record library name/version/temperature coverage, material temperature/density assumptions, geometry model hash, particles/batches, eigenvalue statistics, mesh and reaction tallies, heat normalization, repeated seeds and source/statepoint provenance. Distinguish transport executable checks from surrogate array tests.
3. **Couple transport power/thermal field with real convergence tests.** Carry component-wise deposited heat (fuel solid, coolant, tie solid, each duct, unresolved power) to consistent transient/steady thermal calculations; audit all sinks and structural storage. Iteratively update temperatures/densities/neutronic power where supported, with tolerances for *both* power and physical state.

### P1 — finish transient fluid-cycle physics and numerical robustness

4. **Replace prescribed pump delivery with an operating-point-dependent model.** Introduce head versus flow/speed and efficiency/shaft-power relations tied to appropriate historical pump data; include physical manifold pressure, inlet/feed state, cavitation/NPSH and operating-domain failure conditions. The pump mass and energy supplied to the new header must come from the same pump equations, not a frozen reference.
5. **Add shaft and turbine bleed dynamics.** Couple turbine pressure ratio, turbine work, rotor inertia, pump torque and mass-flow split. Enforce shaft power/torque and flow residuals and examine transient convergence/rejection. Do not conflate compressor/pump hydraulic energy with nozzle heat or reactor deposited power.
6. **Close changing core/nozzle discharge demand.** Connect downstream header enthalpy/pressure and H2 mass to actual reactor-channel pressure/temperature and nozzle throat/choking; calculate compatible turbine/bleed paths. Solve the full closed-loop operating point with robust damping/newtonization and conservative intercomponent fluxes.
7. **Extend geometric and thermophysical fidelity.** Implement credible 1D/2D channel/tie dimensions and wall/solid properties; calibrate convection, pressure loss, two-phase flashing, boiling and CHF limitations against relevant evidence. Verify convergence with axial/radial refinement, acoustic CFL/time-step study and property derivative sensitivity.

### P2 — NASA model credibility and release validation

8. **Establish a historical comparison matrix:** keff/reactivity and spatial power (where NASA data allow), reactor/chamber pressure and temperature, LH2/turbopump flows, branch pressure drops, thrust, nozzle conditions and Isp. Include uncertainties, numerical error and documented model deficiencies. Never label a nominal preset's output an independent historical validation.
9. **Perform end-to-end verification:** energy/mass inventory across every actual boundary; wave/flow reversal; low-flow and startup/shutdown cases; hard invalid EOS/phase conditions; pump/turbine and chamber/nozzle solver convergence; transport and thermal resolution studies; physical regression tolerances independent of the code being tested.
10. **Document reproducibility and acceptance:** reference inputs, data provenance, code SHA, nuclear-data hashes, exact invocation, result artifacts and comparison plots. Keep synthetic CI separate from expensive externally configured OpenMC benchmark runs.

## 4. Acceptance gates — do not claim full completion until all apply

- [ ] **Reference traceability:** one unambiguous NASA NERVA baseline with source-linked geometry, mass inventories, operating points, materials and uncertainties.
- [ ] **Neutronics evidence:** reproducible *actual* OpenMC transport with appropriate cross sections, explicit statistics, spatial heating and criticality checks where relevant.
- [ ] **Thermal coupling:** real reactor deposition drives verified fuel/tie/coolant models with nonduplicated storage, loss and unresolved heat terms.
- [ ] **Hydraulic closure:** both regenerative branches and headers, *pump-delivered flow*, pressure losses and phase-domain states are simultaneously compatible—not just individually conservative.
- [ ] **Shaft/flow closure:** turbine, pump, rotor energy, reactor channels and nozzle throat all satisfy the same transient or converged steady operating point.
- [ ] **Numerical independence:** grid/time-step sensitivity and independent analytical or external cross-checks; not solely conservation residuals of the implemented discrete equations.
- [ ] **Historical validation:** quantified comparisons to applicable NASA experimental/model data for the selected configuration with credible error bands.
- [ ] **Reproducible handoff:** a fresh developer can run the specified baseline and regenerate plots/reports with pinned code and data.

**Working definition of “100%”: all gates above demonstrated for a specifically declared historical reference and stated operating envelope.** It does not mean full certification of flight hardware.

## 5. Latest implementation evidence and reproducible checks

**Snapshot:** [PR #38 — dual dynamic H2 manifolds](https://github.com/dubakoor13/NUKE/pull/38) merged at commit `b45339058b1b846e2b12250f0747c562775f1911`.

Recent incremental milestones:

| PR | Addition |
|---|---|
| [#30](https://github.com/dubakoor13/NUKE/pull/30) | Rigid-volume cryogenic hydrogen **internal-energy** and mass/EOS-pressure conservation |
| [#31](https://github.com/dubakoor13/NUKE/pull/31) | Pressure-driven tie finite-volume mass and internal-energy flux |
| [#32](https://github.com/dubakoor13/NUKE/pull/32) | Finite interface inertance/momentum screening |
| [#33](https://github.com/dubakoor13/NUKE/pull/33) | Distributed 1D hydrogen **mass, momentum and total-energy** Euler model |
| [#34](https://github.com/dubakoor13/NUKE/pull/34) | Shaft-closed engine steady boundary states provided to distributed tie solver |
| [#35](https://github.com/dubakoor13/NUKE/pull/35) | Correction to unphysical hot-header ghost outflow numerical diffusion |
| [#36](https://github.com/dubakoor13/NUKE/pull/36) | Dynamic EOS common header and bidirectional tie/header feedback |
| [#37](https://github.com/dubakoor13/NUKE/pull/37) | Dynamic pressure-conditioned nozzle/reflector coolant branch |
| [#38](https://github.com/dubakoor13/NUKE/pull/38) | Conservative upstream pump manifold and dual-header EOS feedback |

Selected existing validation commands (in an environment with installed repository dependencies and CoolProp):

```bash
python -m ntr.nerva.validate_distributed_tie_euler
python -m ntr.nerva.validate_distributed_tie_engine_boundary
python -m ntr.nerva.validate_dynamic_header_tie
python -m ntr.nerva.validate_dynamic_parallel_header
python -m ntr.nerva.validate_dual_header_tie
python -m ntr.nerva.validate_regenerative_cooling
python -m ntr.nerva.validate_turbine_bleed_cycle
python -m ntr.nerva.validate_openmc_cycle_coupling
```

Public analysis entry points (inspect arguments before use):

```bash
python -m ntr.nerva.run_dynamic_header_tie --help
python -m ntr.nerva.run_distributed_tie_euler --help
python -m ntr.nerva.run_openmc_transport --help
```

For the **dual-manifold** synthetic calculation, `run_dynamic_header_tie.py` requires both `--dynamic-nozzle-branch` and `--upstream-manifold-volume-m3`, plus the downstream `--header-volume-m3` and referenced engine/power input files. The OpenMC transport runner must be supplied a compatible nuclear-data library and an actual executable environment; invoking help/export is not a completed transport run.

**Last observed CI:** [NERVA model check, 2026-10-21, run 39564249337](https://github.com/dubakoor13/NUKE/actions/runs/39564249337): 121 steps succeeded, zero failed. It was run on the #38 code. This status note describes that snapshot; later changes must be reassessed.

## 6. Recommended next iteration — one clearly bounded physics milestone

**Target:** close the *dynamic pump-discharge boundary* rather than adding further independently passing synthetic header variants.

1. Build a separately verified pump mass-flow/head/efficiency or shaft-coupled map with explicit valid operating domain and provenance; do not invent NASA pump curves or imply historical calibration.
2. Replace `m_pump_ref` and fixed `h_pump_ref` in `dual_header_tie.py` with the solved pump inlet/outlet enthalpy and mass flux, preserving the present shared face balances and rejecting bad states. Introduce shaft work/torque accounting.
3. Add a case with changing downstream header pressure that measurably alters pump flow *and* the pressure-balanced nozzle/reflector path. Assert combined transient H2 inventories, pump hydraulic-energy/shaft power and reactor/nozzle demand compatibility.
4. Record both numerical convergence and physical assumptions. Keep the model flagged **partial** until pump/turbine/core/nozzle demand is solved consistently.
5. In parallel, identify the first selected NASA baseline and real nuclear-data configuration, since otherwise numerical progress cannot justify materially reducing the **60–70% historical-fidelity gap**.

## 7. References, ownership and interpretation

- **Primary user-supplied NASA historical document:** [NASA NTRS 19680007802](https://ntrs.nasa.gov/api/citations/19680007802/downloads/19680007802.pdf). **A detailed source-to-model parameter comparison remains pending.**
- Additional background NASA records already referenced in [NERVA_ENGINE.md](NERVA_ENGINE.md). Verify variant applicability before treating values as benchmark truth.
- Read the [physics architecture and solver limitations](NERVA_ENGINE.md) before porting or changing one of the alternative hydrogen models.
- [GitHub workflow source](../../.github/workflows/nerva-model-check.yml) and [most recent validated run](https://github.com/dubakoor13/NUKE/actions/runs/39564249337).

**Handoff rule:** report (a) code written, (b) tests actually executed, (c) actual OpenMC transport executed or not, (d) physical NASA verification achieved or not, and (e) which part of the remaining work is engineering model physics versus implementation/infrastructure. Do not automatically decrease the completion percentage after a green synthetic CI run.
