# NESS / ELES / NUKE — source-grounded physics and numerics audit (2026-10-24)

**Scope:** Historical lineage, verified program overlap, and interface/validation gaps. This is **not** a port or an OpenMC/NASA historical qualification.

## Sources and provenance
- NASA CR-191080, *NESS Vol. 1* (1993): https://ntrs.nasa.gov/citations/19930014687
- NASA CR-191081, *NESS Version 2.0 User's Guide* (1993): https://ntrs.nasa.gov/citations/19930014686
- NASA 2008 NESS upgrades: https://ntrs.nasa.gov/api/citations/20080033115/downloads/20080033115.pdf
- ELES reconstruction: https://github.com/dubakoor13/ELES

**Important evidence limit:** GitHub lists `NUKE/main/nucke engine simul.pdf` and `NUKE/main/aaa neuronic code.pdf`, but their binary contents were not readable with the current GitHub connection. This audit uses the independent NASA-hosted manuals above; it **does not certify the contents of the uploaded PDFs**.

## What NASA actually built
NESS extended an NTP variant of the Expanded Liquid Engine Simulation (ELES) with Westinghouse ENABLER-I/II reactor system models. NASA CR-191081 Section 2.1 documents the loop: estimate nozzle performance and mass flow; initialize reactor including nozzle heat-load estimate; compute reactor coolant, thermal, cycle and pressure schedules; solve turbopump/turbine and performance; repeat for consistency. The historical preliminary-model initial nozzle heat load is 1% of reactor power and a 10% heat-load mismatch triggers recomputation. These are historical heuristics, **not** modern acceptance criteria.

**Critical:** NASA CR-191081 Section 3.3 states NESS **does not perform neutron transport analysis**. Peripheral nuclear heating is specified as a fraction of reactor power. OpenMC's real neutron/photon transport and qualified, uncertainty-aware power tallies should remain authoritative in NUKE.

## Exact reuse limits

| Domain | ELES Python | Existing NUKE | NESS-specific work still pending |
| --- | --- | --- | --- |
| Conventional engine sizing | `python/eles/integrated.py`, `performance.py`, `program.py` | `ntr/nerva/nerva_engine.py`, `nasa_cea_rocket.py` | **Do not use** original dual-propellant `PerformanceInputs` unchanged for hydrogen-only nuclear heating. Use the nuclear hydrogen/nozzle pipeline. |
| Pump and turbine sizing | `python/eles/turbopump.py`, `turbine.py` | `turbine_bleed_cycle.py`, `dynamic_pump_shaft.py` | NESS axial pump, multi-leg redundant pump-out operation, true off-design maps, support/reflector turbine routing, shaft dynamics and stage constraints remain unqualified. |
| Cycle topology | `python/eles/schedule.py` | `engine_stream_balance.py`, `dual_header_tie.py` | NTP cold H2, tie, reflector, core, turbine and nozzle routing needs a separate physically conservative graph; the ELES generic fuel/oxidizer graph is not an NTR graph. |
| Core power and temperature | no reactor transport | `model.py`, `thermal_flow_coupling.py`, `openmc_cycle_coupling.py` | Replace NESS prescribed cosine/peak-channel screening with resolved OpenMC heat deposition only after proving source normalization, material allocation, uncertainty and mesh convergence. |
| Hydrogen properties | `python/eles/providers.py` requires supplied property backends | `hydrogen_properties.py`, `nasa_hydrogen.py`, `lh2_feed_cycle.py` | Maintain EOS validity and a **single enthalpy reference** across CoolProp, hot NASA7 H2 and stage boundaries; historical NESS had a documented near-saturation interpolation problem addressed in a later upgrade. |
| Benchmarks | ELES technical manual reconstruction (historical Programmer's Manual still absent) | historical presets, synthetic tests | Compare against independent NASA CR-191081 Section 4 eight-case tables, with source units, genuine model inputs and no fitted observations masquerading as validation. |

## NASA-specific core detail
NESS Section 3.3 models the peak channel with a prescribed axial power shape, a coolant/wall heat-transfer calculation, gas property dependent heat-transfer and pressure-loss correlations, and a peak fuel-temperature approximation. It determines a reactor coolant pressure drop and then returns to engine-cycle closure. Because that code **lacks neutronics**, agreement of one such screening calculation with OpenMC heating is useful as an independent numerical comparison but cannot establish neutron-transport fidelity.

## First safe implementation
The accompanying `ness_system_balance.py` is a **control-volume audit**, not a new reactor model. It takes mass flow and **reference-coherent** enthalpies at named ports, prescribed external heat/shaft power and storage rate and checks:

`sum(m_in) - sum(m_out) = 0`

`sum(m_in h_in) - sum(m_out h_out) + Q_fluid + W_shaft_to_fluid - dU/dt = 0`

It also converts ELES quantities (lbm/s, psi, Rankine, Btu/lbm, hp) into SI and reports energy/mass residuals. `validate_ness_system_balance.py` exercises eight independent synthetic and adversarial checks. This only verifies conservation **of supplied data** and is not NASA historical validation.

## Physics-first next gates
1. Inspect the two checked-in PDFs as PDF pages, identify exact titles/versions, compare listing coverage and isolate source equations/data.
2. Freeze one historically consistent NERVA/ENABLER configuration (never conflate 1972 design, XE test and ENABLER-II scaled fuel).
3. Define cold-feed, tie/reflector, core, bleed, shaft and nozzle station contracts and common EOS/enthalpy zero.
4. Apply verified, neutron-data-qualified OpenMC deposition to separate solid and fluid domains; track unallocated heat, axial shape and statistical errors.
5. Solve non-linear pump/core/nozzle mass-flow, pressure and shaft-energy closure with both branch flows and valid thermodynamic operating ranges.
6. Compare NASA sample-deck outputs, then refine mesh/timestep, cross sections, Monte Carlo seeds and hydraulic maps. Do **not** equate a green unit test with historical physics validation.
