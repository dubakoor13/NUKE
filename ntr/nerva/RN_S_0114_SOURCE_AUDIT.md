# RN-S-0114 — NASA 1964 Transient Engine Simulation Source Audit

**Source identity:** H. Hutchison, W. Madsen, R. Steiger, *Transient engine
simulation program. NERVA program*, Aerojet-General, August 1964,
RN-S-0114 / TID-23639 / NASA-CR-91272, NTRS **19680007802**.
NTRS record: https://ntrs.nasa.gov/citations/19680007802
NASA PDF: https://ntrs.nasa.gov/api/citations/19680007802/downloads/19680007802.pdf

## Provenance warning — document has NOT been fully inspected

On 2026-10-20 the NASA-hosted PDF returned HTTP 403 to the retrieval
environment. Only the bibliographic record and independently available
summary of RN-S-0114 from RN-S-0150 were accessible. This is **not** a
transcription of RN-S-0114 equations or IBM 7094 software.

Corroborating source: Aerojet RN-S-0150, publicly indexed at
https://www.osti.gov/servlets/purl/4181881/
Section B ("Digital Computer Model") identifies RN-S-0114 and describes
the computational structure, but DOES NOT give all implementation details.

Additional, DISTINCT successor publication (not the same as RN-S-0114):
R. A. Wells, *Transient-engine-simulation program (Fortran IV)*,
Aerojet-General RN-S-0432 (1967), DOI: 10.2172/4257009,
https://digital.library.unt.edu/ark:/67531/metadc1024713/
Do not confuse the 1967 FORTRAN IV program with the 1964 IBM 7094 program.

## Original model architecture reported by RN-S-0150 (secondary evidence)

- Explicit time-marching finite-difference engine model.
- Components include turbopump, hot-bleed port, reflector, shield,
  reactor core, nozzle, tanks and connecting lines.
- Thermal and fluid pressure states evolve throughout the engine.
- Turbopump model includes rotational energy storage.
- One-fluid, single-channel heat-exchanger treatment with an effective
  void-fraction representation in reflector, shield, and core.
- Reported node counts: 10 nozzle, 27 reflector, 14 shield, 27 core.
  These are descriptions in **RN-S-0150**; do not claim they are exact
  original array sizes in the inaccessible RN-S-0114 without checking.
- Nozzle coolant and bleed port treated as counter-current heat exchangers.
- Bleed mixing/losses and pump discharge-pressure closure require
  coupled nonlinear algebra; the predictor used by RN-S-0150 is noted.

## NUKE implementation gap matrix

| Historical model area | NUKE present status | Exactness |
|---|---|---|
| Coupled transient solid/fluid energy | \`heritage_transient.py\` verification kernel | Reconstructed generic conservation equations only |
| Distributed finite-difference nodes | Arbitrary serial lumped volumes | NOT historical nodalization |
| Shaft rotational inertia | Energy-conserving finite-difference update | Generic equation only, NO actual pump/turbine maps |
| Hydrogen thermo-chemistry | Official CEA rocket solver, CoolProp optional | Partially verified steady-state |
| Core/reflector/shield heat-source evolution | OpenMC tally/thermal outputs exist | NOT transiently coupled to this kernel |
| Hot-bleed / turbine / pump | Static stream accounting exists | NOT transiently solved |
| Engine pipeline pressures | Screening pressure-loss routines elsewhere | NOT dynamic, nonlinear full-system closure |
| Reactor kinetics/control drum transient | Not implemented in kernel | NOT historical |
| Historical input decks and curves | Not extracted | NOT benchmarked |

## New verification kernel equations

For gas cells connected in flow order, with separate solid nodes:

\`\`\`text
Csolid_i * dTsolid_i/dt = Qdeposition_i - UA_i*(Tsolid_i - Tgas_i)
Cgas_i   * dTgas_i/dt   = UA_i*(Tsolid_i - Tgas_i)
                          + mdot*cp*(Tupstream_i - Tgas_i)
\`\`\`

Each explicit Euler step is checked against

\`\`\`text
delta_E_stored = dt*(sum(Qdeposition) + mdot*cp*(Tin - Tout))
\`\`\`

with all fluxes evaluated at the OLD state and a calculated timestep
positivity bound. Shaft stored energy uses

\`\`\`text
Eshaft = 0.5*I*omega^2
delta_Eshaft = dt*(P_turbine - P_pump - P_losses)
\`\`\`

The energy form prevents a numerical division-by-zero at startup.
These are standard first-principles equations; they have **not** been
verified to be the particular discretization used in RN-S-0114.

## Next *unverified* source-extraction gates

1. Obtain and page-inspect RN-S-0114, including appendices and numerical
   sample runs, then preserve its exact equations and units.
2. Obtain and compare RN-S-0432 FORTRAN IV listing and 1964 IBM 7094
   program; identify revisions rather than merging them blindly.
3. Extract source tables, pump/turbine maps, pressure loss/heat transfer
   correlations, actual node arrangement and timestep specification.
4. Validate component-by-component using referenced NRX/CFDTS results.
5. Implement pressure/momentum dynamics and couplings with
   NASA CEA/OpenMC/CoolProp, and publish toleranced regression fixtures.

**Status:** source architecture identified; thermal/shaft conservation tests
implemented; NASA-equivalent transient NERVA program still pending.
