# NERVA OpenMC → transient thermal feedback integration

## Implemented source and solver path

\`postprocess_statepoint\` turns a **real** OpenMC eigenvalue statepoint into
\`nerva_mesh_fields.npz\` using heating-local eV/source normalized to an
EXTERNALLY PRESCRIBED reactor thermal power. \`openmc_transient.py\` integrates
the 3-D mesh into four non-overlapping axial power arrays:

1. fuel solid;
2. fuel coolant direct nuclear heating;
3. tie-tube solid;
4. tie-tube hydrogen direct nuclear heating.

Remaining deposited energy (reflector, shield, vessel, tally partition gaps)
goes into an explicit **unresolved energy reservoir**, never silently added
to coolant or discarded. The reactor *global* thermal ledger closes through
\`stored + unresolved + outlet fluid = deposited + inlet fluid\`.

Two axial aggregate solid/gas branches use the conservative
\`heritage_transient\` timestepper, with separate direct gas heating and
explicit positivity timestep checks. Fuel and tie hydrogen are two idealized
co-current branches — this is **NOT** a historically accurate NESS tie-tube
counterflow or parallel per-channel network. Inputs must supply heat
capacities, UA, cp and mass flows; they are NOT inferred from reactor power.

A fixed-point interface runs a fresh OpenMC calculation at iterated
**uniform** fuel-solid and coolant-gas material temperatures. It resets each
thermal trial to the same old state. For real data it uses the existing
\`run_openmc_transport\` with its nuclear-data checks and provenance. A
synthetic NPZ is never admissible to the feedback interface.

## Explicit limitations

- OpenMC computes **quasi-static spatial transport**, NOT time-dependent
  neutron kinetics, delayed neutrons, reactor control dynamics or a real
  time-dependent reactor-power solution.
- Reactor thermal power is user imposed (MW). It is **not** calculated from
  reactor reactivity or an uncontrolled transient.
- Uniform OpenMC material temperature/density proxies cannot represent
  channel-wise axial feedback, fuel conduction, reflector temperatures, or
  realistic gas property variations.
- Energy conservation verifies numerical accounting, not historical
  NERVA predictions. The OpenMC reactor remains a surrogate.
- A real neutron-transport executable and compatible cross-section library
  are required for real feedback runs; CI tests deliberately use synthetic
  spatial fields or an explicitly tagged test callback and cannot claim
  real transport validation.
- No burnup, depletion, fissile-load optimization, or control-drum search.

See the run CLI documentation for configuration and QA requirements.


## Run commands

Frozen normalized OpenMC data (no feedback):

\`\`\`bash
python -m ntr.nerva.run_openmc_transient --statepoint PATH/TO/statepoint.h5 \
  --power-mw 10 --steps 10 --dt-s 0.01 \
  --inlet-temperature-k 500 --initial-solid-k 650 --initial-gas-k 500 \
  --fuel-mass-flow-kg-s 2 --tie-mass-flow-kg-s 0.4 \
  --fuel-solid-capacity-j-k 1000000 --fuel-gas-capacity-j-k 200000 \
  --tie-solid-capacity-j-k 400000 --tie-gas-capacity-j-k 100000 \
  --fuel-ua-w-k 100000 --tie-ua-w-k 20000
\`\`\`

Bidirectional, **quasi-static** temperature-feedback to freshly rerun OpenMC:

\`\`\`bash
python -m ntr.nerva.run_openmc_transient \
  --cross-sections /PATH/TO/cross_sections.xml \
  --power-mw 10 --steps 10 --dt-s 0.01 \
  --inlet-temperature-k 500 --initial-solid-k 650 --initial-gas-k 500 \
  --inlet-pressure-mpa 5 \
  --fuel-mass-flow-kg-s 2 --tie-mass-flow-kg-s 0.4 \
  --fuel-solid-capacity-j-k 1000000 --fuel-gas-capacity-j-k 200000 \
  --tie-solid-capacity-j-k 400000 --tie-gas-capacity-j-k 100000 \
  --fuel-ua-w-k 100000 --tie-ua-w-k 20000 \
  --feedback-max-iterations 4 \
  --feedback-temperature-tolerance-k 0.5
\`\`\`

Capacity, UA and flow inputs above are illustrative **user assumptions**,
not recovered NERVA parameters. For real studies substitute defensible
material masses, specific heat, heat-transfer coefficients and flow rate.

At every macro timestep the feedback driver calls real OpenMC transport at
updated uniform fuel and hydrogen temperatures and an ideal-gas H2 density.
The new tally normalization is used for the next thermal trial, always
starting from the same OLD thermal state. A failure of XS temperature-range
audit, negative heating, double-counted power, large unmodeled deposition,
unstable timestep or fixed-point convergence aborts the run. The code
records the input nuclear-data hash, statepoint hashes, iteration history,
source type and stepwise conservation residuals.

For a synthetic CI-only demonstration, replace \`--statepoint\` with
\`--fields build/nerva_synthetic/nerva_mesh_fields.npz --allow-synthetic\`.
This mode *never* calls OpenMC and is tagged \`synthetic_unverified\`.

Outputs: \`openmc_transient_report.json\` and
\`openmc_transient_history.csv\` under the configured output root.


## Variable-enthalpy and implicit transient integration (current)

The time integrator now supports:

\`\`\`bash
--integrator implicit --gas-enthalpy-model nasa7-h2
\`\`\`

This is the **default for the transient runner**. The legacy
\`--integrator explicit --gas-enthalpy-model constant\` remains available
for regression comparisons.

The implicit solver updates each axial gas cell using a monotonic,
bounded nonlinear solve of the mass-specific hydrogen **enthalpy**
equation. It uses the existing pure-molecular (frozen H2) GRI-Mech
NASA-7 polynomial for sensible enthalpy; pressure is assumed fixed in
the thermodynamic equation. Variable cp(T) is accounted for through
\`h(Tnew)-h(Told)\`, and upstream advection uses **new-time enthalpy**.

\`\`\`text
C_s (T_s,new - T_s,old) / dt
     = Q_s - UA*(T_s,new - T_g,new)

m_g (h(T_g,new) - h(T_g,old)) / dt
     = UA*(T_s,new - T_g,new) + Q_direct_gas
       + mdot*(h(T_up,new)-h(T_g,new))

m_g = C_g,reference / cp_reference
\`\`\`

The gas-mass inventory is held fixed over a timestep and inferred from
a **supplied effective reference gas heat capacity**. The solid uses
a supplied effective constant heat capacity. Both capacities are
per axial cell, not per engine.

The algorithm is backward Euler: unconditionally stable for this
co-current passive heat-exchange system within its single-phase
assumptions; it can still be **temporally inaccurate for large steps**
and rejects expansion beyond the supported NASA-7 200–3500 K domain.
It does not introduce physically correct pump-driven pressure changes,
material-dependent solid properties, equilibrium dissociation kinetics,
or genuine transient neutronics. OpenMC feedback still uses an
axially-uniform material-temperature proxy and externally supplied
reactor power.

Validation command:

\`\`\`bash
python -m ntr.nerva.validate_implicit_hydrogen
\`\`\`

The checks include energy conservation, exact zero-flow heat balance,
a timestep far larger than the explicit stability threshold,
causal upwind enthalpy transport, nonconstant cp, failure outside
the thermo range, and an iterative feedback accounting regression.
