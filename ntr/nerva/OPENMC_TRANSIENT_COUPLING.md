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


## Channel-resolved transient fuel and tie-tube solver

The optional resolved solver now consumes **existing** power reconstruction
NPZs from two standard workflows:

- fuel: \`solve_element_channels\` output
  \`element_channel_power_reconstruction.npz\`, preserving each of the 19
  fuel coolant-channel axial solid and direct-H2 nuclear heating arrays
  plus each branch's flow;
- tie tube: \`solve_tie_instances\` output
  \`tie_instance_reconstruction.npz\`, preserving each tie's axial solid,
  supply direct-H2 and return direct-H2 nuclear heating arrays plus flow;
- reference: normalized OpenMC \`nerva_mesh_fields.npz\`, against which
  **every independent reconstructed energy partition is verified**. Total
  deposition not represented by these branches remains in an explicit
  unresolved-energy reservoir, instead of artificially heating fuel/H2.

The implementation is in \`resolved_transient.py\`. Every fuel channel is
integrated independently with the conservative NASA-7-H2 variable-enthalpy
backward Euler solver. Each tie-tube receives a **shared solid thermal
node per axial bin**, exchanging energy with BOTH H2 passages. The supply
flows from z=0 to z=L, and the return flows from z=L back to z=0; the
far-end supply outlet is the return inlet. Tie coupling iterates to a
specified temperature tolerance and aborts if it fails to converge.
The global first-law check combines all fuel/tie solids, gas storage,
fluid inlet and outlet enthalpy, direct nuclear H2 heat, and unresolved
heat.

The resulting model now has **distinct repeated elements, 19 individual
coolant channels per element, per-tie supply and return paths, and
opposing hydrogen flow directions**. It is still an assumed-UA,
assumed-capacity, fixed-flow/pressure model. It has no transient pump
network, channel pressure loss, 3-D thermal conduction or time-dependent
neutronics. The heating maps are frozen over time: the earlier
\`run_openmc_transient\` option with quasi-static OpenMC temperature
feedback is a **separate aggregate-branch solver**, not yet connected
to this resolved solver. Do not claim this is a fully coupled
statepoint-rerun resolved-feedback NERVA simulation.

**Run on a real OpenMC reconstruction** (obtained by the full OpenMC
pipeline and accompanied by the transport provenance file):

\`\`\`bash
python -m ntr.nerva.run_resolved_transient \
  build/nerva_power/nerva_mesh_fields.npz \
  build/nerva_element_channels/element_channel_power_reconstruction.npz \
  build/nerva_tie_instances/tie_instance_reconstruction.npz \
  --statepoint PATH/TO/statepoint.h5 \
  --transport-provenance PATH/TO/transport_provenance.json \
  --steps 10 --dt-s 0.1 --initial-solid-k 500 \
  --initial-gas-k 300 --inlet-temperature-k 300 \
  --fuel-solid-capacity-j-k 5000 --fuel-gas-capacity-j-k 500 \
  --fuel-ua-w-k 100 \
  --tie-solid-capacity-j-k 3000 \
  --tie-supply-capacity-j-k 600 \
  --tie-return-capacity-j-k 700 \
  --tie-supply-ua-w-k 80 --tie-return-ua-w-k 90
\`\`\`

All capacities (J/K per thermal node), UA (W/K per node), temperatures,
flows and power above are **illustrative user inputs**, not NASA-derived
engine conditions. The real-data flag validates the supplied statepoint's
SHA-256 against transport provenance; it does **not** cryptographically
link arbitrary reconstruction NPZ files to that statepoint. The latter
require their generation in the same analysis run, and the independent
mesh/reconstruction power checks catch mismatches in total energy.

Synthetic numerical test and CLI smoke run:

\`\`\`bash
python -m ntr.nerva.validate_resolved_transient
python -m ntr.nerva.validate_resolved_transient --write-fixture build/resolved_fixture
python -m ntr.nerva.run_resolved_transient \
  build/resolved_fixture/mesh_fields.npz \
  build/resolved_fixture/fuel_reconstruction.npz \
  build/resolved_fixture/tie_reconstruction.npz \
  --allow-synthetic --steps 3 --dt-s 0.1 \
  --initial-solid-k 500 --initial-gas-k 300 --inlet-temperature-k 300 \
  --fuel-solid-capacity-j-k 5000 --fuel-gas-capacity-j-k 500 \
  --fuel-ua-w-k 100 --tie-solid-capacity-j-k 3000 \
  --tie-supply-capacity-j-k 600 --tie-return-capacity-j-k 700 \
  --tie-supply-ua-w-k 80 --tie-return-ua-w-k 90
\`\`\`

Outputs: \`resolved_transient_summary.json\`,
\`resolved_transient_history.csv\`,
\`resolved_transient_final_state.npz\`. Synthetic output is explicitly
labeled unverified and cannot substitute for OpenMC validation.
