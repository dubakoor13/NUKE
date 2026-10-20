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


## Channel-resolved quasi-static OpenMC re-transport feedback (new)

The \`resolved_openmc_feedback.py\` time integrator joins the previously
separate capabilities:

1. **New OpenMC transport snapshot for each feedback iterate**, with
   uniform fuel and H2 temperature and a *fixed-pressure ideal-gas*
   hydrogen density proxy. Fuel elemental composition/geometry remain
   unchanged.
2. \`run_analysis --no-plots\` on that exact statepoint, with the
   repository's existing \`heating-local\` normalization, repeated-cell
   element/channel heating reconstruction and per-tie counterflow
   heating reconstruction.
3. Re-check all material power partitions against the normalized mesh,
   preserve direct hydrogen nuclear heating distinct from solid heat,
   and retain nonmodeled deposition in an energy reservoir.
4. Advance individually resolved 19-channel fuel elements and both
   opposite-flow tie passages with implicit pure-H2 enthalpy conservation.
5. Repeat **from the same pre-timestep thermal state** until both the
   uniform OpenMC material-state reduction and all resolved solid/gas
   temperature profiles satisfy the user-specified Picard tolerance.
6. Refuse stale statepoints, changing flow/topology, changed nuclear-data
   library hashes, missing QA, excess unmodeled power and nonconvergence.

**This implementation is a real runnable orchestration path**, but was
validated in CI using synthetic providers; the pipeline has not yet been
executed with an installed OpenMC transport binary, compatible HDF5
neutron data and historical NERVA test datasets in CI.

### Energy-consistent uniform coolant temperature feedback

The channel-resolved thermal solver stores the gas inventory as
`m_H2 = Cgas/cp_reference` and advances `m_H2 * h(T)` with NASA-7
frozen molecular H2 enthalpy, rather than advancing `Cgas * T`.
When reducing the many fuel-channel and tie-passage temperatures to
the single uniform OpenMC H2 material-temperature surrogate,
`reduce_openmc_material_state` now conserves that modeled hydrogen
enthalpy:

```text
sum_i m_i * h_NASA7(T_i) = (sum_i m_i) * h_NASA7(T_uniform)
```

A monotone bounded inversion over 200–3500 K determines `T_uniform`.
The constant-cp mode retains the equivalent gas-mass-weighted arithmetic
temperature average. Invalid NASA-7 temperatures fail closed.
This eliminates a temperature-projection *energy mismatch*; it does NOT
change the imposed reactor power, solve coolant density/momentum feedback,
or provide spatially resolved OpenMC temperatures. The current transport
still uses one uniform H2 density proxy based on a prescribed pressure.

Verification: `python -m ntr.nerva.validate_resolved_openmc_feedback`
checks NASA-7 enthalpy preservation for nonuniform channels, constant-cp
behavior, invariance of a uniform state, and thermo-domain rejection.

### Run a real-data feedback study

\`\`\`bash
python -m ntr.nerva.run_resolved_openmc_feedback \
  --cross-sections /PATH/TO/cross_sections.xml \
  --power-mw 1 --steps 2 --dt-s 0.1 \
  --assembly reactor --rings 5 --particles 4000 \
  --batches 80 --inactive 20 \
  --inlet-pressure-mpa 5 \
  --initial-fuel-solid-k 650 --initial-hydrogen-k 500 \
  --inlet-temperature-k 500 \
  --fuel-mass-flow-kg-s 2 --tie-mass-flow-kg-s 0.4 \
  --fuel-solid-capacity-j-k 5000 \
  --fuel-h2-capacity-j-k 500 --fuel-ua-w-k 100 \
  --tie-solid-capacity-j-k 3000 \
  --tie-supply-h2-capacity-j-k 600 \
  --tie-return-h2-capacity-j-k 700 \
  --tie-supply-ua-w-k 80 --tie-return-ua-w-k 90 \
  --feedback-max-iterations 5 --feedback-tol-k 0.5
\`\`\`

**All thermal capacities, UA and flow values in this example are
illustrative assumptions, not recovered NASA NERVA parameters.** The
\`--power-mw\` is EXTERNALLY IMPOSED, not a computed neutronic power
transient. Real OpenMC execution requires matching temperature-dependent
nuclear data with valid material temperatures and adequate statistical
uncertainty; the driver aborts on XS coverage/transport/QA failure.

Each snapshot is saved alongside the NERVA OpenMC transport provenance
(including statepoint SHA-256 and neutron-data SHA-256), the analyzed mesh
power fields, and individual channel/tie reconstructed powers. Final
summary and checkpoint artifacts are placed under the output root.

### Remaining physics and validation gaps

- Spatial OpenMC material state is still **uniform per material** (one
  fuel-solid and one coolant H2 temperature plus uniform hydrogen density);
  it is NOT an element-/channel-/axial-material-instance update.
- OpenMC eigenvalue transport produces a power **shape** at the imposed
  reactor power. It does not supply point kinetics, delayed-neutron
  response, reactor control or a transient reactor power solution.
- Turbopump and pressure-network dynamics, mass inventory, thermal
  expansion, variable UA and solid thermophysical parameters remain
  prescribed or absent.
- Monte Carlo statistical noise can prevent Picard convergence: the
  driver fails instead of silently accepting a thermal step.
- Physical benchmark to RN-S-0114, XE-Prime, NESS and NERVA source
  measurements requires source data, real runs and uncertainty evaluation.

CI numerical validation:

\`\`\`bash
python -m ntr.nerva.validate_resolved_openmc_feedback
python -m ntr.nerva.run_resolved_openmc_feedback --help
\`\`\`

The test callback is explicitly marked \`synthetic-test\`, never a
transport-equivalent demonstration.


## Real-statepoint provenance and Monte Carlo quality gate

The resolved OpenMC runner now calls
\`statepoint_quality.audit_fresh_transport\` **after each** fresh
OpenMC+analysis run, before accepting any reconstructed heating field.

It independently checks the completed statepoint pathname, SHA-256 and
size against \`transport_provenance.json\`; verifies the exact
\`cross_sections.xml\` filename and SHA-256; checks temperature-coverage
status and the actual material fuel temperature, hydrogen temperature
and density used by transport against the Picard request; confirms that
the \`power/metadata.json\` references the **same** statepoint and has
the imposed MW/W normalization; and checks the analysis QA root,
all reported closure flags and the mesh-heating uncertainty statistics.

The new **\`--max-heating-p95-rel-error\`** option defaults to \`0.5\`:
a 50% ceiling on the 95th-percentile *local mesh heating relative
standard error* reported by the existing OpenMC postprocessor. If that
threshold is exceeded, this is a **hard error**, not silent acceptance.
For higher-fidelity studies tighten the threshold with adequate neutron
sampling. This statistic is a proxy for spatial Monte Carlo tally noise,
**not** an uncertainty bound on engine thrust, Isp or reactor criticality.
It also does not prove uncertainty independence between Picard steps.

A numerical test constructs entirely artificial statepoint/XS bytes and
synthetic metadata to check all failure modes without an OpenMC binary.
That test is **not real transport**; the production runner still requires
a genuine, completed statepoint and a compatible nuclear-data library.

\`\`\`bash
python -m ntr.nerva.validate_statepoint_quality
python -m ntr.nerva.run_resolved_openmc_feedback --help
\`\`\`

These additional integrity checks are necessary for a credible real
OpenMC coupling run but they are not a substitute for historical
experimental validation or a converged neutron-transport demonstration.


## Time-discretization verification on channel-resolved thermal physics

\`temporal_convergence.py\` now performs an independent numerical
timestep-refinement audit using NUKE's existing channel-resolved
solid/hydrogen solver:

- Solve the **same fixed external OpenMC heating**, coolant mass
  flows, heat capacities and solid/gas properties at exactly one common
  end time with \`dt\`, \`dt/2\`, \`dt/4\`, etc.
- Compare **every** solid/H2 state array (fuel solids, fuel H2,
  tie solids, tie supply H2, tie return H2), using each array's
  maximum absolute temperature difference in kelvin.
- Record per-step and cumulative first-law residuals and apparent
  observed temporal order. The scheme is backward Euler and therefore
  first-order accurate in the time step for suitably smooth problems,
  though observed order can vary outside the asymptotic regime.
- Require that the last two solutions differ by no more than a
  specified physical temperature tolerance. Reject nondecreasing
  differences unless the numerical comparison is near roundoff.
- Preserve the production run's original state and return an explicitly
  \`PASS\`/\`FAIL\` report; no Richardson-extrapolated temperature is
  ever inserted as a new thermodynamic state.

The existing CLI accepts:

\`\`\`shell
python -m ntr.nerva.run_resolved_transient \
  MESH_FIELDS.npz FUEL_RECONSTRUCTION.npz TIE_RECONSTRUCTION.npz \
  --statepoint /PATH/TO/statepoint.h5 \
  --transport-provenance /PATH/TO/transport_provenance.json \
  --steps 8 --dt-s 0.1 \
  --refine-levels 4 --refine-tolerance-k 0.25 \
  --initial-solid-k 500 --initial-gas-k 300 \
  --inlet-temperature-k 300 \
  --fuel-solid-capacity-j-k 5000 --fuel-gas-capacity-j-k 500 \
  --fuel-ua-w-k 100 \
  --tie-solid-capacity-j-k 3000 \
  --tie-supply-capacity-j-k 600 --tie-return-capacity-j-k 700 \
  --tie-supply-ua-w-k 80 --tie-return-ua-w-k 90
\`\`\`

The capacities, UA, temperatures and flows are purely illustrative
assumptions, not a validated historical NERVA operating point.
\`--refine-levels 0\` (default) disables the additional computation;
\`3..8\` enables refinement. Output is
\`temporal_convergence_report.json\`, nested into the
\`resolved_transient_summary.json\`. A failed convergence threshold
causes a nonzero CLI exit status.

For a mathematically generated test dataset:

\`\`\`shell
python -m ntr.nerva.validate_temporal_convergence
\`\`\`

**Scope warning:** This verifies the **thermal numerical time step with
a frozen spatial heating field**. It is **not** an OpenMC neutron history
convergence study, Monte Carlo sampling error estimate, 3-D fuel
temperature validation, coupled temperature-feedback convergence
test, or historical NERVA engine accuracy claim. The real OpenMC feedback
driver remains a distinct tool, and coupling its temperature feedback
to the timestep-refinement audit is further work.


## Independent OpenMC spatial-heating replica convergence

The new two-run comparison verifies that apparent temperature-feedback
convergence is not merely *sampling noise* in OpenMC tallies. The
\`run_openmc_transport\` command now accepts an explicit \`--seed\` and
records it in transport provenance. Two real runs should use different
seeds and otherwise IDENTICAL neutron transport configuration,
geometry, materials, tallies and cross-section XML checksum.

At the SAME imposed reactor thermal power, \`openmc_replica_convergence.py\`
and \`run_openmc_replica_convergence.py\` compare:

1. Total variation of normalized three-dimensional energy deposition
   (half the integrated absolute difference in per-voxel power fractions).
2. The largest difference between the two normalized AXIAL power
   distributions, calculated by summing over x and y.
3. The fraction of active bins with a heating difference above three
   nominal combined standard errors. Reported \`std_dev\` is the
   **raw Monte Carlo tally diagnostic after power normalization**:
   dividing through the same noisy total-power estimate introduces
   correlations, so treating these z-scores as independent significance
   tests or confidence intervals would be INCORRECT.
4. Optionally compare the complete fuel-solid/H2 and tie-solid/H2
   temperature fields after applying each heating realization to
   an IDENTICAL, user-specified channel-resolved transient thermal
   problem. This isolates the thermal consequence of the two
   spatial heating shapes.

The result is marked PASS only when all configured heating and,
if supplied, thermal tolerances are satisfied. A missing uncertainty
array, changed mesh, inconsistent power, reused statepoint, reused
random seed, changed transport settings, changed nuclear XS data,
or a missing provenance file causes a HARD ERROR.

### Prepare two independently seeded transport runs

\`\`\`shell
python -m ntr.nerva.run_openmc_transport \
  --cross-sections /PATH/cross_sections.xml \
  --seed 1729 --assembly reactor --rings 5 \
  --particles 4000 --batches 80 --inactive 20 \
  --output build/nerva_replica_a/transport

python -m ntr.nerva.run_openmc_transport \
  --cross-sections /PATH/cross_sections.xml \
  --seed 2027 --assembly reactor --rings 5 \
  --particles 4000 --batches 80 --inactive 20 \
  --output build/nerva_replica_b/transport
\`\`\`

After both complete, process the individual statepoints using the
existing \`postprocess_statepoint\` on each file at the **same MW**
(e.g. replace STATEPOINT_A and STATEPOINT_B below with the actual
statepoint paths recorded in each transport provenance):

\`\`\`shell
python -m ntr.nerva.postprocess_statepoint STATEPOINT_A \
  --power-mw 1 --output build/nerva_replica_a/power

python -m ntr.nerva.postprocess_statepoint STATEPOINT_B \
  --power-mw 1 --output build/nerva_replica_b/power

python -m ntr.nerva.run_openmc_replica_convergence \
  --power-a build/nerva_replica_a/power \
  --power-b build/nerva_replica_b/power \
  --provenance-a build/nerva_replica_a/transport/transport_provenance.json \
  --provenance-b build/nerva_replica_b/transport/transport_provenance.json \
  --maximum-total-variation 0.05 \
  --maximum-axial-difference 0.05 \
  --maximum-significant-voxel-fraction 0.10 \
  --output build/nerva_replica_convergence.json
\`\`\`

The 0.05 / 0.05 / 0.10 limits shown above are **example
engineering acceptance criteria**, NOT NASA NERVA validated tolerances.

For optional channel-resolved thermal impact, also independently run
NUKE's standard \`run_analysis\` pipeline on each statepoint to obtain
the fuel-element and tie-instance reconstruction NPZs. Provide them as
\`--fuel-a\`, \`--tie-a\`, \`--fuel-b\`, \`--tie-b\` and supply a
\`--thermal-case\` JSON with the \`ResolvedParameters\` fields and
\`steps\`, \`dt_s\`, \`initial_solid_k\`, \`initial_gas_k\`,
\`inlet_temperature_k\`, and \`max_temperature_difference_k\`.
The thermal case is intentionally **externally prescribed** and is
not fitted to any historical NERVA data.

Regression test without neutron transport:

\`\`\`shell
python -m ntr.nerva.validate_openmc_replica_convergence
\`\`\`

Limitations: different seeds do not guarantee statistical independence
of the stationary fission source, and the coefficient of variation of
a normalized tally lacks full covariance. Two runs also do not establish
a rigorous reactor heating confidence interval. Actual engineering
qualification requires repeated runs, particles/batches refinement,
source convergence diagnostics, subchannel hydraulic model sensitivity,
real cross-sections, and historical data comparison. There is no
neutronic power/kinetics coupling in this test.


## Four-plus independent OpenMC heating-replica convergence

A new \`openmc_replica_ensemble.py\` analyzer and CLI
\`run_openmc_replica_ensemble.py\` extend the two-seed comparison to
**four or more separately seeded normalized OpenMC statepoints**.

Given a common external thermal power, mesh and transport model, the
analyzer uses each voxel's fraction of total deposited reactor power.
Across \`n\` independent seeded runs, it calculates the unbiased sample
standard deviation (divisor \`n-1\`) and standard error (divisor
\`sqrt(n)\`) of each voxel power fraction. It reports:

- the 95th percentile of \`SEM / mean\` among active voxels;
- the maximum sample-SEM of the axially integrated power *fraction*;
- total-variation distance between the average heating fields of the
  first and second groups (input ordering MATTERS);
- mean/max all-pair power-shape total variation and the largest
  leave-one-out difference;
- mean, sample standard deviation and SEM power-fraction arrays,
  exported to \`ensemble_heating_statistics.npz\`.

The CLI refuses fewer than four runs, different meshes/power,
mismatched nuclear data, repeated OpenMC random seeds/statepoints,
changed transport input parameters and mixed synthetic/real sources.
Threshold failures are explicitly reported as \`FAIL\` and exit
nonzero. Defaults of 10% p95 relative SEM, 1% maximum axial SEM and
5% half-split total variation are **software QA starting points,
NOT published NASA NERVA limits**. Statistical power and multiple-run
power refinement should drive final thresholds.

Run separately seeded OpenMC transport/analysis four or more times,
then compare the existing normalized power directories:

\`\`\`bash
python -m ntr.nerva.run_openmc_replica_ensemble \
  --power-dirs run1/analysis/power run2/analysis/power \
               run3/analysis/power run4/analysis/power \
  --provenance-files \
               run1/transport/transport_provenance.json \
               run2/transport/transport_provenance.json \
               run3/transport/transport_provenance.json \
               run4/transport/transport_provenance.json \
  --max-p95-relative-sem 0.10 \
  --max-axial-fraction-sem 0.01 \
  --max-half-split-total-variation 0.05 \
  --output build/nerva_openmc_replica_ensemble
\`\`\`

This command **analyzes** completed real statepoints; it does not
invoke an OpenMC executable or infer transient reactor power.
A compatible cross-section library and installed OpenMC transport
engine remain prerequisites for producing those statepoints.

The numerical-only CI test is:

\`\`\`bash
python -m ntr.nerva.validate_openmc_replica_ensemble
\`\`\`

It fabricates four nearby spatial-power maps to test exact sample
standard-error and split-sample calculations, rejects duplicate or
mismatched provenance, and checks the JSON and NPZ products. No
reactor physics is solved in this synthetic regression.

**Statistical limitation:** the usual \`sd/sqrt(n)\` estimate requires
approximately independent realizations of the same stationary
distribution. Different OpenMC seeds alone do NOT guarantee fission
source convergence or absence of source bias. The ensemble SEM
also does not include systematic cross-section uncertainties,
spatial-model error, normalization covariance, or historical
NERVA qualification. Additional benchmark experiments remain pending.


## Subsonic compressible hydrogen channel energy and momentum (new)

The repeated-fuel-element **steady thermal-hydraulic reconstruction**
now accepts \`--channel-physics compressible-nasa7\`; its previous
friction-only, approximate-\`cp\` behavior remains available as
\`--channel-physics legacy-friction-only\` (the default for backward
compatibility). This is a substantive physics improvement: the
high-fidelity option solves local **streamwise momentum and
stagnation enthalpy** self-consistently for the given channel diameter,
imposed flow, axial heat deposition and inlet pressure.

Per cylindrical, constant-area axial segment:

\`\`\`text
G = mass_flow / A
v = G / rho

mass_flow * [(h_out + 0.5*v_out^2)
              - (h_in + 0.5*v_in^2)]
    = Q_solid_wall + Q_direct_nuclear_H2

p_in - p_out
    = G*(v_out-v_in)
      + f_D*(dz/D)*G*(v_in+v_out)/4
\`\`\`

The second term is Darcy friction using the mean specific volume
of the two segment endpoints. **Flow acceleration** (the first term)
is calculated separately and reported, preventing the old model's
failure to deduct acceleration losses from the total pressure drop.
The solver computes state-dependent pure-molecular-H2 ideal-gas
density and NASA-7 \(h(T)\) and \(c_p(T)\), rather than treating
hydrogen heat capacity as a fixed constant. It uses existing
screening correlations for viscosity, conductivity and heat transfer;
the current constants should be replaced with a pressure-/temperature-
dependent transport correlation in subsequent work.

Example on a genuine OpenMC fuel-sector reconstruction:

\`\`\`bash
python -m ntr.nerva.solve_element_channels \
  build/nerva_full_run/analysis/power/nerva_mesh_fields.npz \
  build/nerva_full_run/analysis/instance_power/instance_power_fractions.npz \
  --fuel-mass-flow-kg-s 2.0 \
  --inlet-temperature-k 500 --inlet-pressure-mpa 5 \
  --channel-physics compressible-nasa7 \
  --channel-max-mach 0.85 \
  --solid-conduction-model equivalent-annulus-sector \
  --output build/nerva_compressible_fuel
\`\`\`

Adjust the paths to match the existing \`run_analysis\` output.
All numeric inputs here are **illustrative placeholders**, not
NASA-verified NERVA inlet states or a design point.
The CSV channel summary now includes Mach, separate friction and
acceleration pressure losses, and independent energy/momentum residuals
when the advanced channel solver is enabled.

The algorithm uses a bounded Newton solve for pure-H2 stagnation
enthalpy at fixed outlet pressure, plus a damped Newton pressure solve.
A near-choked or non-subsonic solution is a **hard failure**, not an
artificially limited pressure drop. It also refuses temperature
solutions outside the NASA-7 200–3500 K domain or combinations with
CoolProp transport properties without a consistent real-fluid
enthalpy/density backend.

### Physics limitations

- Quasi-steady, constant-area, single-phase pure-H2 gas in each channel;
  no liquid cryogenic inlet, phase change, pump start dynamics, shock,
  transient pressure waves or choked-flow boundary condition.
- External imposed mass flows, no inlet/exit minor losses or orificing.
- Simple Darcy and Dittus-Boelter screening correlations, fixed dynamic
  viscosity and thermal conductivity; no validated high-temperature
  wall friction, erosion/corrosion, radiation, axial conduction or
  fuel-coating damage model.
- This solver improves the **steady OpenMC-channel postprocessor**.
  The channel-resolved **transient** thermal solver still has
  prescribed flows and no dynamic pressure/momentum evolution.
  Do not claim the two are yet a fully transient thermo-hydraulic plant.
- The nuclear heat deposition remains externally normalized to prescribed
  reactor thermal power; no reactor kinetics have been introduced.

Validation:

\`\`\`bash
python -m ntr.nerva.validate_compressible_channel
\`\`\`

Tests independently recompute per-cell mass, total-enthalpy and momentum
closure, compare low-Mach friction with the Darcy limit, preserve direct
hydrogen heating, and reject unacceptable Mach and thermodynamic domains.
They are mathematical verification, NOT actual NERVA experimental
benchmarking. Historical NASA work discusses channel pressure loss and
flow choking: see NTRS technical memo NASA-TM-105867,
https://ntrs.nasa.gov/citations/19930009917 and
NASA thermal-hydraulic methodology
https://ntrs.nasa.gov/citations/20130001802.
