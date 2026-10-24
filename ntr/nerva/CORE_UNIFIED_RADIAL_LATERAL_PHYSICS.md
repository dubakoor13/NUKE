# Unified radial–axial–lateral conjugate reactor-core thermal numerics

**Implemented:** \`core_hex_radial_lateral_fv.py\`  
**Independent energy audit:** \`core_hex_radial_lateral_balance.py\`  
**NUKE reconstruction runner:** \`run_core_hex_radial_lateral_fv.py\`  
**Numerical tests:** \`validate_core_hex_radial_lateral_fv.py\`

## Scope

This option unifies the previously separate physics capabilities:

1. The single-channel radial–axial annular fuel and ZrC finite-volume model
2. The 19-channel clipped Voronoi geometry and lateral Fourier exchange
3. Independent axial H2 enthalpy transport in each of the 19 bores
4. One monolithic implicit sparse solid+gas solve, transient or steady

Fuel and coating temperatures are unknowns \`Tsolid[axial,channel,radial]\`.
Hydrogen is \`TH2[axial,channel]\`. Radial nodes are independent for each
of the 19 unequal-area Voronoi sectors, and have a shared axial grid.

The model is **not** a true hexagonal 3-D mesh. Each irregular fuel
sector has an equal-volume annular subgrid in radius. Transverse links
connect the *outermost* fuel nodes of neighboring annuli using a
two-point conductance based on the actual shared face length, channel
center spacing and fuel conductivities. Connecting outermost nodes is a
reduced-order **subgrid closure**: the associated distances should not
be interpreted as a conforming control-volume geometry. This is the
remaining geometric limitation, rather than pretending a ring is a
hexagonal XY finite-volume mesh.

## Discretization

For radial fuel/coating cell \`(i,c,j)\`, units W:

\`\`\`text
V rho * [u(T_new)-u(T_old)] / dt =
    Q_nuclear
  - (F_radial_outer-F_radial_inner)
  - (F_axial_positive-F_axial_negative)
  - F_lateral_net
\`\`\`

The radial cylindrical interior face Fourier conductance is

\`\`\`text
G_r = 2*pi*dz /
  ( ln(r_face/r_center_inner)/k_inner
    + ln(r_center_outer/r_face)/k_outer
    + [contact_R''/r_face] at the fuel–coating interface )
\`\`\`

Axial conductance uses series half-cell resistances \`dz/(2*k)\`,
with each annular physical face area \`pi*(r_out^2-r_in^2)\`.

Shared lateral Voronoi face \`a-b\`:

\`\`\`text
G_ab = L_face * dz /
   ( pitch_ab/(2*k_outer_a) + pitch_ab/(2*k_outer_b) )
F_ab = G_ab * (T_outer_a - T_outer_b)
\`\`\`

Each pair appears *once* and contributes \`+F_ab\` to the outgoing
heat balance of sector \`a\` and \`-F_ab\` to that of sector \`b\`.
The same exchange conserves total solid energy even when neighboring
sector radii, power deposits and fuel conductivities differ.

Inner bore convection uses series half-cell cylindrical conduction
and film resistance. Exactly the same wall heat flux with opposite
sign enters the hydrogen control volume:

\`\`\`text
m_H2_i*[h(T_new)-h(T_old)]/dt
 + mdot * [h(T_new_i)-h(T_upstream_new)]
 = Q_wall + Q_direct_H2
\`\`\`

H2 is frozen molecular NASA-7 or a specified constant-cp model,
not a reacting/real-fluid hydrogen kinetic solver. Pressure,
momentum, film coefficient, gas inventory and mass flow are external
specified inputs, not additional solved conservation equations.
Fuel/coating cp(T) and k(T) use the explicitly user-supplied
linear-in-temperature positive-property law.

In steady mode the storage terms vanish; transient mode uses backward
Euler and enthalpy secants. Picard iterations update the conductance
and enthalpy coefficients and relax the complete \`[solid,gas]\` state.
A nonconverged Picard iteration is not a physical timestep.

## Source normalization and provenance

- Nuclear source is externally specified \`[axial,19]\` W or full
  \`[axial,19,radial]\` W.
- A per-sector/axial total is distributed only to fuel radial cells,
  proportional to physical annular volume; none is invented in ZrC.
- Explicit \`[axial,19,radial]\` source input preserves detailed local
  deposition, including coatings, only if the caller actually supplies
  and has qualified it.
- Direct hydrogen nuclear power is an independent input and must not
  be included again in solid deposition.
- The repository's prior cold NNDC neutron data still lack physical
  high-temperature evaluated KERMA/thermal-scattering qualification.
  This solver **cannot** qualify the source data by applying a desired
  total reactor thermal-power normalization.

A production-like runner accepts existing NUKE
\`element_channel_power_reconstruction.npz\` arrays:

\`\`\`bash
python -m ntr.nerva.run_core_hex_radial_lateral_fv \
  build/nerva_element_channels/element_channel_power_reconstruction.npz \
  --element-index 0 \
  --inlet-temperature-k 500 --initial-temperature-k 500 \
  --hydrogen-density-kg-m3 3 --film-w-m2-k 5500 \
  --hydrogen-model nasa7-h2 \
  --fuel-density-kg-m3 1850 --fuel-cp-j-kg-k 1250 \
  --fuel-conductivity-w-m-k 23 \
  --coating-density-kg-m3 6400 --coating-cp-j-kg-k 650 \
  --coating-conductivity-w-m-k 19 \
  --coating-radial-cells 2 --fuel-radial-cells 8 \
  --dt-s 0.1 --steps 2 --output build/unified_core
\`\`\`

Numerical values shown are **illustrative, user-specified assumptions**,
not calibrated NERVA operating conditions. Outputs include
\`unified_radial_lateral_core_fields.npz\` and
\`unified_radial_lateral_core_report.json\` with stored energy, face heat,
lateral exchange and first-law audits.

## Independent numerical verification

The new \`core_hex_radial_lateral_balance.py\` computes physical radial,
axial and lateral Fourier face fluxes and H2 advection from temperatures
and material laws independently of the assembled sparse matrix, and
checks *each* solid and coolant cell's first law (in watts). This
is separate from the solver's own global first-law check.

The tests include:

- Lateral hotspot transfer from an asymmetric prescribed heat source
- Positive and oppositely signed heat transfer across 19-channel faces
- Identical input state with all transverse faces suppressed
- The disconnected radial-lateral formulation agreeing with NUKE's
  **separate radial/axial single-channel solver** to numerical tolerance
- Full local and global steady first-law closure
- Full local and global transient first-law closure with enthalpy storage
- A \`[z,channel]\` heat field matching an explicitly volume-distributed
  \`[z,channel,radial]\` source field
- Fuel/coating contact thermal resistance sensitivity
- No-source equilibrium and invalid input/domain rejection
- CI execution of the command-line runner against synthetic but
  NUKE-shaped \`[element,19,z]\` heating arrays

Prior annular analytical Fourier mesh-convergence tests remain
in \`validate_core_conjugate_fv.py\` and are not superseded.

## Physical validation still pending

Genuinely high-temperature evaluated nuclear data with local energy
deposition, independent OpenMC fission-source convergence,
neutronic/thermal fixed-point coupling, physical fuel conductivity and
contact properties, validated H2 thermal/hydraulic correlations and
**conforming true 3-D hex geometry** are not established here.
Tests validate a specified reduced set of equations and source
bookkeeping, NOT a historical NERVA reactor operating point or
neutron-criticality condition.
