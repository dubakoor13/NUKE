# 19-channel laterally coupled reactor fuel-element thermal solver

**Implementation:** `core_hex_lateral_fv.py`  
**Runner:** `run_core_hex_lateral_fv.py`  
**Independent audit:** `core_hex_lateral_balance.py`  
**Numerical checks:** `validate_core_hex_lateral_fv.py`

This is a conservative *thermal* model of one NERVA-derived 19-bore fuel
element. It does not solve reactor multiplication, neutron transport,
control drum movements, nuclear heating kernels, power feedback or
reactor-core-wide cross-element conduction.

## What changed compared with independent equivalent annuli

The earlier `core_conjugate_fv.py` solves detailed radial/axial solid
conduction around a **single** coolant channel; the outer radius is
adiabatic. This solver instead connects the **19 channel-associated
Voronoi fuel sectors laterally** using the exact shared Voronoi face
length and channel-to-channel center distance computed from the
existing `fuel_sector_geometry.py`.

The two alternatives are complementary, **not yet mathematically
combined into a simultaneously radial-resolved, laterally coupled
3-D solid model**. Neither is an actual 3-D element mesh.

### Spatial degrees of freedom

At each axial bin `z` and channel-sector `i`:

- `T_f(z,i)`: fuel-sector thermal node (lumped fuel matrix)
- `T_c(z,i)`: ZrC-coating thermal node
- `T_H2(z,i)`: mixed hydrogen-control-volume temperature

Each fuel cell has transverse Fourier links to geometrically adjacent
Voronoi fuel-sector cells, axial Fourier links to the same sector in
adjacent axial planes, a radial fuel-to-coating link, and a direct
nuclear heat-source input. Each coating has axial conduction and
convection via the hydrogen boundary. The H2 has optional direct
deposition, axial enthalpy advection and transient enthalpy storage.

### Discrete energy equations

Fuel, for each axial/channel cell:

```text
d(E_f)/dt
  + sum_faces G_ij (T_f,i - T_f,j)
  + Q_f_to_coat = Q_nuclear_f
```

Coating:

```text
d(E_c)/dt
  + sum_axial_faces G_c,ij (T_c,i - T_c,j)
  + Q_coat_to_H2 - Q_f_to_coat = 0
```

Hydrogen:

```text
m_g (h(T_new)-h(T_old))/dt
  + mdot [h(T_new)-h(T_new,upstream)]
  - Q_coat_to_H2 = Q_direct_H2
```

Every shared face is assembled with opposite signs in its two
neighboring equations. Nonlinear solid conductivities and heat
capacity, plus frozen H2 NASA-7 enthalpy, use Picard coefficient
updates; the complete system is assembled as a sparse matrix for each
iteration, with backward Euler or steady state as requested.

The source is **prescribed** by existing NUKE axial reconstruction,
not inferred from a desired reactor power or `k_eff`.

### Lateral face conductance

For two neighbors `i,j`, the two-point Fourier closure uses

```text
G_ij = L_shared * Delta_z /
  ( (pitch_ij / 2)/k_f(T_i) + (pitch_ij / 2)/k_f(T_j) )
```

The faces belong to the actual clipped channel Voronoi polygons
including the external hexagonal envelope. For the current public
19-channel layout, numerical tests found **42 interior shared faces**.

The two-point approximation is naturally orthogonal for the Voronoi
site locations, but the fuel node is an **area-average subgrid
temperature surrounding a circular bore**, not a true temperature
point at the (excluded) coolant-center site. Thus the coarse lateral
resistance is an explicit closure assumption requiring mesh-based
error assessment. A full high-fidelity step will replace these nodes
with an explicit 2-D meshed fuel cross section.

### Radial heat-transfer closure

Fuel-to-coating and coating-to-H2 links include cylindrical conduction
in each equivalent fuel/coating annulus, optionally a fuel/ZrC contact
resistance, and a prescribed convection coefficient. The equal-area
fuel annulus is defined by the original sector fuel-material area.

In transient mode, fuel/ZrC energies integrate user-supplied
temperature-dependent heat capacity; gas energy integrates H2
enthalpy. Mass flows and H2 inventory are externally supplied and
do not evolve through pressure/continuity equations here.

## Run from NUKE's existing channel reconstruction

```bash
python -m ntr.nerva.run_core_hex_lateral_fv \
  build/nerva_element_channels/element_channel_power_reconstruction.npz \
  --element-index 0 \
  --inlet-temperature-k 500 --initial-temperature-k 500 \
  --film-w-m2-k 5500 --hydrogen-density-kg-m3 3 \
  --gas-enthalpy-model nasa7-h2 \
  --fuel-density-kg-m3 1850 --fuel-cp-j-kg-k 1200 \
  --fuel-conductivity-w-m-k 24 \
  --coating-density-kg-m3 6400 --coating-cp-j-kg-k 650 \
  --coating-conductivity-w-m-k 18 \
  --dt-s 0.1 --steps 2 --output build/nerva_hex_lateral
```

**All numbers above are illustrative input placeholders**, not material
properties from a qualified operating NERVA fuel assembly.

The CLI outputs a JSON summary and NPZ spatial fields containing the
19 fuel, coating and gas temperatures; lateral interface heat flows;
and coating-to-coolant heat flows.

## Verification and limitations

`python -m ntr.nerva.validate_core_hex_lateral_fv` exercises:

1. Genuine 19 Voronoi-sector shared-face topology and positive areas
2. Nonuniform heating inducing physically signed lateral heat flux
3. Comparison with transverse conduction suppressed, at fixed source
4. Exact assembly-level energy closure from input heat to H2 outlet
5. Transient solid/gas thermal energy storage
6. Fuel/coating contact resistance sensitivity
7. No-source thermal equilibrium and domain-guard tests
8. An **independent cellwise audit** of transverse/axial Fourier flux,
   solid heat capacity, H2 enthalpy advection and wall transfer

The independent verifier does not reuse the solver's assembled sparse
matrix and reports local residuals separately from the global sum.
Passing those tests establishes numerical consistency of the stated
reduced equations, not hardware accuracy.

**Not demonstrated:** qualified hot evaluated nuclear cross sections,
nonzero calibrated local KERMA, nuclear temperature feedback,
neutron-source statistical stationarity, operating criticality,
historical NASA model agreement, fully resolved 3-D thermal conduction
between all fuel channels/elements, or a validated hydrogen pressure/
momentum model. Neither this solver nor synthetic heating inputs
should be presented as a certified reactor model.
