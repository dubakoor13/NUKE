# Reactor-core conjugate finite-volume thermal physics

**Status:** implemented Python numerical model, with independent discrete-balance
checks and analytical radial-conduction screening. **Not** historical NERVA
reactor qualification; **not** a criticality or reactor-power solution.

## Source code

- `core_conjugate_fv.py`: monolithic implicit 2D solid/1D hydrogen model
- `core_conjugate_fv_diagnostics.py`: independently reconstructed face fluxes,
  control-volume storage and advection residuals
- `run_core_conjugate_fv.py`: production-shaped adapter for existing
  `element_channel_power_reconstruction.npz` inputs
- `validate_core_conjugate_fv.py`: deterministic analytical/synthetic tests
- `.github/workflows/nerva-core-fv-physics.yml`: fast numerical CI

## Conservation equations

For axial bin `i` and radial coating/fuel bin `j`:

```text
[rho_j * V_ij * (u_j(T_new)-u_j(T_old)) / dt]
  + (F_radial_outer - F_radial_inner)
  + (F_axial_positive - F_axial_negative)
  = Q_nuclear_solid_ij
```

The thermal internal-energy increment uses the integral of the user-provided
temperature-dependent `cp_j(T)`, implemented as an exact secant for the
currently supported linear-in-temperature property law. In steady state the
storage term is zero.

At each radial interface, a face is shared by *exactly two* neighboring solid
cells with equal and opposite flux. Conductance consists of two logarithmic
cylindrical half-cell resistances. At the fuel/coating interface, an optional
resistance `R''_contact / A_interface` is added in series. Axial faces use
series half-cell Fourier resistances. The outer annulus and the upper/lower
axial end faces are adiabatic.

The coated hydrogen bore uses a solid half-cell Fourier resistance and a
film resistance in series; the identical interface flux enters the hydrogen
equation with opposite sign.

```text
m_H2_i * (h(T_new_i)-h(T_old_i))/dt
  + mdot * [h(T_new_i) - h(T_new_upstream_i)]
  = Q_solid_to_H2_i + Q_direct_neutron_H2_i
```

The upwind hydrogen enthalpy is implicit at the new time level. For constant
H2 cp, `h=cp*T`; for frozen molecular H2, the existing NASA-7 `h(T)`
implementation is used. Gas mass, mass flow, film coefficient and pressure
are currently prescribed. They are not advanced through continuity and
momentum equations.

At convergence, the *combined domain* equation is:

```text
d(E_solid + E_H2)/dt =
  Q_nuclear_solid + Q_direct_nuclear_H2
  + mdot * (h_in - h_out)
```

The independent checker reconstructs every radial/axial heat flux and each
gas-bin enthalpy increment **without reusing the sparse matrix coefficients**,
then reports local and integrated signed residuals.

## Geometry and power input

The current channel-resolved model uses NUKE's 19 fuel-channel Voronoi-area
reconstruction, replacing each clipped irregular sector with an *equal-fuel-
area annulus*. ZrC coating is explicitly meshed. This captures a radial
gradient and longitudinal solid conduction, but **not** angular conduction
or actual 2D hexagonal cross-channel heat transfer.

The code accepts `[axial]` sector nuclear power, distributed to fuel radial
cells in proportion to their volume; or an explicit `[axial,radial]` power
matrix to preserve supplied radial source detail. Direct nuclear power to H2
remains disjoint from solid power. Never infer reactor deposition from
`k_eff` or substitute unqualified MT901 data. The CI NNDC cold test
library remains inadequate for qualified hot local KERMA.

## Numerical controls

The sparse assembled system solves all solid and gas temperatures together
using `scipy.sparse.linalg.spsolve`. Picard updates temperature-dependent
conductivities and both solid/gas enthalpy secants; under-relaxation is
available. Failure to converge, non-positive material coefficients, invalid
temperature domains, corrupt mesh dimensions and first-law failure stop
execution. Each Picard iteration represents the **same** timestep; it must
not update the previous-time state until convergence.

One transverse sector can be solved steady (`--dt-s` omitted) or in
backward-Euler transient mode (`--dt-s` specified). The CLI advances the
requested number of timesteps and serializes the final spatial fields and a
per-channel first-law ledger.

## Example invocation (arbitrary screening properties, NOT NERVA reference)

```bash
python -m ntr.nerva.run_core_conjugate_fv \
  build/nerva_element_channels/element_channel_power_reconstruction.npz \
  --element-index 0 --channel-index 0 \
  --inlet-temperature-k 500 \
  --initial-solid-temperature-k 500 \
  --initial-gas-temperature-k 500 \
  --convection-w-m2-k 4500 \
  --coolant-density-kg-m3 3 \
  --gas-enthalpy-model nasa7-h2 \
  --fuel-density-kg-m3 1850 \
  --fuel-cp-j-kg-k 1250 \
  --fuel-conductivity-w-m-k 23 \
  --coating-density-kg-m3 6400 \
  --coating-cp-j-kg-k 650 \
  --coating-conductivity-w-m-k 19 \
  --coating-radial-cells 2 --fuel-radial-cells 16 \
  --dt-s 0.1 --steps 2 \
  --output build/nerva_core_conjugate
```

`core_conjugate_report.json` includes independent per-cell residual
summaries; `core_conjugate_fields.npz` stores radial/axial temperatures and
wall heat-transfer arrays.

## Verification versus physical validation

| Test | Meaning |
|---|---|
| Zero-deposition equilibrium | Detects artificial heating |
| Analytical annular steady solution + radial grid refinement | Tests continuum Fourier operator and radial boundary conditions |
| Variable-property transient backward Euler | Tests storage and nonlinear enthalpy numerics |
| Independent cellwise flux reconstruction | Detects local energy creation/loss, including shared interfaces |
| Fuel–coating contact resistance sensitivity | Tests added series boundary resistance |
| Bad input/domain checks | Fail closed on nonphysical mesh/state |
| Real OpenMC hot cross sections, KERMA, source convergence | **Not established** |
| Historical NASA NERVA criticality, power and temperatures | **Not established** |

For qualified hot reactor predictions, independently characterize materials,
film correlations, coolant state/pressure, nuclear deposition, lateral sector
coupling, thermal scattering and NASA reference geometry. Neither synthetic
tests nor passing CI can substitute these tasks.
