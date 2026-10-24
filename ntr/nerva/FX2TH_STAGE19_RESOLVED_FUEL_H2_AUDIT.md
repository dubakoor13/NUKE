# NUKE / FX2-TH Stage 19 — radial fuel conduction and axially advected hydrogen coupled to r-z neutron kinetics

**Date:** 24 October 2026. **Parent:** Stage 18 PR #130. **Scope:** core neutron and resolved heat-transfer mathematical equations only.

## Existing NUKE physics: what is reused and what is new

NUKE already contains:
- `ntr/nerva/fuel_conduction.py`: screening estimates of fuel-channel half-ligament and equivalent annular solid conduction, including a cylindrical coating logarithmic resistance. It does **not** evolve radial temperature degrees of freedom.
- `ntr/nerva/resolved_transient.py`, `implicit_hydrogen.py`, and `thermal_flow_coupling.py`: axial channel-resolved fuel/tie cooling, independent enthalpy and pump/cycle iteration, and heating partition ledgers. Some H2 models use NASA-7 enthalpy, but H2 pressure/density is not fully iterated. They are NOT automatically connected to the new FX2-TH delayed-neutron operator.
- Stage 18 `fx2th_monolithic_rz_core.py`: full nonlinear neutron-temperature feedback with a **lumped** solid and hydrogen node in each spatial neutronic cell, and a validated external energy-deposition interface.

Stage 19 adds **multiple radial fuel solid rings per neutronic r-z cell**, finite-volume radial and axial conduction, upwind axial coolant enthalpy transport, explicit parallel channel count and first-law conservation. It also implements a one-step fixed-point coupling to the Stage 17 temperature-dependent multigroup neutron operator and Stage 7 family-resolved implicit kinetics. It does NOT replace existing resolved-transient / CoolProp models, nor does it claim a historical NERVA fuel geometry.

## Geometry, unknowns and units

A macroscopic neutronic cell `[radial_zone, axial_z]` is assigned ONE user-specified equivalent channel-associated fuel annulus with **N_parallel explicit channels** and a separate coolant segment. This equivalent annulus is a thermal geometrical surrogate; it is NOT the full Stage 12 macroscopic cylindrical neutron cell. An actual geometrically faithful bridge from hexagonal NERVA fuel cells to these annuli and OpenMC tallies remains pending.

For each radial zone p, axial bin j and fuel ring i:

- `r_edges[p,i]` m: positive ascending micro-annular ring radii (inner radius is strictly positive).
- `z_edges[j]` m: ascending axial bin faces.
- `N_channels[p]`: positive integer count per macroscopic radial zone.
- `rho_s[p,j,i]` kg/m3, `cp_s[p,j,i]` J/(kg K), `k_s[p,j,i]` W/(m K): externally supplied, positive solid properties, frozen over a physical step.
- `m_H2[p,j]` kg: positive stored hydrogen mass per cell; `cp_H2[p]` J/(kg K): constant along one flow path, `m_dot[p]` kg/s: prescribed nonnegative flow in the +z direction.
- `UA_wall[p,j]` W/K: supplied TOTAL inner-ring-to-hydrogen wall conductance (all parallel channels), **inclusive of any external film/coating heat resistance**. It is NOT recalculated from local Reynolds/Prandtl or wall temperature in this stage.
- `T_in[p]` K: prescribed inlet flow temperature.
- `Q_solid[p,j,i]` W, `Q_direct_H2[p,j]` W and `Q_other[p,j]` W: explicit partition of user-supplied deposited power with declared total W and provenance label. `Q_other` is never silently deposited a second time.

A solid FV ring volume, total channels included, is

```
V[p,j,i] = N[p] * pi * (r[i+1]^2-r[i]^2) * (z[j+1]-z[j]).
Cs[p,j,i] = rho_s[p,j,i] * cp_s[p,j,i] * V[p,j,i].
```

For radial ring-centre radii r_c = sqrt((r_in^2+r_out^2)/2), the harmonic cylindrical interface conductance is

```
G_rad = 2*pi*N*dz / (ln(r_face/r_c_left)/k_left
                         +ln(r_c_right/r_face)/k_right)     [W/K].
```

For axial adjacent bins of the same fuel ring, the face area is the whole annular section `A=pi*N*(r_out²-r_in²)`, and

```
G_ax = A / (dz_lower/(2*k_lower)+dz_upper/(2*k_upper))   [W/K].
```

The outer radial face and both axial end faces are insulated. **No heat crosses macroscopic radial zones:** that model boundary is an assumption for independent representative coolant/fuel paths, not a claim that the actual reactor cannot exchange heat across elements.

## Conservative implicit energy equations

For each ring, backward Euler `Cs*(Ts_new-Ts_old)/dt` equals external solid deposition plus signed radial/axial Fourier conduction plus signed wall transfer (inner ring only). For the coolant cell, `Ch*(Th_new-Th_old)/dt` equals direct H2 deposition + wall transfer + `m_dot*cp*(T_upstream_new-Th_new)`; the first bin uses the given inlet temperature. This uses **NEW-TIME upstream** temperatures (positive-flow first-order upwind spatial discretization), not a per-cell reset to the inlet. The entire fuel/coolant system is assembled as ONE sparse linear solve. Backwards Euler is first-order globally and the upwind transport discretization introduces axial numerical diffusion.

The global energy ledger is

```
Delta U_fuel + Delta U_hydrogen
    + dt * sum_paths[mdot*cp*(T_hydrogen_outlet_new-T_in)]
    + dt * Q_other_total
    = dt * Q_declared_total.
```

All internal radial, axial and wall conductances cancel globally; they are NOT additional nuclear heat sources. The signed wall flux can reverse direction. Failing the global first-law residual by more than 1e-9 relative raises an error.

## Coupling to Stage 17 r-z neutronics

`fx2th_resolved_rz_feedback.py` forms a **solid-volume-weighted effective fuel temperature** for every neutron r-z cell. This is an explicit approximation: true multigroup material homogenization requires cross-section processing and source/adjoint weighting of the physical fuel, moderator and reflector distributions. The effective temperature samples Stage 17's externally supplied temperature-knot tables with strict no-extrapolation. Stage 7's actual full spatial multigroup/family delayed-neutron kinetics is re-solved from the SAME old physical neutron state on every fixed-point trial. External power must arrive as a Stage-19 `ConjugateCoreDeposition`; no neutron eigenmode or arbitrary-normalized flux is converted to watts.

Each iteration solves:
1. Nuclear material and full neutron/precursor BE at trial effective fuel temperature.
2. Explicit power callback for fuel-ring, direct hydrogen and other compartments at that trial neutron/material state.
3. Complete **new** resolved fuel conduction / axial H2 advection BE from the SAME old thermal state.
4. Relaxation in fuel-ring and gas temperatures; convergence requires all resolved temperature components agree, not only the fuel volume average.

If **ANY individual fuel ring** (including a ring obscured by a temperature average) leaves the supplied nuclear-tabulation domain, depositions are invalid, or iterations fail, the entire physical step is rejected; it does not silently clip or substitute synthetic material data. A fuel-volume average within the table range is not sufficient evidence of coverage for all constituent rings. This method is a conservative **Picard** temporal interface, not an automatically stable monolithic nonlinear resolved-core method.

## Independent numerical regressions

`validate_fx2th_conjugate_fuel_transport.py`:
- Analytic total annular volume and channel-count weighting.
- Exact logarithmic radial resistance and nonuniform harmonic axial conductance.
- Correct upstream/downstream (non-symmetric) sparse coolant advection matrix.
- Large-time stationary heated-annulus cylindrical resistance, wall-film drop and H2 outlet temperature.
- Independent steady coolant enthalpy-rise profile from cumulative deposited powers.
- Uniform closed-system equilibrium with zero power and zero flow.
- Conservation of solid and gas storage, external outlet enthalpy and explicitly unmodeled deposition.
- Reversible wall heat flux, no spurious cross-path flux, nonuniform material conductivity/density, sparse solver versus independent dense linear solve, volume-weighted XS-effective fuel temperature, invalid-geometry/power/time guards and input immutability.

`validate_fx2th_resolved_rz_feedback.py`:
- Converged spatial neutron flux, family precursors, resolved ring/gas temperatures and independent global first-law ledger.
- Agreement against an independently evaluated Stage 7 single neutron BE from the same OLD neutron state using the final Stage 17 temperature field.
- Agreement against a standalone Stage 19 thermal solve at the final externally calculated deposition.
- Zero-heating one-way thermal/neutron reduction, external source, material fingerprint change when radial fuel-ring temperature changes, incompatible meshes, nonconvergence, deposition provenance and reproducibility.
- Dedicated GitHub Actions also runs Stage 18 monolithic benchmark regressions.

## Unresolved physical fidelity — do not claim a hot NERVA benchmark

This is **physics-equation implementation and synthetic mathematical verification, not NASA NERVA validation**. No evaluated temperature-dependent NERVA multigroup data, transport-driven volumetric nuclear heat (including neutron/photon partition), fuel/reflector real material composition, actual hexagonal fuel-element coating/channel geometry, real hydrogen enthalpy `h(P,T)`, density/pressure work, dissociation chemistry, compressibility/choking, film/boiling correlations, or completed thermal/hydraulic reactor power closure was established. Fixed cp and prescribed m_dot are explicit approximations. These thermal paths are not aligned to a verified NERVA reactor neutron/material mesh. Evaluated data and the existing NUKE higher-fidelity coolant EOS should be the next physical integration priorities, not more synthetic reactor operating numbers.

This work does not alter NUKE main. Stacked on Stage 18 PR #130. Do not merge without parents.
