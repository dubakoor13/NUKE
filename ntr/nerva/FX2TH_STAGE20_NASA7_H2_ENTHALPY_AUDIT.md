# NUKE FX2-TH Stage 20 — resolved fuel / H₂ with nonlinear molecular enthalpy

**Date:** 24 October 2026 · **Parent:** Stage 19 PR #131 · **Scope:** fuel-ring conduction and hydrogen energy equations, mathematical verification only.

## Existing physical model reused

- Stage 19 (`fx2th_conjugate_fuel_transport.py`) already implements exact cylindrical logarithmic radial fuel conduction resistance, axial harmonic Fourier conduction and upwind H₂ energy transport. Its hydrogen thermal term is linear `m*cp*(Tnew-Told)` and its flow term `mdot*cp*(Tup_new-Tnew)`, with `cp` assumed constant.
- Existing `ntr/nerva/implicit_hydrogen.py` provides `hydrogen_specific_enthalpy(T,reference_cp,model)` using either constant cp or the repository's **pure frozen molecular-H₂ NASA-7 thermodynamic polynomial**. The latter calls `ntr/nerva/nasa_hydrogen.py` and is valid on `200 <= T <= 3500 K`.
- Stage 20 reuses those existing H₂ enthalpy and frozen `cp=dh/dT` functions, rather than inventing new coefficients, changing energy references, or claiming a physical dissociation or pressure model.

## Coupled finite-volume energy equations

Every representative thermal channel has externally supplied solid fuel rings (ring index i), axial segment z, and hydraulic path p. Fuel ring heat capacity `Cs_i = rho_i*cp_i*V_i`, with exact Stage-19 equivalent-annulus volume and log-radius harmonic interfaces.

The Stage-19 solid FV equation is **unchanged**:

`Cs_i * (Ts_i(new)-Ts_i(old))/dt = Qsolid_i + sum_internal Kij*(Ts_j(new)-Ts_i(new)) - UA_i*(Ts_i(new)-Th_z(new))`.

Only the innermost ring exchanges heat directly with H₂. Fuel conductivity and solid heat capacity are *externally supplied, constant during a physical step*. No radial/axial flux exits insulated thermal boundaries.

The new hydrogen equation uses exact mass-specific enthalpy:

`m_h2,z * [h(Th_z(new))-h(Th_z(old))]/dt = Qdirect_H2,z + UA_z*(Ts_inner,z(new)-Th_z(new)) + mdot*[h(Tup_z(new))-h(Th_z(new))]`.

For z=0, `Tup=T_inlet`; for z>0, `Tup_z(new)=Th_(z-1)(new)`. Thus the upstream segment's **new** enthalpy is used and every flow path conserves energy. Pressure/momentum are NOT integrated and gas mass per cell remains fixed. This is NOT a compressible nozzle or a model of transient inventory continuity.

A sparse full-state Newton solve uses the exact Stage-19 solid matrix and hydrogen enthalpy residual. Its analytic Jacobian consists of the existing solid Fourier/film terms and:

`dRgas_z/dTh_z = (m_z/dt+mdot)*cp(Th_z)+UA_z`,
`dRgas_z/dTh_(z-1) = -mdot*cp(Th_(z-1))`.

Here `cp(T)=dh(T)/dT` is evaluated using the ORIGINAL NUKE NASA-7 frozen-H₂ `hydrogen_state` function. The same chemical reference `h(T)` is used for inlet, storage, and outlet. When model is set to `constant`, the equations analytically reduce to the Stage 19 constant-`cp` sparse linear system.

Newton is damped by backtracking with strict positive T and NASA-7 gas-temperature bounds. Unconverged, out-of-range, invalid source, bad timestep, singular Jacobian, or insufficient iteration budget fails closed; no temperature clipping, physical parameter rescaling, or arbitrary final-state acceptance.

## Exact energy conservation

The accepted first law is independently evaluated in joules:

`DeltaU_solid + sum(m_h2*[h(Th_new)-h(Th_old)]) + dt*sum_paths(mdot*[h(Tout_new)-h(Tin)]) + dt*Qother = dt*Qdeclared`.

`Qother` is always a separately tracked deposition compartment, never added to solid or gas again. Signed wall and axial heat fluxes cancel internally. An absolute equation-residual gate and a relative first-law balance gate are both mandatory. No arbitrary-normalized neutron flux is converted to reactor watts; deposition comes exclusively from the supplied Stage-19 `ConjugateCoreDeposition` object.

## Independent mathematical checks

The Stage-20 test suite uses synthetic annular geometries, power partitions, masses and flows and compares:
1. Existing NASA-7 enthalpy's numeric d h/dT against the independent frozen-H₂ specific heat evaluated by `hydrogen_state`.
2. The `constant` model's complete fuel/gas temperature and storage/outlet energy values against **Stage 19's original independent sparse linear solver**.
3. One-cell nonlinear H₂ storage and inlet advection against an **independent Brent scalar enthalpy root**.
4. Multi-bin stationary +z H₂ advection against analytically accumulated power in enthalpy, inverted separately at each station with Brent roots.
5. Variable-H₂ cp predictions genuinely differ from constant-cp results in a synthetic heating scenario.
6. Global fuel/H₂ storage plus hydrogen outlet enthalpy plus unmodeled deposition exactly equals external declared power, including multiple annular rings, axial heat conduction, reverse wall fluxes and parallel channels.
7. Full analytic coupled Jacobian-times-vector against an independent centered residual finite difference.
8. No `reference_cp` contamination of NASA-7 H₂ enthalpy, by modifying Stage-19 reference cp but obtaining the same molecular-H₂ result.
9. Cryogenic gas and inlet temperatures below 200 K rejected, and excessive heating that would require gas T above 3500 K rejected rather than extrapolated.
10. Fails on nonphysical power, timestep, or solver budget; deterministic and immutable inputs.

The dedicated GitHub CI reruns Stage 19's 17 conjugate thermal regressions as well as this new suite.

## Limitations that remain

- The existing GRI-Mech 3 NASA-7 molecular `H₂` polynomial is valid for *frozen molecular hydrogen* over **200–3500 K**. It does **not** cover cryogenic liquid-H₂ inlets, high-temperature H₂/H equilibrium dissociation or real-gas compression. Pressure remains externally fixed/unused in this enthalpy calculation. Earlier NUKE CoolProp-based supply/nozzle solvers are separate and must not be confused with this gas-phase thermal model.
- A fixed H₂ mass in each cell combined with externally imposed fixed mass flow does not solve mass or momentum conservation during actual pressurization/depressurization. Full pressure/density coupled hydraulics remain to be implemented/validated.
- Equivalent annular fuel geometry does not reconstruct an actual historical NERVA hexagonal fuel sector or guarantee radial/axial association with the core's nuclear transport cells.
- No actual NASA NERVA material masses, fuel conductivity tables, spatial neutron/photon deposition, verified nuclear-data source, reactor hot-core operating conditions or experimental reactor benchmark was run. Synthetic numerical first-law closure is **not** empirical validation.
- **Stage 20 also supplies an integrated neutron–thermal bridge:** `fx2th_enthalpy_rz_feedback.py` invokes the original Stage 19 Picard neutron/precursor iteration with the nonlinear NASA-7 enthalpy thermal step injected through an optional `thermal_stepper` callback. The old Stage 19 constant-cp path remains the default, unchanged unless the caller explicitly opts into the Stage 20 wrapper. The full loop still requires external W deposition, prescribed fixed coolant mass/flow, synthetic or independently qualified temperature-indexed nuclear data, and valid temperature ranges. This has tests comparing the coupled states against **three independent references**: the Stage 19 constant-cp feedback (when constant mode is selected), the Stage 7 standalone neutron backward-Euler at final material coefficients, and the Stage 20 standalone nonlinear enthalpy solve at final deposition. This is an *equation verification*, not a NERVA physical validation.

Stage 20 is stacked on Stage 19 #131; main is not modified. No historical NERVA design conclusions or reactor performance predictions are claimed.

## Stage 20 spatial neutronics integration verification

The new `validate_fx2th_enthalpy_rz_feedback.py` exercises the coupled full multigroup neutron/precursor/fuel-ring/hydrogen state: molecular H₂ enthalpy first-law, full family precursor balance, exact same old state per Picard trial, constant-cp reduction versus the independent Stage 19 coupled calculation, neutronic independent backward-Euler check at the final homogenized material temperature, independent enthalpy thermal re-evaluation, zero deposited heat reduction, source power provenance, impossible cryogenic NASA7 initial states, iteration nonconvergence and deterministic old-state preservation. The Stage 19 feedback regression is included in GitHub CI to ensure the thermal-stepper option does not change the default.
