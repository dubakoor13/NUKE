# NUKE FX2-TH Stage 18 — fully implicit nonlinear spatial reactor-core thermal feedback
Date: 24 October 2026. Parent: Stage 17 PR #129. Scope: **reactor-core physics/numerical equations, synthetic verification only**.

## Why this is beyond the earlier thermal-boundary stage

Stage 8 supplied an external neutronic-material callback with a fixed-point Picard solver and lumped solid/hydrogen energy equations. Stage 17 introduced spatial temperature-indexed multigroup diffusion, scattering, fission and delayed-neutron data with strict no-extrapolation and analytically checked adjoint slopes. Stage 18 now assembles **one simultaneous nonlinear neutron–precursor–solid/hydrogen step** in which local core material temperature changes alter the actual neutron loss matrix, fission production, and precursor source equations at the end of the timestep.

This is a small-system mathematical benchmark. No real NERVA operating state, temperature law, thermal power, neutron data, flow/network or OpenMC statepoint is invented. External deposited power in watts remains a required callback.

## Mathematically coupled discrete equations

For a fixed nonuniform cylindrical r-z mesh, spatial cell c, energy group g, precursor family f, and fuel/coolant temperatures Ts,Th:

Neutron & precursor unknown vector y=[phi[c,g], C[c,f]], with **cell-integrated** precursor inventory C. Stage 14 supplies the full coupled generator G(Ts), including stage 12 conservative annular leakage, source/removal/scattering and Stage 15 family-resolved prompt/delayed fission birth operators. Stage 17 samples actual supplied tabular D(Ts), Sigma_a(Ts), Sigma_s(Ts), nuSigma_f(Ts), beta(Ts), prompt chi(Ts) and family-delayed chi(Ts). The spatial mass M=V/v, group velocities and decay constants stay fixed, as contracted by Stage 17.

Monolithic backward Euler:
  y1 - y0 - dt*(G(Ts1)*y1 + b) = 0

Explicit separate thermal storage and coolant enthalpy:
  Cs*(Ts1-Ts0) - dt*(Qsolid(y1,Ts1)-H*(Ts1-Th1)) = 0
  Ch*(Th1-Th0) - dt*(QH2(y1,Ts1)+H*(Ts1-Th1)-m_dot*cp*(Th1-Tin)) = 0

Qsolid, QH2 and Qother must be explicitly supplied as a **ThermalDeposition** with a nonblank source identifier, finite nonnegative powers and a declared total W matching all compartments. Under no circumstances is an arbitrary-normalized neutron diffusion eigenflux assumed to equal a physical fission power in W. This also means a physically correct neutron-to-thermal feedback loop remains CONDITIONAL on separately qualified OpenMC heating and amplitude normalization.

The Stage 8 independent ledger independently checks:
  ΔUsolid+ΔUH2+Eexport+Eunmodeled = Δt*Qtotal.

The neutron+precursor balance is checked from the current G(Ts1) and y1, not by comparing its own residual alone.

## Nonlinear numerical method

Damped Newton iteration on x=[phi,C,Ts,Th]. Every residual evaluation:
1. Samples the SAME documented Stage 17 temperature tables at the trial **fuel** temperature field in K.
2. Reassembles the full Stage 12 + Stage 15 neutron kinetics generator.
3. Obtains the required EXTERNAL deposition callback using the trial neutron field and the current material coefficients.
4. Computes all neutron/precursor and two-node thermal backward-Euler residuals simultaneously.

A centered or positivity/range-compatible one-sided finite-difference Jacobian is solved with dense linear algebra, with a residual-decreasing backtracking line search, strict positivity and no temperature extrapolation. Failing the maximum number of Newton attempts, line search or allowable temperature range raises instead of accepting a false result. The maximum DOF is deliberately capped at 120; this is a **small problem verification implementation**, not a scalable NERVA core production Newton-Krylov or resolved thermohydraulic code. Temperature-knot crossings are handled by the actual piecewise tabular interpolation values, not a falsely continuous analytic derivative at a knot.

All stages use the SAME old physical state for one backward-Euler step. The solver never overwrites the previous timestep or takes multiple physical steps inside Newton. Despite Stage 14 SDIRK2 being available, this single integrated method is **first order globally**, because its monolithic thermal and neutron residuals are both backward Euler. No mixed-order misleading claim is made.

## Independent numerical checks

`validate_fx2th_monolithic_rz_core.py` verifies:
- Joint nonlinear residual and independent neutron and thermal budgets.
- Agreement of the final nonlinear solution with the separately implemented Stage 8 Picard coupled equations and Stage 7 full neutron family backward Euler.
- When external heat deposition is zero, equality to the INDEPENDENT standalone Stage 8 thermal two-node solution and independent Stage 7 neutron solution evaluated at final Ts.
- True temperature change alters local nuclear coefficients and the integrated spatial diffusion operator; thermal feedback is not a fictitious standalone reactivity scalar.
- Constant external source response, unmodeled heat retention and source provenance.
- Independent finite-difference Jacobian-vector comparison, positivity/time/mesh limits, nonconvergence failure, and deterministic input immutability.
- Stage 17 temperature/material table numerical regression is separately rerun in dedicated GitHub Actions.

## Physical limitations and next verification gates

The model does NOT use resolved fuel-element radial conduction or physically accurate hydrogen cp(T,P), enthalpy, pressure losses, density feedback and cross-section group collapse. It samples fuel-solid temperature alone as the nuclear temperature; real fuel/moderator/reflector/coolant require independent temperature fields and density-dependent evaluated data. It cannot predict reactor power unless fission/neutron photon energy deposition is externally normalized and validated. Synthetic heating closures in the test suite are NOT evidence of a NERVA reactor's actual thermal response. There is no executed NERVA hot-core OpenMC statepoint or historical NASA experimental comparison associated with this PR.

**Next concrete physical milestone:** apply this coupling only after reconstructing and independently qualifying energy deposition + temperature/density-dependent evaluated nuclear data on matching geometry/mesh; then compare OpenMC+resolved physics against measurements. A mathematically converged feedback system is not a physically validated hot reactor.

No NUKE main changes. Stacked on Stage 17 PR #129. Do not merge independently of upstream prerequisites.
