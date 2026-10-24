# FX2-TH / NUKE Stage 14 — second-order L-stable spatial neutron / precursor kinetics

**24 October 2026** · Parent: Stage 13 PR #125 · Scope: core numerical equations ONLY.

## Motivation

Stage 13 provided embedded local-error control around first-order backward Euler. Stage 14 adds a separate, genuinely **second-order, L-stable** stiff ODE scheme operating on the exact Stage 7 neutron and delayed-family precursor unknowns, including Stage 12 cylindrical r-z multigroup finite-volume loss operators. This is not an original FX2-TH FORTRAN port, OpenMC transport calculation, NASA/NERVA physical benchmark, engine code or fuel-temperature model.

## Coupled sparse generator

The unknown is y=[phi[c,g],C[c,f]], where C is CELL-INTEGRATED precursor inventory, M[c,g]=V[c]/v[g], and A is the externally supplied cell-INTEGRATED loss/scattering/leakage matrix.

Equations:
 M dphi/dt = -A phi + F_prompt(phi) + E C + V q
 dC/dt = B phi - L C.

F_prompt is the cell-integrated prompt fission birth operator, B is the precursor family production operator (V*beta*nuSigmaF) and E is the emission operator (chi_delayed*lambda). The full generator has sparse blocks:

 G = [[M^-1(-A+F_prompt), M^-1 E],
      [B,                -L     ]].

The fixed source vector is [M^-1 V q,0]. All material parameters, birth spectra and mesh volumes come from a Stage 7 SpatialFamilyData. Source coefficients and group speeds are NOT inferred or tuned here. Exact equivalence of G to an independently assembled Stage 7 dense ODE reference is tested on both toy and cylindrical r-z spatial meshes.

## Second-order L-stable SDIRK method

For gamma=1-1/sqrt(2), the two-stage stiffly accurate singly diagonally implicit Runge–Kutta scheme is:

 (I-h*gamma*G) Y1 = y0 + h*gamma*b
 (I-h*gamma*G) Y2 = y0 + h*(1-gamma)*G*Y1 + h*b
 y_next = Y2.

The same sparse LU factorization is reused for both stages. Stability function R(z)=[1+(1-2gamma)z]/(1-gamma*z)^2 tends to zero as z goes to negative infinity (L-stable). The scheme is SECOND ORDER in time with fixed coefficients; for very stiff negative modes even L-stable methods may transiently overshoot through zero. This implementation explicitly REJECTS negative flux/precursor stage outputs rather than silently clipping physical inventories.

The independently audited total integrated population is N_tot = sum(M phi)+sum(C). The stage-weighted balance is:
 N_new-N_old = h*((1-gamma)*dN(Y1)/dt +gamma*dN(Y2)/dt),
where each derivative is obtained from the complete sparse generator and source.

## Adaptive step-doubling

Compare SDIRK2 of size h with two SDIRK2 steps h/2. The accepted fine local error estimate is (fine-coarse)/(2^2-1). The full neutron-flux and precursor arrays are checked against independent absolute+relative tolerances; their integrated neutron and precursor populations are also checked. All six implicit stage solves per candidate step share a full-step LU and a reusable half-step LU for both half steps. Accept ONLY the fine state, not Richardson extrapolation which can violate positivity; reject and decrease h on relative local error > 1, stage negativity, or failed conservative balance. Step growth/shrinkage exponent is 1/3, reflecting second-order global accuracy and O(h^3) local truncation.

This is only a LOCAL truncation-error estimate, not a certified global error bound, neutron-transport error estimate or material-physics validation.

## Independent verification plan

- Exact sparse generator versus *independently assembled dense Stage 7 full coupled generator*, for synthetic 2-cell and r-z 2-group systems.
- Stationary k-effective 1 synthetic one-cell delayed-family equilibrium.
- Observe approximately factor-four accuracy improvement per timestep halving versus scipy.linalg.expm full-state exact solutions; compare equal step count against first-order backward Euler.
- Analytical constant-source reference by an augmented dense matrix exponential.
- Verify neutron/precursor population and internal stage linear residuals.
- Test nonidentical delayed-family emission spectra, cylindrical r-z heterogeneous spatial transient, stiff-mode L-stability and physical positivity guards.
- Test adaptive rejection and recovery, tolerance-tightening global-error reduction, reproducibility, exact endpoint and fail-closed invalid-input behavior.

Run:
    python -m ntr.nerva.validate_fx2th_rz_family_diffusion
    python -m ntr.nerva.validate_fx2th_sdirk2_neutron_kinetics

## Physics still missing

The material values in all tests are SYNTHETIC. Stage 14 is mathematical time integration, NOT independent prediction of actual NERVA neutron transport, reactivity, control drum worth, prompt neutron generation time or hot-core reactor power. No NASA critical test, evaluated multigroup temperature-dependent cross-section library or real OpenMC statepoint has been verified through this integration. Real spatial/thermal fidelity and nuclear cross-section provenance remain unresolved. As with Stage 13, the solver intentionally freezes all nuclear coefficients on each integration interval: physical material-state evolution must be handled through an externally qualified coupled material update, not silently interpolated.

No NUKE main branch edits; stacked on Stage 13 PR #125. Do not merge this PR without parents.
