# FX2-TH Stage 4: two-rate spatial-shape kinetics

**Date:** 2026-10-24. **Scope:** Independent numerical reconstruction inspired by ANL-78-97 II.C. **Not** the original FX2-TH FORTRAN-IV source, a validated NERVA reactor model, or an OpenMC transient run.

## Reference and architecture

Shober, Daly, and Ferguson, *FX2-TH: A Two-Dimensional Nuclear Reactor Kinetics Code with Thermal-Hydraulic Feedback*, ANL-78-97 (1978), Chapter II, especially the amplitude/shape factorization, spatial flux equation, adjoint weighting, and separate shape/reactivity time steps. The complete original source is **not in the PDF**. Original legacy code is catalogued as NEA **NESC0862**.

This branch is **stacked on Stage 3** (PR #113), which is stacked on Stage 2 (PR #112). Stage 1 reduced point kinetics is in the independent PR #111. No change to the real OpenMC solver.

The spatial model is the finite-volume 2D Cartesian multigroup diffusion operator from Stage 2. Let \(A\) be its neutron loss/scattering operator, \(F\) its fission-production operator, \(M\) cell neutron inventory (cell volume divided by group speed), \(E(C)\) the cellwise delayed source and \(Q\) an externally prescribed source.

\[
M\dot\phi=H\phi+E(C)+Q,\quad H=-A+(1-\beta)F,
\]
\[
\dot C_l=\beta_l P(\phi)-\lambda_l C_l,
\quad \phi=n(t)s(t),\quad n=w^TM\phi,\quad w^TMs=1.
\]

The **amplitude** advances on each microstep with a projected implicit neutron balance. Delayed precursors advance **locally, in each spatial cell**. An endpoint spatial shape is solved every macro interval, using **linear shape interpolation**, sparse linear equations, and repeated amplitude/precursor/shape correction.

The **macro endpoint** satisfies the integrated spatial diffusion equation:

\[
M(n_\text{end}s_\text{end}-n_\text{start}s_\text{start})
\simeq\sum_j\Delta t\left[
H(n_js_j)+E(C_j)+Q\right].
\]

The algorithm reports this integrated spatial residual, adjoint gauge residual, integrated physical neutron balance and local precursor equation residual. Nonphysical or nonconvergent states cause explicit failure rather than artificial clipping of the physical solution.

**Difference from historical improved-quasistatic FX2-TH:** We use a self-consistent finite-volume integral shape update with linearly interpolated microstep shapes, *not* the original FX2-TH specific adaptive error controller, shape solver, delayed-neutron nuclear-data procedure or Fortran code. For one microstep it reduces to the fully coupled backward-Euler spatial calculation, within nonlinear convergence tolerance. More microsteps between shape solves deliberately introduce a measurable error.

## Verification

Run:

\`\`\`bash
python -m ntr.nerva.validate_fx2th_iqs
\`\`\`

13 tests use entirely synthetic 1- and 2-group Cartesian coefficients: one-step equivalence to the full implicit PDE, stable critical equilibrium, delayed-neutron cell balances, shape norm preservation, integrated neutron balances, source response, spatially heterogeneous media, time-step refinement, analytic prompt-only one-cell decay and invalid input checks.

An illustrative *2x2 toy* case, four 0.1-s time steps, nonuniform initial flux, with 4-step full backward Euler as the reference:

| Integration | Relative error in final spatial flux |
|---|---:|
| Full spatial solution every microstep | Reference (0%) |
| Evolving shape every microstep | Numerical tolerance, < 0.000001% |
| Evolving shape every two microsteps | ~0.713% |
| Evolving shape every four microsteps | ~4.374% |
| Frozen shape throughout | ~44.313% |

This is a numerical-method test, **not a measured speed-up, physical validation, reactor transient, NERVA estimate or historical FX2-TH output**. Shape-update count is not automatically optimal; a physical error controller still needs to be developed.

## Known limitations

1. Cross sections, diffusion coefficients, speeds, delayed fractions, decay constants and adjoint weights must be supplied by the caller; there is no OpenMC nuclear-data reader or temperature-dependent law.
2. Material coefficients and external source must remain unchanged during each macro interval. This milestone does not solve material/thermal feedback or adaptive restart.
3. Only 2D Cartesian diffusion, reflecting or zero-scalar-flux faces; these **are not full vacuum transport boundary conditions**.
4. Numerical stability/consistency is established on small synthetic cases only; the code does not prove correct transport physics, Monte Carlo stochastic propagation, neutron generation time or safety of reactor operation.
5. The original FX2-TH time-step adaptation, pointwise precursor cross-section updates and historical Fortran computational details remain unimplemented.

**Next physics tasks:** adaptive shape-time-step estimator with dense-reference checks; physically traceable neutron data and adjoint/forward weighting; independently validated NERVA fuel and hydrogen thermal feedback; OpenMC transport-to-kinetics coupling with source, mesh, uncertainty and energy-deposition conservation; comparison against independent historical reactor transient data.
