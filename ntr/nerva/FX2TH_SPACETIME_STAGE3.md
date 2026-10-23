# FX2-TH Stage 3: space-time multigroup diffusion and cellwise precursors

**Date:** 2026-10-24  
**Status:** Synthetic numerical implementation with 14 locally passing regression tests. **NOT** original historical FX2-TH code, a complete IQS solver, a validated NERVA core model or real OpenMC kinetics.

## Primary source

Shober, Daly & Ferguson, *FX2-TH: A Two-Dimensional Nuclear Reactor Kinetics Code with Thermal-Hydraulic Feedback* (Argonne ANL-78-97, October 1978), checked in at NUKE root as \`aaa neuronic code.pdf\`. Especially printed Chapters II.A–C (PDF pages 18–39): cell-centered multigroup diffusion, forward and adjoint flux, the amplitude/shape factorization, pointwise delayed neutron precursors, and distinct shape versus reactivity time steps. The complete historical FORTRAN-IV source decks are **not included** in the report; they were distributed separately. NEA package NESC0862 is an archival potential source, subject to licence and availability.

## Governing semi-discrete equations

Let \(\phi_{k,g}\) be cell-averaged neutron scalar flux on rectangular cell \(k\) and energy group \(g\), \(V_k\) the 2D cell area times unit out-of-plane depth, \(v_g\) group speed, and \(C_{k,\ell}\) the **cell-integrated** delayed-neutron precursor inventory. Let \(\beta_\ell\) and \(\lambda_\ell\) be externally provided effective delayed fractions and decay constants. The Stage 2 operators are \(A\): integrated diffusion, absorption, outscatter minus inscatter; \(F\): integrated total fission-neutron production.

\[
 M\dot \phi = -A\phi + (1-\beta)F\phi + B C + Q,\quad
 \dot C_{k,\ell} = \beta_\ell V_k\sum_g \nu\Sigma_{f,k,g}\phi_{k,g}-\lambda_\ell C_{k,\ell},\quad M_{k,g}=V_k/v_g.
\]

Here \(B C\) is emitted in the same local-group spectrum as prompt fission, \(\chi_{k,g}\sum_\ell \lambda_\ell C_{k,\ell}\). That *equal prompt/delayed emission spectrum assumption* must be reconsidered before physical application. \(Q=q_{k,g}V_k\) is external neutrons/s integrated per cell/group. The model assumes immobile precursors and no flowing-precursor transport.

With \(\Delta t\) fixed, use fully implicit backward Euler and eliminate precursor inventories:
\[
 C_{k,\ell}^{n+1}=\frac{C_{k,\ell}^{n}+\Delta t\beta_\ell V_k\sum_g \nu\Sigma_{f,k,g}\phi_{k,g}^{n+1}}{1+\Delta t\lambda_\ell}.
\]

Resulting sparse linear system:
\[
 \left[\frac M{\Delta t}+A-\kappa(\Delta t) F\right]\phi^{n+1}
  =\frac M{\Delta t}\phi^n+
  \chi\sum_\ell\frac{\lambda_\ell C_\ell^n}{1+\Delta t\lambda_\ell}+Q,
\]
\[
 \kappa=1-\sum_\ell\beta_\ell+\sum_\ell\frac{\Delta t\beta_\ell\lambda_\ell}{1+\Delta t\lambda_\ell}.
\]

The code verifies each cell/group balance and the global neutron conservation identity, including precursor inventory changes and zero-flux face leakage. Strongly positive reactivity or very large steps can invalidate the positive solution; the solver **raises** and requires timestep refinement. Small algebraic residual is **not** timestep accuracy.

## Modules

- \`ntr/nerva/fx2th_spacetime.py\`: sparse, full 2D cell-resolved group diffusion and local delayed-precursor backward Euler; fixed-source input; prescribed coefficients changing at each timestep **on the same mesh**; explicit neutron inventory and equation residuals.
- \`ntr/nerva/validate_fx2th_spacetime.py\`: 14 synthetic checks: critical steady state, analytic prompt-only decay, comparison to independent 3x3 exact matrix exponential, spatial precursor consistency, neutron leakage/global balance, long-time subcritical external source, heterogeneous multigroup scatter, adjoint-weighted normalization, exact full spatial refresh, frozen-shape uniform benchmark, spatial error detection, externally varying coefficients, mesh-remapping guard and invalid inputs.
- \`.github/workflows/fx2th-spacetime.yml\`: isolated numerical CI.
- Dependency: \`fx2th_diffusion.py\` from *Stage 2 PR #112*. This is an intentionally **stacked branch/PR**, not based on \`main\`.

## Two-rate prototype is NOT a completed IQS method

The convenience mode \`run_space_time(..., refresh_every=N, adjoint_weight=w)\` **freezes the shape** between full space-time solves. It projects the amplitude with \(w^T M \phi\), retains cellwise precursor changes, and reports nonzero spatial-equation residual on reduced steps. It is a fast surrogate to quantify spatial-dynamics sensitivity, not the full improved quasistatic **shape PDE** solver in ANL-78-97, and it does not reconstruct the FX2-TH 1978 adaptive step-control laws. For \`refresh_every=1\`, it performs only full spatial solves.

## Unimplemented physical and numerical gates

1. Genuine FX2-TH improved-quasistatic shape equation: temporally varying shape with normalized adjoint weighting, compatible time-varying precursor histories and kinetic parameters, nonlinear shape-step convergence, independently controlled error bounds.
2. Spatial energy dependence of effective neutron velocities, **physical** separate prompt/delayed spectra, nonuniform delayed fractions, validated nuclear-data collapse and adjoint provenance.
3. Nuclear cross-section temperature response, NERVA graphite fuel and hydrogen density/thermal coupling, source-normalized reactor power, feedback kinetics, uncertainties.
4. Non-Cartesian real-core geometry and qualified diffusion-to-transport comparison; no 2D analogue can fully substitute for 3D Monte Carlo OpenMC in leakage/streaming regions.
5. Independent historical FX2-TH benchmark runs require the actual input nuclear data and complete numerical output, and NERVA application requires separate qualified reactor data.
6. Stage 1 reduced point-kinetics PR #111 is independent, *not* imported into this code. To integrate the stacks, explicitly reconcile the corresponding amplitude and precursor time-base contracts.

**Run:** \`python -m ntr.nerva.validate_fx2th_spacetime\` (NumPy+SciPy). Passing tests are a numerical construction check only.
