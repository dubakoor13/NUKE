# FX2-TH / NUKE Stage 13 — adaptive temporal integration of spatial neutron and precursor kinetics

Date: 2026-10-24. Parent: Stage 12 PR #124. Implemented as a separate stacked feature branch, with NO changes to NUKE main reactor/thermal/engine models.

## Purpose
Stage 7 computes the coupled spatial multigroup neutron-flux and delayed-family precursor inventory with implicit backward Euler on a caller-chosen fixed timestep. Stage 12 supplies a genuine conservative axisymmetric r-z finite-volume loss matrix. The Stage 13 numerical controller estimates LOCAL timestep error by comparing a full implicit Euler step h to two successive half steps h/2; it accepts only the fine two-half-step endpoint when the local flux and precursor errors are within explicit caller-provided tolerances.

The controller neither constructs reactor nuclear parameters nor changes geometry, cross sections, thermal feedback, helium/hydrogen material coefficients, reactivity, eigenvalue, neutron source or decay constants. The inputs are still fully external, and the tests remain synthetic.

## Governing equations (from Stage 7)
 M d(phi)/dt = -A phi + F_prompt(phi) + sum_f chi_delayed[f] lambda[f] C_f + V q
 d(C_f)/dt   = V sum_g beta[g,f] nuSigmaF[g] phi[g] - lambda[f] C_f

where M[c,g]=V[c]/v_g, A is the already cell-integrated conservative diffusion operator, and C_f is a cell-INTEGRATED family precursor inventory. Family energy emission spectra and incident-group delayed fractions remain explicit. At each attempted interval the coefficients A, M, F, beta, lambda, chi and external source q are held fixed.

## Algorithm
For candidate dt:
  (1) Uc = BE(Uold,dt)
  (2) Uh = BE(Uold,dt/2)
  (3) Uf = BE(Uh,dt/2)
  (4) E = max over all flux and precursor components:
      abs(Uc - Uf) / (atol + rtol*max(abs(Uc),abs(Uf))).
  (5) Also include differences in integrated neutron population sum(M*phi) and total integrated precursor inventory, with corresponding absolute tolerances.
  (6) Reject if E>1; attempt smaller dt, respecting min_dt and rejection/attempt budgets. Accept Uf otherwise. Report per-half-step neutron/precursor and global balance residuals, without Richardson extrapolation.

Because backward Euler is FIRST-ORDER globally, this step-doubling scheme estimates local temporal error but DOES NOT turn the accepted solution into a second-order ODE integrator. Error control DOES NOT establish a rigorous global-error bound or physical model fidelity. Positivity is preserved by retaining accepted implicit solutions rather than subtracting solution estimates.

## Numerical QA
The test module independently uses scipy.linalg.expm on a densely assembled *full neutron+precursor* coupled ODE generator (NOT the implicit solver's Schur complement), including constant-source augmentation. It checks:
- exactly stationary critical family inventories for arbitrary changes in timestep;
- tight-versus-loose tolerance global accuracy improvements;
- full two-group point/spatial ODE and spatially heterogeneous r-z core benchmark;
- time-dependent synthetic external source;
- rejection/recovery after oversized timestep;
- no silent success when min_dt, max_steps or source/state guards fail;
- exact requested time endpoint and reproducibility;
- family-specific birth spectra, positivity, source provenance and input state immutability;
- global neutron and precursor balance residuals of every accepted half step.

Run:
    python -m ntr.nerva.validate_fx2th_rz_family_diffusion
    python -m ntr.nerva.validate_fx2th_adaptive_core_kinetics

CI also reruns Stage 12 cylindrical FV geometry and kinetics tests.

## Limitations and pending work
- Coefficients and external source are frozen on this integration interval. Temperature feedback, control commands, flux-dependent cross sections and stepwise discontinuities MUST be split into separate externally managed intervals, and each interval independently qualified.
- No real NERVA reactor nuclear data, energy-group condensation, physical prompt lifetime, historical hot-core criticality measurements, actual OpenMC statepoint or thermal material temperatures were used.
- This is not the original historical FX2-TH improved quasistatic scheme, a neutron transport solver or a verified nuclear reactor stability/accident analysis.
- Temporal refinement against a dense synthetic operator verifies mathematical time integration only. Independent spatial mesh convergence, transport comparison and qualified cross section/thermal property data remain open.

Do not merge this stacked branch alone without parent PR #124 and the preceding family contract/matrix integration commits. 
