# FX2-TH / NUKE Stage 7 — spatial precursor families
Date: 2026-10-24. Parent Stage 6 PR #118. Status: synthetic numerical solver only; no NERVA physical validation.

## Scope and mathematical reference
The Argonne ANL-78-97 FX2-TH report describes time-dependent multigroup diffusion with delayed precursor and thermal feedback. This independently developed module models the multigroup flux and cellwise delayed-family precursor inventories. It uses the Stage 6 family data contract for distinct family decay constants and prompt/delayed birth spectra. It is not original FX2-TH FORTRAN, historical IQS source or OpenMC neutron transport.

Unknowns are scalar flux phi[c,g] and cell-integrated precursor inventories C[c,f]. The independently supplied cell-integrated diffusion loss/scattering matrix is A. Neutron time-derivative mass is M[c,g] = cell volume / speed[group]. Cross sections nuSigmaF[c,g] have inverse-length units. Family beta[c,g,f] sums to <1 per incident energy group. Prompt and delayed emitted-group spectra are separately normalized.

Coupled ODE:
M dphi/dt = -A phi + P_prompt(phi) + sum_f chi_delayed[f]*lambda[f]*C[f] + V*q
dC[c,f]/dt = V[c]*sum_g(beta[c,g,f]*nuSigmaF[c,g]*phi[c,g]) - lambda[f]*C[c,f].

At each implicit backward Euler step, C_new is exactly eliminated algebraically from the matrix for phi_new. The delayed fission matrix is weighted by dt*lambda[f]/(1+dt*lambda[f]); known precursor inventories emit with lambda[f]/(1+dt*lambda[f]). All family-specific prompt/delayed birth-energy spectra remain resolved.

## Implemented
- ntr/nerva/fx2th_family_spacetime.py: externally supplied material-state contract, implicit spatial flux and precursor evolution, per-equation and global neutron balance, independent Stage 6 family importance adapter, and external material-state scheduling with fixed mesh.
- ntr/nerva/validate_fx2th_family_spacetime.py: independent dense full-state backward Euler matrix oracle, dense matrix-exponential time convergence, analytic stationary one-cell and fixed-source cases, source/precursor conservation, nonuniform spatial and energy family effects, and invalid-input guards.
- .github/workflows/fx2th-spatial-family.yml: runs Stage 6 interface and Stage 7 solver checks in a clean interpreter.

## Important limitations
This is a mathematical benchmark with synthetic parameters. A must be independently supplied with consistent cell-integration conventions; no NERVA reactor geometry, physical fuel mixture, thermophysical properties, evaluated neutron-data libraries or transport-comparison evidence are supplied. No thermal-feedback model was implemented. Externally scheduled data snapshots are NOT a coupled thermal solver. No reactor power, criticality, control setting, prompt lifetime or measured NERVA transient is predicted or validated. Backward Euler is first order and requires temporal convergence studies on physical cases.

## Next gates
Verify continuous-energy-to-multigroup data provenance including delayed family spectra and temperature dependence; qualify the external diffusion loss operator against real transport; then implement and validate energy-conservative core fuel/flow/thermal feedback on specified nonweapon benchmarks. Do not count synthetic numerical tests as reactor validation.
