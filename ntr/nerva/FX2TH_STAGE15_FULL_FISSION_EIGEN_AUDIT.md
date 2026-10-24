# FX2-TH / NUKE Stage 15 — full family-resolved equilibrium fission operator

24 October 2026. Source branch: feature/fx2th-full-fission-family-eigen-stage15-20261024. Parent: Stage 14 PR #126. Physics/numerics only; no reactor design parameters or NUKE main changes.

## Real mathematical gap closed

Stage 11 and Stage 12 used a *single total emitted neutron fission spectrum* for the static forward/adjoint eigenvalue calculation A phi = F phi/k. Their explicit reference guards rejected otherwise valid Stage-7 neutron-precursor models when prompt and delayed families used different birth-energy spectra: the stationary total fission operator was NOT the original F. Stage 15 calculates the actual total family-split fission operator and its own eigenmode, so it does not silently substitute the wrong critical flux/adjoint or force identical emitted neutron spectra.

This is an independently assembled mathematical multigroup diffusion/kinetics operator, not historical FX2-TH FORTRAN, evaluated nuclear data, continuous-energy transport, or a physically validated NERVA fuel model.

## Full fission operator — incident and emitted group kept distinct

For cell c, incident energy group h and emitted group g, with the nuclear data already present in a Stage-7 SpatialFamilyData:

F_prompt[c,g,h] = V[c] chi_prompt[c,g] nuSigmaF[c,h] (1 - sum_f beta[c,h,f]).
F_delayed_f[c,g,h] = V[c] chi_delayed[c,f,g] nuSigmaF[c,h] beta[c,h,f].
F_total = F_prompt + sum_f F_delayed_f.

The neutron loss/leakage/scattering matrix A is unchanged, **already cell integrated**, and built by the existing Stage-12 conservative r-z (or the Stage-11 Cartesian) finite-volume solver. No extra V is applied to A or to F. The eigenproblem is A phi = (F_total phi)/k. The adjoint solves A.T psi = (F_total.T psi)/k. Normalize psi.T F_total phi=1 and phi to unit neutron population sum(M phi)=1 with M[c,g]=V[c]/v_g. Those normalizations are numerical gauges and do NOT imply any physical reactor power, flux magnitude or neutron source rate.

At steady state, the family-specific precursor inventory satisfies:
lambda[f] C[c,f] = V[c] sum_h beta[c,h,f] nuSigmaF[c,h] phi[c,h].
Thus delayed emission in the time-dependent equation is *exactly* sum_f F_delayed_f phi. Its stationary eigenproblem contains different delayed spectra without approximation. Family decay rates DO NOT change k at steady state, but DO change the individual precursor concentrations and transient response.

Adjoint-weighted family effective beta:
beta_eff,f = psi.T F_delayed_f phi / (psi.T F_total phi).
The corresponding unweighted emitted delayed production fraction is sum(F_delayed_f phi)/sum(F_total phi). These must not be conflated. The importance-weighted neutron generation-time parameter (psi.T M phi)/(psi.T F_total phi) is reported for this mathematical reference; it is not a measured prompt neutron lifetime.

## Verification

The new suite independently checks:
- Prompt/family matrices against separate manual spatial-and-group summation.
- Integral fission neutron birth conservation per unit time and exactly one cell-volume factor.
- Recovery of the original Stage-12 total fission matrix when all prompt/family emitted spectra coincide; reduction to prompt-only F when delayed production is zero.
- Genuine difference in the generalized eigenproblem when families emit at different neutron energies.
- Forward and adjoint k against SciPy's *dense generalized eigensolver* using the independently assembled original conservative A.
- Neutron adjoint equation, psi.T F phi=1, family beta_eff and unweighted delayed fractions independently contracted from the sparse matrices.
- Invariance of stationary eigenpair to family decay constants, while stationary family precursor inventories change.
- Creation of a critical state by uniformly scaling synthetic fission strength from computed k and checking **BOTH Stage-7 backward Euler and Stage-14 SDIRK2** preserve the prompt/delayed equilibrium for distinct spectra and varied timestep.
- Existing common-spectrum reference rightly refuses to mislabel its old eigenvector as equilibrium in the distinct-spectrum case.
- Nonuniform r-z annular geometry, mismatched criticality, tolerance/input guards.

CI reruns the original Stage-12 r-z numerical benchmark alongside the new family eigen suite.

## Physical limits that remain
All numerical material data are synthetic. No NERVA isotope-dependent delayed spectra, evaluated-temperature nuclear cross sections, actual fuel/reflector/control geometry, real OpenMC statepoint neutron transport, NASA experimentally observed k_eff, measured power, neutron lifetime or thermal feedback has been provided or validated. The generalized eigenproblem is still diffusion approximation with prescribed cross sections and boundary conditions, not a continuous-energy Boltzmann transport reference.

Next practical core-physics gap: use independently qualified prompt/delayed emission and material cross sections on a matched mesh; compare the eigenpair, beta_eff and time/space transients against converged OpenMC transport with appropriate nuclear-data uncertainty and actual hot-core thermal state. Further algebraic tests are not a substitute for those measurements.

This PR is stacked on Stage 14 PR #126; do not merge alone or overwrite prior branches.
