# FX2-TH / ANL-78-97: source audit and NUKE numerical milestone 1

**Date:** 2026-10-24  
**Implementation status:** *synthetic verification kernel added; historical source not recovered; not a validated NERVA transient solver.*

## Documentary sources

- Shober, Daly, Ferguson, Argonne National Laboratory, ANL-78-97, *FX2-TH: A Two-Dimensional Nuclear Reactor Kinetics Code with Thermal-Hydraulic Feedback* (October 1978), https://doi.org/10.2172/6387620, archival report https://digital.library.unt.edu/ark:/67531/metadc1207026/
- Checked-in copy: [aaa neuronic code.pdf](../../aaa%20neuronic%20code.pdf), **382 PDF pages**.
- OECD-NEA Computer Program Service catalogue, **NESC0862 / FX2-TH**: https://www.oecd-nea.org/tools/abstract/search?q=FX2-TH ; recorded FORTRAN-IV, tested at NEADB 1981-05-26.
- NEA CPS licence/request procedure: https://tdb.oecd-nea.org/jcms/pl_25468/the-computer-program-service-cps

The PDF states that source decks for IBM/CDC were provided on **separate tapes** (PDF p. 15). **Appendix A** (PDF p. 135 onward) is descriptions of subroutines, not all source bodies. Appendix B covers input files; Appendix E common blocks; Appendix F sample problem output. The original Fortran package **was not acquired**; do not claim that this numerical kernel is an extracted port, or distribute licensed NEA source decks in a public repository without confirming permission.

## Historical mathematical model, distinct from this milestone

| Historical source | Actual FX2-TH coverage | Current kernel |
| --- | --- | --- |
| PDF pp. 18–25, chapter II.A–B | 2D space/energy **multigroup diffusion**, critical eigenfunctions, adjoint, stationary source iteration | **Not implemented** |
| PDF pp. 26–33, chapter II.C | Point kinetics, adjoint normalization, improved quasistatic temporal amplitude and periodically recomputed spatial shape | **Reduced point-kinetics ODE, shape gauge only** |
| PDF pp. 33–39 | Pointwise precursors, automatic reactivity/shape time-step selection, shape PDE | **Not implemented** |
| PDF pp. 41–60, chapter III | Fuel pin/gap/cladding/coolant thermal feedback | **Not implemented**; original fast-reactor fuel-pin model cannot be substituted for NERVA graphite/hydrogen |
| PDF pp. 61–67, chapter IV | Temperature-dependent cross-section approximations and nonlinear neutronic-thermal iteration | **Not implemented** |

## Implemented

- [fx2th_kinetics.py](fx2th_kinetics.py): stiff reduced delayed-neutron equations for user-supplied **effective** generation time, group fractions, decay constants, reactivity and source. No hard-coded NERVA values, reactivity histories, enrichment, control drum laws, or Doppler coefficients.
- Backward-Euler step with exact *algebraic elimination of precursor variables*; checks nonnegativity and limits prompt-positive timesteps. The positivity guard alone is **not accuracy proof**.
- Adjoint/energy-group inverse-velocity and cell-volume **flux shape normalization** and exact reconstructed flux algebra; it does not generate flux shapes, deduce a kinetic reactivity, or recover absolute reactor power.
- Piecewise-constant coefficient schedule supplied externally.
- [validate_fx2th_kinetics.py](validate_fx2th_kinetics.py): 12 synthetic tests of equilibrium, analytical prompt-only decay, independent eigen-decomposition matrix exponential comparison, stiff equation residuals, source equilibrium, substep protection, gauge invariance, shape rejection and input guards.

Model:

\[
\dot n = \frac{\rho-\beta}{\Lambda}n + \sum_i \lambda_i c_i + q,\qquad
\dot c_i = \frac{\beta_i}{\Lambda}n-\lambda_i c_i.
\]

This is the **standard point-kinetics reduction** of the FX2-TH time-dependent equations, not the historical full IQS algorithm. The implicit elimination denominator is

\[
D=1-\Delta t\,\rho/\Lambda+
\sum_i \frac{\Delta t\,\beta_i}{\Lambda(1+\Delta t\lambda_i)}.
\]

Shape normalization:

\[
A=\sum_{cell,g}W_{cell,g}v_g^{-1}\phi_{cell,g}V_{cell},
\quad S=\phi/A,\quad \sum Wv^{-1}SV=1.
\]

The adjoint/inverse-speed normalization comes from the report's chapter II.C, **but the required transport or diffusion adjoint must be supplied separately and verified**. A normalized array does not constitute a genuine transport calculation.

Run:

\`\`\`bash
python -m ntr.nerva.validate_fx2th_kinetics
\`\`\`

Initial local independent implementation: **12/12 tests passed**. The repository branch should run its own CI before merging. Synthetic tests do *not* validate NERVA physics or thermal feedback.

## Next physics and numerical gates

1. Find or request original FX2-TH archive **NESC0862** via NEA, verify licence and source-tree completeness. Until then source-faithful conversion is not possible.
2. Develop an independent, benchmarked 2D diffusion operator and multigroup source/eigenvalue solvers with real & adjoint flux; make geometric boundary assumptions explicit.
3. Add truly coupled IQS *shape* PDE updates, adaptive shape/reactivity steps, and spatial delayed-neutron precursor histories. Current code does not implement these steps.
4. Validate collapsed multigroup data and sensitivities independently; no simple reactivity inference from arbitrary OpenMC eigenvalue flux tallies.
5. Add NERVA-appropriate core solids, channel-hydrogen transport and thermodynamic feedback with traceable materials and temperatures, while keeping existing OpenMC models unaffected.
6. Qualify numerical convergence, independent external benchmarks, and energy/reactivity uncertainty before coupling to NESS/ELES.

**No historic FX2-TH, NERVA, or OpenMC transient validation is claimed.**
