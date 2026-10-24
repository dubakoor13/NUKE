# FX2-TH numerical reconstruction — Stage 2: 2D multigroup diffusion

**Date:** 2026-10-24. **Status:** complete *synthetic numerical* benchmark module; not a historical FX2-TH port, neutron-transport simulation, or NERVA physical validation.

## Source and scope

- R. A. Shober, T. A. Daly, D. R. Ferguson, *FX2-TH: A Two-Dimensional Nuclear Reactor Kinetics Code with Thermal-Hydraulic Feedback*, Argonne ANL-78-97 (1978), original repository PDF \`aaa neuronic code.pdf\`.
- Section II.B.1–5: steady multigroup diffusion equation, cell-integrated finite differences, outer fission-source iteration and transpose adjoint.
- The historical code used Chebyshev-accelerated fission iteration, line over-relaxation, and supported x, r, x-y, r-z, theta-r and triangular coordinates. **This milestone implements none of those historical implementations or alternative geometries**. Only a clean-room, 2D Cartesian finite-volume analogue is implemented using SciPy sparse LU and power iterations.
- Original FORTRAN-IV source was **not** present in the report and has **not** been recovered. Its separate NEA package is NESC0862; check access and licensing before obtaining or redistributing.

## Equations and discretization

The steady-state benchmark solves the *integrated cell* matrices:

\[
A\phi=\frac{1}{k_\mathrm{eff}}F\phi,\qquad
A^T\psi=\frac{1}{k_\mathrm{eff}}F^T\psi.
\]

For cell \((i,j)\) and neutron energy group \(g\), diffusion loss includes leakage, absorption and transfer out to other groups minus transfer in from other groups. The fission operator is \(\chi_g\sum_{g'}\nu\Sigma_{f,g'}\phi_{g'}\). \(\chi\) is normalized separately in each cell. \(\Sigma_s[g_\mathrm{from},g_\mathrm{to}]\) has **zero diagonal**.

Flux unknowns are located at rectangular cell centers. The integrated neutron current across a shared face uses symmetric transmissibility:

\[
T_{i,j,g}=\frac{\mathcal A_\mathrm{face}}{
 \Delta_i/(2D_{i,g})+\Delta_j/(2D_{j,g})},\quad
J_\mathrm{from\ to} = T_{i,j,g}(\phi_i-\phi_j).
\]

The two adjoining cell equations use equal and opposite face currents, so their contributions cancel in the global neutron balance. Face boundary choices are either \`reflecting\`, zero normal current, or \`zero_flux\`, scalar flux zero at the **boundary face** (not a physical diffusion/vacuum Marshak boundary). Cross sections have inverse meters, D meters, length meters, and computations are per unit out-of-plane depth.

The **reported physical balance** verifies

\[
\sum_\mathrm{cells,groups}\Sigma_{a,g}\phi_gV
+\sum_\mathrm{boundary}J_{\rm leak}
-\frac{1}{k_{\rm eff}}\sum_\mathrm{cells,groups}
 \chi_g\sum_{g'}\nu\Sigma_{f,g'}\phi_{g'}V=0.
\]

No neutron transport/angular effects are calculated. \`forward\` is normalized to unit integrated scalar flux, and \`adjoint\` is normalized such that \(\psi^T F\phi=1\). This scaling is only an algebraic gauge and **not** source strength/power normalization.

## Implemented files

- \`ntr/nerva/fx2th_diffusion.py\`: validated Cartesian grid; conservative sparse A and F assembly; forward/adjoint \(k_\mathrm{eff}\) eigenproblem with residuals and global neutron balance; subcritical steady fixed-source \((A-F)\phi=qV\).
- \`ntr/nerva/validate_fx2th_diffusion.py\`: 11 reference-free synthetic mathematical tests: reflecting homogeneous analytic solution, analytic Dirichlet single cell, heterogeneous-interface harmonic coefficient, multigroup dense generalized eigencomparison, adjoint transpose and biorthogonality, 2D bare slab sine-mode mesh convergence, subcritical fixed source and supercritical rejection, mixed boundaries, invalid-data rejection and mesh rotation symmetry.
- \`.github/workflows/fx2th-diffusion.yml\`: independent GitHub Actions numerical test job.

Run with \`python -m ntr.nerva.validate_fx2th_diffusion\`. **Local result 11/11 passing**. Require GitHub CI before merge.

## Important limitations / pending physics

1. This is the **steady diffusion base**, not the two-dimensional time-dependent neutron diffusion and precursor PDEs of FX2-TH. It does **not** yet connect to \`fx2th_kinetics.py\` in the separate Stage 1 pull request.
2. No genuine material cross-section preparation, temperature feedback, historical design constants, validated neutron generation times, kinetic adjoint importance or OpenMC scattering/fission homogenization. Zero/negative absorption, zero-fission models, anisotropic diffusion and non-Cartesian geometry are currently unsupported.
3. No historical FX2-TH benchmarks or NERVA-specific cases were run, and no independently evaluated nuclear data used. A physically meaningful \(k_\mathrm{eff}\) for NERVA cannot be inferred from the synthetic example values.
4. The mesh uses a unit depth in z; do not interpret this as an actual 3D reactor volume without specifying a real geometry and out-of-plane leakage/buckling.
5. Tests verify mathematics for the supplied *toy* medium; they cannot establish transport approximation fidelity, boundary applicability, material temperature dependence or reactor power.

## Next physics-first iteration

- Stacked integration with Stage 1 reduced point kinetics and stable adjoint normalizations, including genuine shape-update operations and spatial precursor balances, **without** manufacturing effective reactivity from a \(k_\mathrm{eff}\) snapshot.
- Independently compare Cartesian multigroup diffusion to transport-consistent reference/group libraries in controlled non-reactor benchmarks.
- Once formal nuclear-data provenance, temperature dependence and neutron physics agree with independent measurements, apply in the documented NERVA thermal-feedback architecture.
