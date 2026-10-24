# NERVA CORE BUNDLE — selective diffusion integration and independent audit
**Date:** 2026-10-24
**Branch:** `feature/nerva-bundle-axisymmetric-diffusion-20261024`

## Source provenance
- Source archive: [NERVA-CORE-BUNDLE.zip](../../NERVA-CORE-BUNDLE.zip), 1,619,915 bytes, SHA-256 `ede9990ddeac28c2537699b4af838429d4b0e7936686bc9b2e15f292b569bb28`.
- Original archive module path: `NERVA-CORE-BUNDLE/ntrv2/nerva/`, relocated here to `ntr/nerva/`. The module namespace mismatch is a packaging error in the archive.
- Forensic comparison against original `main` in PR #115: 287 files identical, 82 absent, 1 differing; **no mass extraction** and no overwrites. Exactly four selected existing archive files are imported here: `diffusion.py`, `diffusion_2d.py`, `validate_diffusion.py`, `validate_diffusion_2d.py`.
- The unmodified archive remains on `main`; this branch deliberately contains numerical amendments to the imported source.

## Independent mathematical audit findings
1. **Outer radial zero-flux boundary conductance**: The original cell-centered solver treated the Dirichlet face as if it were one full radial mesh interval away from the outer cell's center. Face distance is **dr/2**, requiring twice the neighbor-centre conductance. Added the missing contribution to the final radial diagonal.
2. **Axial end zero-flux conductance**: Both boundary faces lie dz/2 from the first/last axial cell center. Added the corresponding extra diagonal diffusion conductance at each end.
3. **Cylindrical volume integration**: Original reporting used `shape.sum(axis=1)`, which sums radial density without the annular `2*pi*r*dr*dz` volume factor. Corrected radial power fractions and volume-mean radial peaking using `r_i` weights. Annular power shares can peak away from the axis even when local power density peaks at the axis.
4. Updated source validator's radial power assertion so it does not conflate local maximum flux with maximum integrated annular power.

This is **not a change to reactor materials, enrichment, control parameters, or operating conditions**. It is a geometric finite-volume and postprocessing correction.

## Tests
Original bundle test modules were executed successfully in a clean GitHub Actions environment on the unchanged archive (`validate_diffusion` and `validate_diffusion_2d`). After importing and correcting, this branch adds an **independent** numerical test based on the cylinder's Bessel eigenfunction:
`J0(alpha * r/R)`, where `J0(alpha)=0`; integrated annular power is proportional to
`[r*J1(alpha*r/R)]_{r_a}^{r_b}`. Axial integrated power follows a sine half-wave. This checks radial **power volume**, axial shape, radial peak/mean and mesh-eigenvalue convergence independently of the archived test oracles.

Run:
```bash
python -m ntr.nerva.validate_diffusion
python -m ntr.nerva.validate_diffusion_2d
python -m ntr.nerva.validate_bundle_rz_geometry_independent
```

## Physical limitations
All verification uses synthetic homogeneous diffusion data. The bare cylindrical diffusion benchmark is **not a historical NERVA model**, not OpenMC continuous-energy neutron transport, and not thermal feedback. The archive's own `MANIFEST.md` records only 3 of 8 complete acceptance gates and no offline real OpenMC neutron transport. This PR does not change those gates or claim reactor validation.

## Next integration
- Independently audit `adjoint.py` and `delayed_parameters.py`; do not substitute textbook kinetic constants for NERVA measurements.
- Compare the r-z cylindrical diffusion solver to the Cartesian FX2-TH module on mathematically equivalent limiting configurations, without equating distinct boundary/geometry assumptions.
- Keep mock transport results distinctly labeled synthetic, then verify cross-section qualification and observed full-core evidence separately before enabling any physical coupling.
