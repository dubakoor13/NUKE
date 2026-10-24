# NUKE / FX2-TH Stage 17 — spatial temperature-indexed multigroup nuclear material tables

Date: 24 October 2026. Parent: Stage 16 PR #128. **Numerical interpolation/verification only; no evaluated NERVA material coefficient set is supplied.**

## Existing physical evidence is not silently replaced

The repository's existing `core_material_temperature_qualification.py` and `executed_neutron_temperature_energy_grid.py` inspect *executed root-reachable OpenMC materials*, isotope temperature grids and evaluated neutron-data availability. Those modules are read-only evidence gates, not providers of multigroup diffusion-group nuclear reaction coefficients. An evaluated microscopic nuclear dataset at a temperature is not automatically equivalent to a physically qualified, temperature-dependent, region-homogenized multigroup diffusion/fission table. No neutron spectral condensation, fuel composition, nuclear criticality calibration, or temperature-derivative data are invented here.

Stage 17 supplies a **fixed-mesh explicit multigroup table interface** so that independently qualified data can eventually be connected to the Stage 12/15/16 conservative core diffusion/eigen/adjoint calculations.

## Table contract and physical units

`RZTemperatureTables` contains at least two strictly increasing absolute-temperature knot values in Kelvin, one consistent cylindrical r-z mesh (all knot slices have the identical spatial and energy group grid), and a strictly increasing energy-group boundary array in eV. Values per knot, spatial cell and incident/emitted group:

- `D[c,g]`: diffusion coefficient in meters, strictly positive.
- `Sigma_a[c,g]`: macroscopic absorption coefficient in inverse meters, strictly positive.
- `Sigma_s[c,g_incident,g_emitted]`: macroscopic out-scattering in inverse meters, nonnegative and zero group self-scattering in this discretization.
- `nuSigma_f[c,g_incident]`: total-neutron fission source coefficient in inverse meters, nonnegative, with nonzero total fission strength at every knot.
- `chi_prompt[c,g_emitted]`: normalized prompt neutron birth spectrum.
- `beta[c,g_incident,family]`: family-delayed neutron fractions, nonnegative with total delayed fraction below one.
- `chi_delayed[c,family,g_emitted]`: individually normalized delayed birth-energy spectra.
- Group neutron speeds in meters/second, and delayed family decay constants in inverse seconds; fixed on this table grid.

The table's `source_kind` is explicitly either `synthetic` or `externally-tabulated`. The latter is a **description of provenance, never a physical certification**. An externally declared 64-character nuclear manifest SHA-256 is preserved as an identifier but this module does not read that manifest or authenticate the evaluated nuclear-data files.

## Spatial interpolation and analytic derivatives

For a temperature T in one cell, select the bracketing pair of supplied knots T0,T1. Each coefficient and birth fraction/spectrum is interpolated by the **linear convex combination**

    property(T) = (1-w)*property(T0) + w*property(T1)
    w = (T-T0)/(T1-T0).

Different spatial cells may have different temperatures and bracket different input knot intervals. No extrapolation beyond the outermost table nodes. Every field's derivative at non-knot temperature is

    d property/dT = (property(T1)-property(T0))/(T1-T0).

Input spectra are normalized at each knot; a convex combination stays normalized. Every emitted-energy spectrum derivative consequently sums to zero over emitted groups; beta cannot become negative or exceed its allowed group total between valid endpoints. Fission source/birth closure is recomputed using the full Stage 15 prompt + family-specific delayed matrices, with exactly one physical cell-volume integration.

Interior knots have piecewise derivatives that are generally not unique. The state value can be sampled at any knot, but an eigenvalue derivative at a knot with nonzero dT/dtheta is **refused**, rather than assigning an unannounced left/right derivative.

The Stage-17 `evaluate_rz_temperature_direction` takes an explicitly supplied signed vector dT_cell/dtheta in K per unit external parameter and evaluates

    d ln(k)/dtheta = psi.T (dF/dtheta - k*dA/dtheta) phi / (psi.T F phi),

by feeding analytical tabular slopes and the chosen temperature direction to the **actual Stage 16** conservative annular leakage and complete family fission derivatives. This is mathematically a temperature-table slope; it is NOT automatically a physical Doppler or hydrogen moderation coefficient. An imposed `dT_cell/dtheta=1` uniformly yields the derivative for a mathematical uniform temperature perturbation *along the supplied tabular material curves*.

Every output Stage 7 nuclear material-state source identifier includes an SHA-256 digest over source identifier, declared manifest hash and the exact spatial temperature map to help reject reuse of stale nuclear operator matrices.

## Independent numerical verification

`validate_fx2th_rz_temperature_tables.py` creates **clearly synthetic** 2-energy-group, 2-family, nonuniform-temperature material slices with different spatial slopes for absorption, diffusion, scattering and nu-fission, including temperature-dependent prompt and family-specific birth spectra.

Tests verify:
- Exact knot values and per-cell independent spatial temperature bracketing.
- Analytical field slopes versus centered difference of independently interpolated states.
- Preservation of neutron birth number, per-family normalizations, group energy order and cylindrical volumes.
- No extrapolation; rejection of malformed nuclear-data tables, beta and spectrum budgets, wrong group edges and invalid temperature meshes.
- Identical source state for identical input and different fingerprint for changed temperature/source manifest.
- **Exact dense generalized eigenvalue recomputation** for uniform temperature change, heterogeneous multi-cell perturbation, and a single local core-cell perturbation. These are independent of the Stage 16 adjoint sensitivity routine.
- Core reactivity `rho=1-1/k` derivative compared with directly recomputed dense `rho(T+eps*dT)` versus `rho(T-eps*dT)`.
- No invented derivative at interior tabular knots; validity of zero-temperature directions.
- Existing Stage 16 adjoint-sensitivity regressions are rerun in dedicated GitHub CI.

## Missing physical qualifications

This branch contains **no real evaluated nuclear cross sections, NERVA isotope inventory, hot-fuel transport data, hydrogen molecular scattering kernel, material density feedback, covariance, or hot-core OpenMC comparison**. It assumes a fixed mesh, energy group structure, neutron-group velocities and delayed-neutron decay constants. Piecewise linear interpolation is a numerical method and must be checked against independent evaluated temperature points and resonance/Doppler treatment; a physically inconsistent dataset will produce a mathematically consistent but physically false eigenvalue.

Stage 17 also does not determine the relationship between temperature and pressure/density, phase composition, coolant flow, fuel-element power or reactor control. `core_material_temperature_qualification.py` and any actual transport benchmark remain separate gates that cannot be bypassed merely by interpolating synthetic or user-declared tables. Do not report a NERVA d(k)/dT until the tables, spectrum collapsing, material chemistry, and physical benchmark have been independently qualified.

Branch stacked on Stage 16 PR #128; NUKE main unchanged.
