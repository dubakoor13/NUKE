# NUKE Stage 24 — authentic OpenMC neutron economy and native heating-score evidence

**Date:** 2026-10-25  
**Repository:** https://github.com/dubakoor13/NUKE  
**Parent:** Stage 23 PR #136, Stage 24 branch `feature/nerva-real-tally-physics-closure-stage24-20261025`  
**Exact original executed evidence:** GitHub Actions [run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), original artifact `nerva-real-openmc-four-seed-core-evidence`, ID 12492499186, pinned original ZIP SHA-256 `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`. Original HDF5 per-file SHAs independently required by the Stage 21 gate.

## Scientific objective

Stage 21–23 established that genuine neutron eigenvalue simulations were executed but their spatial fission-source **stationarity is unqualified**. Stage 24 asks a separate and important question: do the *native observed transport reaction rates* and deposition tallies actually support a conserved neutron population and any usable local thermal-power field? These questions must be answered from the saved original OpenMC HDF5 arrays, not from nominal reactor thermal power, synthetic neutron flux or a copied JSON summary.

NUKE main already contains `core_neutron_balance.py`, `core_reaction_history.py`, `core_reaction_ensemble.py`, `core_reaction_estimator_agreement.py`, `heating_data_qualification.py` and multiple physical source audits. Stage 24 does **not** replace or redesign those tools; it is an independent **cross-tally source-of-truth check** which operates on the original bytes in the Stage 21 evidence bundle and enforces the same physics gaps without estimating reactor power.

## Native HDF5 scores inspected

For each genuinely executed OpenMC 0.15.3 final `seed_<seed>/statepoint.120.h5` file (seeds 13007, 23011, 33013, 43019; 120 batches, 30 inactive, 90 active batches, 3000 particles/generation) the audit reads and independently sums the native HDF5 tally **first and second raw moments**, requiring the recorded tally `n_realizations=90`.

| Original native tally name | Estimator | Scores reviewed | Role |
|---|---|---|---|
| `nerva_3d_neutronics` | tracklength | flux, absorption, fission, nu-fission, heating, heating-local, kappa-fission | Whole-core 3-D mesh |
| `nerva_material_transport` | tracklength | flux, absorption, fission, nu-fission, heating, heating-local | Material partition |
| `nerva_material_energy_transport` | tracklength | flux, absorption, fission, nu-fission, heating-local | Material × incident energy partition |
| `nerva_global_reaction_tracklength` | tracklength | absorption, nu-fission | Independent global tally representation |
| `nerva_global_neutron_balance` | analog | absorption, scatter, nu-scatter, nu-fission | Actual collision and multiplicative scattering accounting |
| `global_tallies[3]` | native global OpenMC | leakage | Whole-core neutron leakage (not an arbitrarily oriented surface current) |

The mesh, material and material-energy scores sample the **same source histories**; matching totals across them verify coverage/normalization and do NOT create new independent samples. Treat differences between analog and tracklength as separate estimators with covariance not reconstructed merely from marginal errors.

For every common reaction score, Stage 24 checks agreement between the 3D-mesh integral, all material bins, and all material-plus-incident-energy bins. It also checks that global tracklength absorption and nu-fission match those partition sums. Numerical equality for these original snapshots is enforced with strict tolerances.

## Native neutron conservation

In an eigenvalue generation, fission daughters are *banked into the next generation*, not counted as instant additional neutrons in the present history. The correct observed neutron-population equation includes multiplicative scattering and global vacuum leakage:

```
 analog_absorption_per_source + native_vacuum_leakage_per_source
    - (analog_nu_scatter_per_source - analog_scatter_per_source)
 = 1 transported starting neutron per eigenvalue generation
```

The independent archive's original run reports the four-seed mean:

| Quantity, per starting neutron | Actual observed mean |
|---|---:|
| Analog absorption | 0.039666666666666656 |
| Tracklength absorption | 0.03991528763875892 |
| Native global vacuum leakage | 1.0238555555555557 |
| Net neutron production from multiplicative scattering | 0.06352222222222137 |
| Tracklength banked fission neutrons | 0.011688432934274583 |
| Analog banked fission neutrons | 0.011669789908648396 |

Those values satisfy the per-generation native neutron accounting identity **to roundoff**. This is a verification of the actual transport accounting convention, **not evidence of a critical or source-stationary NERVA reactor**. In particular, `k_eff≈0.011698` in the original run remains very far below critical, and source-bank spatial cloning/instability remains unqualified.

## CRITICAL — fission energy is NOT automatically local deposited heat

The same native, physically executed OpenMC neutron-only statepoints have **nonzero** `heating` and `kappa-fission` scores (first moments), while `heating-local` has exactly **zero raw first and second moments** in all relevant material, material-energy and 3D mesh tallies, for every saved final seed. The fact that some nuclear energy score is nonzero is NOT evidence that a usable material-disjoint **neutron+photon local deposited-power map** exists.

The Stage 24 fail-closed audit explicitly sets:

- `physical_local_heating_qualified = false`.
- `hot_core_reactor_power_normalization_allowed = false`.
- `source_stationarity_demonstrated = false`.
- `historical_nerva_operating_state_validated = false`.

A nuclear `kappa-fission` tally may reflect recoverable fission energy in an estimator but does not establish **where** neutron and photon energy is deposited, whether photons escape, or how much goes directly into hydrogen/tie/reflector/fuel. An independently justified source-rate or absolute physical power in W is also absent. Do not substitute `heating` for `heating-local` solely to get nonzero heat, invent reactor power, or feed this cold archive into Stage 20's full hot-core power closure.

## New code, tests and strictly read-only workflow

- `ntr/nerva/real_native_tally_physics_ledger.py`: validates real Stage 21 archive identity and genuine OpenMC HDF5 tallies, raw moment/realization counts, estimator names, cross-filter consistency, native neutron generation accounting, actual nonlocal energy score versus zero local power and explicit physical blockers. Outputs small provenance-labelled JSON.
- `ntr/nerva/validate_real_native_tally_physics_ledger.py`: tests four original HDF5 seeds, independent measured reaction/leakage means, current-cycle vs banked fission neutron distinction, zero locally deposited heating despite nonzero other scores, statepoint hashes, failure on missing genuine artifacts, byte-level HDF5 corruption and forged heating report rejection.
- `.github/workflows/nerva-native-tally-physics.yml`: official GitHub cross-run artifact download, requires the actual four original HDF5 files and declared evidence JSON, runs the native audit, then **repackages the verified original extracted evidence ONLY for destructive tampering regression tests**. This repacked regression ZIP is not claimed to match the original ZIP SHA; each native statepoint SHA is checked against the Stage 21 recorded manifest. Uploads a small derived JSON report; no new reactor data generated.

Run the following with the original extracted source evidence available:

```bash
python -m pip install numpy h5py
python -m ntr.nerva.real_native_tally_physics_ledger /path/to/actual_extracted_openmc \
    --output build/real_native_neutron_heating_ledger.json
NUKE_REAL_CORE_EVIDENCE_PATH=/path/to/actual_original_or_repacked_archive.zip \
    python -m ntr.nerva.validate_real_native_tally_physics_ledger
```

## Remaining engineering / physical requirements

1. **Independent, source-stationary real transport** with approved physical core geometry/materials, fission-site genealogy and statistically justified source stationarity (Stage 22/23 have documented significant cloning and spatial drift).
2. **Actual evaluated, temperature-/density- and composition-qualified nuclear data**, including bound thermal scattering and relevant photon interaction/energy deposition, for the target reactor state; the current cold surrogate cannot prove hot material validity.
3. **Genuine spatial neutron+photon local heating** from a coherent, properly scored transport model and source/physical-power normalization. Verify all fuel, hydrogen, tie, structural and reflector partitions without overlap, with leakage and uncertainty.
4. **Matched geometry/energy deposition projection** into the resolved Stage 19/20 fuel-ring and hydrogen thermal meshes, including correct region-wise energy, physical power and independent first-law checks.
5. **Independent historical NERVA measurements** for the same selected as-built or design reference and held-out comparisons. A cold far-subcritical surrogate is not a historical critical reactor measurement.

**Status remains numerical/evidence verification on the cold original OpenMC model; physical hot NERVA nuclear performance remains unqualified.**
