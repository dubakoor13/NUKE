# NUKE Stage 25 — executed-material, native HDF5 and evaluated-data fidelity gate

**Date:** 2026-10-25. **Parent:** Stage 24 PR #137. **Evidence:** authentic four-seed OpenMC 0.15.3 cold NERVA-derived surrogate, [Actions run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), original artifact `nerva-real-openmc-four-seed-core-evidence` (ID 12492499186).

## Purpose

Stage 24 independently established that the original neutron transport has a conserved per-generation neutron ledger but **no usable local neutron+photon deposited heating**. Stage 25 addresses a different P0 physical blocker: the exact **executed material and nuclear temperature data**, not the contents of a Python material-definition function or a declared-but-unplaced input material.

This read-only module reconciles, on each of four actually executed seeds, the original OpenMC `summary.h5` native material isotope and atom density records, native geometry cell-to-material assignments and effective temperatures, original `materials.xml`, the executed physical-material/temperature report, and the externally claimed isotope HDF5 file manifest. It is independent of the existing qualification code's calculations in the sense that original HDF5 and XML are reopened and compared rather than copying reported flags; however it cannot re-evaluate neutron nuclear cross sections without the actual evaluated isotope HDF5 bytes.

## Verified physical model actually executed

| Scope | Original executed evidence |
|---|---|
| Materials declared in `materials.xml` | 9 |
| Materials *actually assigned to physical HDF5 cells* | 4 — IDs 19, 20, 21, 22 |
| Material-filled geometry cells observed in each source summary | 60 |
| Unplaced declared materials | 5 — IDs 23–27 |
| Isotopes present in the physically used four materials | 13 |
| Isotope neutron HDF5 file **digest entries claimed by original manifest** | 29 |
| Explicit temperature in executed `materials.xml` | 1/4 physically used materials |
| Physically used materials with implicit/default temperature | 3/4 |
| Evaluated neutron isotopic temperature knots in original report | **293.6059129982851 K only** |
| Physically qualified `S(alpha,beta)` bound solid scattering | **No**; original run used free-gas surrogate |
| Independently identified certified data-release edition | **No** |
| Actual evaluated isotope HDF5 binary files included in the archived source ZIP | **No** |

The executed, root-reachable material inventory is:

1. ID 19: **conceptual (U,Zr)C-graphite surrogate** (10 nuclides); NO explicit material temperature.
2. ID 20: **hydrogen coolant** (H1,H2); explicitly exported temperature **293.6059129982851 K**.
3. ID 21: **ZrC protective coating** (6 nuclides); NO explicit material temperature.
4. ID 22: **beryllium radial reflector** (Be9); NO explicit material temperature.

The five **declared but not actually reached** material types are tie-tube graphite filler (ID 23), conceptual ZrH moderator (24), simplified Inconel (25), B4C control absorber (26), and aluminum vessel surrogate (27). Their presence in `materials.xml` and the 29-entry library manifest is **not** evidence their neutron physics was exercised by this particular original OpenMC run. In particular, a declared absorber or hydride does not establish control-drum worth, tie thermalization or bound-hydrogen moderation in the executed geometry.

The source `summary.h5` contains effective cell temperature values near 293.6 K; this is **not** equivalent to a physically documented, explicitly assigned hot fuel/coating/reflector temperature, nor does it establish evaluated temperature coverage. The original report says the assigned nearest-temperature method itself is not independently hot-qualified.

## Numerical/evidence checks in Stage 25

`ntr/nerva/real_native_material_xs_audit.py` reopens the real original archived binary files and checks:

- Native material IDs/names/ordered nuclides and **positive atom densities** in `summary.h5` agree with physically executed `materials.xml`; sum of HDF5 isotope number densities matches the reported total atom density to conservative numerical tolerance.
- Every material-filled native geometry cell references a declared material and carries a finite positive observed cell temperature; a cell with an explicit XML temperature has the same effective OpenMC temperature (within recorded precision).
- Native geometry reachability is independently reconstructed from material-filled HDF5 cells and checked against the original physical-material summary. This avoids accidentally including unplaced materials in active-core analysis.
- All physically reached isotopes occur in the available library **metadata** entries, with nonduplicate identities, plausible byte sizes and well-formed SHA-256 hash strings. It explicitly distinguishes metadata **hash claims** from actual independent isotope-binary SHA verification.
- All isotope temperature entries reconcile with the manifest; original evaluated neutron data give one ~293.6-K knot. No invented high-temperature extrapolation.
- Native executed XML and material/temperature source report agree for every seed. The original per-seed thermal-scattering physics audit identifies the **free-gas** approximation, no bound-solid-scattering qualification.
- Any missing/corrupted actual statepoint, material summary, XML, manifest/report or misdeclared material temperature must reject the physical evidence rather than substitute a mock, or falsely declare the hot model qualified.

`validate_real_native_material_xs_audit.py` includes numeric regressions on the actual four binary summaries and destructive misdeclaration tests. `.github/workflows/nerva-native-material-xs.yml` downloads the actual 2026-10-23 artifact by official run ID/name, requires the real sources and reuses the original Stage 21 HDF5 authenticity gate. The ZIP repackaged **for destructive test mutation only** is not misrepresented as the original source ZIP.

## Critical coverage and reproducibility limitations

The archive contains an index `verified_xs_files.json` with the names, byte sizes and SHA-256 strings of the 29 evaluated isotope data files in the OpenMC runner. That was enough for the **original runner** to load the files and execute a cold neutron transport. **Those actual isotope HDF5 binary bytes are NOT in the archived artifact.** Therefore Stage 25 can verify that the **reported metadata** is internally well formed and consistent but **cannot independently reproduce its isotope-file SHA values or inspect the energy-dependent nuclear reactions, S(alpha,beta) kernels, delayed yields, heating/KERMA coverage or Doppler broadening from original evaluated nuclear files**. This is not a gap that an XML hash or a Python assertion can close.

The execution was a cold strongly subcritical/source-cloned **NERVA-derived surrogate**, not a historical NERVA operating configuration. The very small `k_eff≈0.011698`, failed source-shape stationarity, absent local heating and missing historical measurements are independent blockers; a passed Stage 25 material/XML identity audit does NOT qualify those physics.

## P0 engineering tasks following this audit

- **DATA-01, EVIDENCE-02:** preserve the original OpenMC source archive before reported **2026-11-13** Actions expiration AND archive the **actual** isotope HDF5 binaries/cross_sections.xml with independently reproducible hashes, evaluated release/license/source identity. Avoid Git repository bloat; register immutable external artifact paths and checksums.
- **DATA-02/04:** independently qualify actual evaluated temperature/density-dependent isotope data and appropriate solid thermal-scattering kernels for the matched, truly present core materials. Supply physically documented temperatures and density/phase states for fuel, coating, reflector, H2 and other physically reached materials; explicitly fail outside the evaluated domain.
- **REF-01/TRAN-01:** choose a traceable core configuration, not merely a material declared in a surrogate XML. Confirm which graphite, tie, ZrH, absorber and vessel regions are actually present in the intended geometry and validate geometry/atom inventories independently.
- **TRAN-02:** overcome unqualified fission-source stationarity with a physically defensible transport model and genuine independent seeds; no extrapolation from the old k SEM.
- **HEAT-01–05:** qualify genuine neutron/photon local deposition and physical source-rate/power normalization; Stage 24 found zero local heating in original cold results.
- **COUPLE-01/03:** only connect truly mapped geometry/material/temperature/data states to the Stage 17–20 multigroup and thermal coupling. Existing synthetic coefficient tables remain legitimate numerical benchmarks, not measured reactor hot-data inputs.

**Stage 25 outcome: verified *what material and isotope inventory was actually transported*, and why its nuclear-data coverage is insufficient. NOT physically qualified hot NERVA core or new nuclear transport simulation.**

## Verified Stage 25 CI (2026-10-25)

The dedicated original-data [GitHub Actions run 40167366262](https://github.com/dubakoor13/NUKE/actions/runs/40167366262) **completed SUCCESS, 10/10 tests passed, no skips** on the original four-seed 2026-10-23 OpenMC binary source archive. CI independently reopened genuine per-seed `summary.h5` and `materials.xml` and verified 4 physically occupied materials, 60 material-filled cells per seed, 13 actual material isotopes and exactly one evaluated neutron temperature ~293.605913 K. The tests included deliberately inconsistent source XML, native HDF5 material name, forged report temperature, forged evaluated isotope temperature grid, missing-original-archive and physical nonqualification assertions. This passes identity/cross-representation verification **only**; it does not qualify actual evaluated isotope HDF5 bytes, hot reactor materials, full thermal scattering, independent source convergence, local nuclear power, or historical NERVA criticality.
