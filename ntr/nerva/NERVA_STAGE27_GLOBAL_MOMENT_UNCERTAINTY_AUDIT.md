# NUKE Stage 27 — actual OpenMC global tally moments and uncertainty integrity

**Date:** 2026-10-26. **Parent:** Stage 26 [PR #139](https://github.com/dubakoor13/NUKE/pull/139).  
**Original real evidence:** [NERVA core neutron-transport run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), four OpenMC 0.15.3 cold surrogate seeds, 120 batches / 30 inactive / 90 active, original artifact ID 12492499186.

## Scientific purpose

Stages 21–26 inspected genuine source banks, neutron accounting, tally conservation, and root-geometry connectivity. Stage 24 checked mean scores, but the reported SUM_SQ alone does not establish an integrated uncertainty if multiple mesh/material bins contribute to the same global quantity. This stage computes the **native OpenMC conditional standard error** only for genuine one-bin, whole-domain global tallies and the original global neutron leakage estimator. It separates seed-to-seed variation from within-seed batch uncertainty **without treating nonstationary fission-source histories as independent or scientifically qualified**.

No reactor-design change, nuclear-data substitution, control-drum tuning, source modification, XS generation, neutron/photon heating synthesis or absolute power normalization is performed.

## Native estimator convention

At the end of an active generation/batch OpenMC accumulates the score's first- and second-moment sums, `SUM = Σ x_i`, `SUM_SQ = Σ x_i²`, and `N = n_realizations`. For one globally defined score bin, OpenMC's variance of the sample mean is:

```
mean = SUM/N
SEM_native² = max[0, SUM_SQ/N - (SUM/N)²] / (N - 1)
```

The formula is grounded in `src/state_point.cpp` and `src/tallies/tally.cpp`, including the native `global_tallies[3]` neutron-leakage row. This is a **nominal within-run active-batch SEM**, not a guarantee of correct frequentist confidence coverage. Source genealogies and serial correlation can make it misleading.

The audit refuses:

- Any multi-bin tally for global `SUM_SQ` arithmetic without cross-bin covariance.
- Missing/duplicate global tally names, wrong analog versus tracklength estimator, unsupported score set or inconsistent active-batch count.
- Impossible raw negative variance beyond numerical roundoff, nonfinite values, or original statepoint SHA / Stage 24 mean disagreement.
- Forged source stationarity, qualified local thermal power, hot materials, or historical NERVA flags.

### Reports, explicitly labelled *descriptive*

For four genuinely independent RNG seeds, the report lists each raw first/second moment, single-bin conditional batch SEM, arithmetic seed mean, empirical between-seed standard deviation, and a variance-overdispersion indicator (between-seed sample spread divided by mean squared native conditional SEM). It also evaluates the **paired between-seed difference** of tracklength and analog absorption estimates. These two estimators score correlated histories; **no independent-score error propagation** is attempted.

A small four-seed sample plus nonstationary source banks does **not** make a statistically calibrated hot reactor confidence interval. The output explicitly marks confidence qualification false.

## Code and direct checks

- `real_native_global_moment_uncertainty.py` — genuine source HDF5 read, exact single-bin score moment extraction, native conditional SEM, four-seed descriptive diagnostics and fail-closed physical qualification labels.
- `validate_real_native_global_moment_uncertainty.py` — 8 adversarial mathematical/parser tests on artificial small HDF5 (never passed off as physical neutron simulations), and 2 independently executed checks against original real HDF5.
- `.github/workflows/nerva-native-global-moments.yml` — downloads original four-seed artifact by exact original workflow run and name, requires original binaries, executes numerical tests and publishes small observational JSON.

Reproduce with an environment containing numpy/h5py and the **original extracted artifact**, not fabricated scores:

```bash
python -m ntr.nerva.real_native_global_moment_uncertainty build/original \
  --output build/native_global_uncertainty_audit.json

NUKE_REAL_CORE_EVIDENCE_PATH=build/original \
python -m ntr.nerva.validate_real_native_global_moment_uncertainty
```

## Strict scientific status

**Code staged; first original-HDF5 CI execution pending at authoring.** No numerical outcomes from Stage 27 are claimed before native workflow success.

Regardless of this check, the executed physical state is a neutron-only, cold, far-subcritical NERVA-derived surrogate. The unresolved P0 gates remain:

1. Physical source/seed stationarity and relevant temperature-dependent evaluated nuclear/thermal scattering data.
2. Verified neutron and photon local material-resolved energy deposition (original `heating-local` is zero).
3. Unambiguous material volume and correct geometry-to-thermal fuel/channel mapping.
4. Physically justified reactor power/source normalization and coupled material/coolant states.
5. Independent NASA comparison for one declared historical reactor configuration.

Source-preservation dependency: the original evidence artifact has a reported expiry **2026-11-13**. Its actual original bytes, not just SHA strings, must be preserved along with usable neutron/photon isotope libraries; this code does not perform archival.


## Executed original-data results (2026-10-26)

**Dedicated GitHub Actions [run 40301688412](https://github.com/dubakoor13/NUKE/actions/runs/40301688412) completed SUCCESS.** It verified the official original 2026-10-23 four-seed ZIP download digest `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`, ran **8/8 mathematical parser tests** and **2/2 real native HDF5 regression tests**, and uploaded derived JSON artifact ID **12639799832** (`nuke-stage27-original-four-seed-raw-global-moments`). No source simulation was invented.

Real original data, **per transported starting neutron**, with sample SD **across the four seeds** (NOT a physically calibrated 95% interval):

| Global native score | Unweighted four-seed mean | Observed between-seed sample SD |
|---|---:|---:|
| Analog absorption | 0.039666666666666656 | 0.0004243287283280404 |
| Tracklength absorption | 0.03991528763875892 | 0.00010656584857075645 |
| Vacuum neutron leakage | 1.0238555555555557 | 0.000743187317651375 |

Native score conditional SEMs are retained individually in the JSON artifact; because the original cold fission-source stationarity remains unsupported, **no valid hot-core confidence interval or neutron/photon power distribution** follows from the numerical success. The broad `NERVA model check` is an independent workflow and is not counted as successful here unless its own job completes.

**Acceptance:** Stage 27 raw global SUM/SUM_SQ arithmetic and real evidence extraction VERIFIED; source-stationary neutron transport and hot nuclear-thermal physics still BLOCKED.
