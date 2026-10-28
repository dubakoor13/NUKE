# Stage 38 — verified actual OpenMC inactive-generation reproducibility experiment

**Core physics: genuine C++ OpenMC neutron/photon source genealogy and deposited-energy verification.** No change to reactor geometry, material, control state, fuel, evaluated nuclear data or thermodynamic assumptions. This is the cold far-subcritical three-ring surrogate; historical hot NERVA remains unqualified.

## Real original OpenMC evidence

- [Successful source-reproducibility audit GitHub Actions run 40750293544](https://github.com/dubakoor13/NUKE/actions/runs/40750293544). **11/11 adversarial mathematical tests + 2/2 mandatory original coupled-HDF5 tests passed**. Exact input geometry, material and tally XML and evaluated nuclear-data index SHA256 were checked.
- Existing [successful Stage37 transport 40747291683](https://github.com/dubakoor13/NUKE/actions/runs/40747291683) supplied original neutron+photon source banks at batches 060,100 with **20 inactive / 100 total**, 3,000 particles per generation; broad and compact initial distributions.
- A **new genuine four-history OpenMC C++ v0.15.3 transport** with **40 inactive / 100 total** completed successfully in [run 40749064255](https://github.com/dubakoor13/NUKE/actions/runs/40749064255). Its original 8 native HDF5 source banks and input provenance survive in artifact [12843212697](https://github.com/dubakoor13/NUKE/actions/runs/40749064255/artifacts/12843212697). That original first run's high-level comparison correctly **failed** because it had required *exact same-source-path equality*, and the actual paths proved different. The subsequent successful audit **reused these original physically executed HDF5 files** rather than running new transport.
- The final audit reopened **16 authentic StatePoint HDF5 files**: 4 configurations (two initial source shapes × seed labels 13007,23011) × two inactive thresholds × batches 060,100. Both runs in each set had exact internal `k_generation` and entropy prefixes, valid native source banks, neutron+photon mode and true tally realizations.
- Reconstructed physically nonoverlapping active-window energy sums: reference 20 inactive has [21–60] and [61–100]; candidate 40 inactive has [41–60] and [61–100]. Every window's **3D mesh = all materials = neutron+photon+electron+positron** deposited-energy totals were verified within 1e-9 relative. Retained only source-normalized eV per Monte Carlo starting neutron.

## Actual source-path non-reproducibility — major finding

**The same integer RNG seed does not prove the same native neutron transport trajectory.** The observed `k_generation`, source entropy and final source-bank records in these reruns **diverged at generation 1–4**, even though original physical material, geometry and evaluated data fingerprints were identical. This is an observed deterministic-reproducibility failure, but without a same-threshold rerun control it **does NOT prove that changing the inactive threshold caused it**. A baseline repeatability test must first separate normal run-to-run variability or OpenMC parallel reproducibility from changes to the scoring threshold.

| Original physical metric | Measured result |
|---|---:|
| Exact k-generation, source entropy and 3000 source bank equality across different inactive runs | **False** |
| Mean original source-bank spatial TV at final batch 100, old vs new, 4×4×10 | **0.5756666667** |
| Mean late-window 61–100 **3D heating-fraction TV**, old vs new, 12×12×10 | **0.1344351486** |
| Genuine native material/mesh/particle energy partition closure | **Passed** |
| Actual 16 HDF5 / 4 new independently executed runs | **Verified** |
| Within-run native k/entropy full-generation prefix consistency | **Passed** |

The source-bank and heat-field TV are **differences between two distinct stochastic realizations**, not isolated effects of tally burn-in. These 4 paired labels do not establish either paired common-random-numbers validity or statistical independence, and cannot yield calibrated uncertainty or source-stationarity conclusions.

### Physical per-run observations

| Source initialization | RNG seed label | Within the SAME reference history mean k after batch 20 | After batch 40 | Between-rerun final source TV | Between-rerun late 3D heat TV |
|---|---:|---:|---:|---:|---:|
| Broad | 13007 | 0.0119633 | 0.0118019 | 0.49467 | 0.12730 |
| Broad | 23011 | 0.0116584 | 0.0115602 | 0.56033 | 0.17147 |
| Compact | 13007 | 0.0115053 | 0.0115651 | 0.60700 | 0.13636 |
| Compact | 23011 | 0.0117596 | 0.0117419 | 0.64067 | 0.10261 |

The mean-k comparison in columns 3–4 is **a mathematically valid truncation of the SAME actual original 100-generation `k_generation` sequence**. In contrast, the final TV columns compare **distinct reruns**, not the same histories; they must not be interpreted as pure inactivity-setting bias.

Native neutron/photon total deposited heat differs between the reruns at active generations 61–100 as well; because the original neutron tracks are different, that is NOT a physical energy-conservation failure and NOT proof that increasing burn-in caused this difference.

## Where this leaves the NERVA numerical qualification

The measured strongly subcritical `k_eff≈0.012` still corresponds to a cold 19-fuel-element surrogate rather than a physically calibrated historical NERVA reactor. No independent fission-source stationary distribution, genealogically independent particle count, valid physical reactor W/cm³ normalization or hot evaluated data is certified.

**Next core numerical gate:** repeat an identical *same inactive threshold* OpenMC configuration at the same seed, geometry, nuclear data, OpenMP/thread and source settings, and check exact neutron histories. If that is not reproducible, investigate parallel neutron-bank ordering and OpenMC RNG source initialization before testing a causal burn-in effect. After that, prioritize historical NERVA physical geometry, high-temperature evaluated neutron/scattering data and coupled hot-core thermal-material feedback over unrelated reporting.

## Archived machine-readable result

The original **full scientific JSON** (per-run statepoint SHA256, exact/native k and entropy nonidentity, source spatial TV 60/100, reconstructed heat arrays reduced by spatial resolution, mean source-normalized heating and conservative energy closure) is artifact [12843536512](https://github.com/dubakoor13/NUKE/actions/runs/40750293544/artifacts/12843536512). Original HDF5 archives in Stage37 and Stage38 have limited retention to **2026-11-18**. Preserve source binaries and evaluated-data fingerprints for long-term physical verification.
