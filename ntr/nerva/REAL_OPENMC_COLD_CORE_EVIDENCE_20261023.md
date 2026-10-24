# Stage 21 — REAL OpenMC Cold-Core Evidence and Physical Blockers

**Audited 2026-10-25.** This is observational nuclear-transport QA, not reactor redesign.

## Immutable genuine execution evidence

- GitHub Actions run: https://github.com/dubakoor13/NUKE/actions/runs/39933618638
- Successful OpenMC 0.15.3 C++ neutron eigenvalue transport, executed 2026-10-23 for a **three-ring NERVA-derived, historically uncalibrated COLD surrogate**.
- Real evidence artifact: https://github.com/dubakoor13/NUKE/actions/runs/39933618638/artifacts/12492499186
- Artifact ID: 12492499186; name: nerva-real-openmc-four-seed-core-evidence.
- Original ZIP SHA-256: cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274.
- 111 ZIP entries, including 20 actual neutron source HDF5 checkpoints (4 seeds x 5 batches), four final statepoint.120.h5 files, neutron source diagnostics, exported executed material/geometry settings, and original physical qualification reports.
- Expected archive expiry: **2026-11-13**. Preserve this and its original HDF5 files before expiry.
- Original real_core_transport_summary.json SHA-256: 830a006751aa3a10bd4b5637a6e0a3a481fba20cc1e26a065791a6a72f6bfa21.
- Evaluated nuclear data index SHA-256: 71c6be06e5584e5bd23de27f148286f6a6d21d9c771e03ede629ba66741841e0.

All four final HDF5 statepoints were directly inspected by h5py from actual ZIP bytes. Each was checked for its native seed, exact 120 batches / 30 inactive / 90 active, 3,000 starting neutron histories per generation, eigenvalue mode, OpenMC 0.15.3, saved source bank, and native k_combined. Its SHA-256 matches the run summary manifest.

| Seed | Real OpenMC k_combined | Reported single-seed SEM | Original final statepoint SHA-256 |
|---:|---:|---:|---|
| 13007 | 0.01181822339995014 | 0.00015007440099844432 | 615e03f42fa57fd0b37430c18d9d2a6ab45be17c958dd2cb5ae2505d7b511ca2 |
| 23011 | 0.011635440258488425 | 0.000150638018874634 | 379cbcde736bc641d0ff080c7c53878e67a91e375a81ef84783740ff93d8c79a |
| 33013 | 0.011694374033855901 | 0.00016461240935610076 | 9ef95aba72ca4f99cefadfc915ffcd0e00f1b23c000684e13e9ef1a14f58cd72 |
| 43019 | 0.011641999580734662 | 0.0001514216094956803 | 09133fea6ebdb2b51c9d9319a91fc4f3298bd6b99d343e26546b09733c45babe |

Recomputed inverse-variance mean k: **0.01169802857789661**. Original formal pooled SEM: 0.0000769260125465282. Because fission-source genealogies are correlated and stationarity unproven, the formal SEM is NOT a certified total uncertainty or an experimentally measured reactor quantity.

## Genuine core nuclear-physics qualification outcomes

| Gate | Original real-transport evidence | Result |
|---|---|---|
| OpenMC C++ executed on a real model | Four independent declared RNG seeds, 20 saved native source bank statepoints, verified metadata/hashes | **YES — genuine neutron transport executed** |
| Source spatial stationarity | Mean final cross-seed spatial total variation about 0.68572; source convergence report explicitly false | **NOT ESTABLISHED** |
| Source genealogy independence | Minimum effective independent source-site fraction about 0.006748; substantial source clones | **NOT ESTABLISHED** |
| Critical self-sustaining reactor operation | Mean k around 0.0117; original criticality necessary-condition screen reports FAR_BELOW_CRITICAL; even descriptive +10 SEM upper screen <0.0151 | **NO — cold surrogate far below critical** |
| Evaluated neutron data for hot core | Root-reachable real materials: 4 of 9 declared; 13 physical nuclides; available temperatures in this execution only around 293.606 K | **NOT QUALIFIED** |
| Physical material temperature assignment | Only 1 of 4 reachable materials explicitly temperature-labelled; 3 rely on unspecified solver defaults | **INCOMPLETE** |
| Thermal scattering | Cold free-gas assumptions; bound graphite/solid kernel and mixture-phase qualification not established | **NOT QUALIFIED** |
| Real neutron local energy deposition | All four native heating-local scores = **0 eV/source**, ordinary heating scores nonzero but do not qualify local deposition | **BLOCKED** |
| Normalization into physical fuel/coolant W | Independent heating_data_qualification says thermal_power_normalization_allowed=false | **PROHIBITED FOR THIS EVIDENCE** |
| Historical NASA NERVA comparison | Uncalibrated NERVA-derived surrogate, not locked experimental reactor test configuration | **NOT VALIDATED** |

### Critical interpretation

The archive is genuine and valuable for cold neutron **execution evidence, neutron-conservation checks and falsification of a hot operating-core claim**. It does NOT satisfy real power-producing reactor validation. Do not convert its nonzero ordinary heating score into qualified heating-local, change k by algebraic rescaling, infer absolute reactor MW from eigenvector amplitude, or identify its 293.606 K nuclear-data input as a hot-core cross-section library.

## Specific remaining P0 physics work identified from real data

1. **TRAN-01/02:** Preserve original executed cold transport artifacts; independently demonstrate spatial source stationarity and source-bank effective independence with new real neutron-history evidence. Existing k dispersion is not enough.
2. **DATA-01/02/03:** Qualify every root-reachable isotope, material, temperature, S(a,b) scattering kernel, phase, density and composition on a nuclear library actually covering the desired hot state. Do not invent evaluated hot XS.
3. **HEAT-01/02/03:** Score meaningful local neutron/photon deposited energy from physically applicable data and independently verify units/partitions/statistical error. Only then allow W normalization to the actual fuel/H2/tie geometry.
4. **REF-01:** Identify a specific historical NASA tested article and source dataset; keep this three-ring subcritical surrogate labeled an exercise until explicitly validated against independently documented hardware.
5. **COUPLE-01/02:** Once transport and deposition are physically qualified, connect source-identical temperature-dependent multigroup operators to Stage 17–20 fuel/H2 thermal solvers with consistent mesh/energy ledgers.

## Reproducible Stage 21 audit

Stage 21 adds two files:
- ntr/nerva/audit_real_core_bundle.py — read-only original ZIP/HDF5 provenance, physics-readiness and k comparison.
- ntr/nerva/validate_real_core_bundle.py — requires actual saved four-seed statepoints; detects altered HDF5 checksum, duplicate seeds, counterfeit synthetic transport label, mismatched XS and criticality audits and false hot-Xs qualification.

A dedicated GitHub Actions workflow downloads the existing successful 39933618638 artifact, packages its files into a ZIP without modifying HDF5 bytes, runs the evidence auditor and real-data corruption tests, and uploads the lightweight audited JSON. This reruns the **evidence check**, not neutron Monte Carlo transport itself.

To reproduce manually, download the original real transport artifact before expiration and run:

    python -m ntr.nerva.audit_real_core_bundle --artifact PATH/TO/ARCHIVE.zip --workflow-run-id 39933618638 --output build/nerva_real_evidence_audit.json

Use an archive with the original relative member paths, not a Python-generated mock. The original binary ZIP digest applies to the original ZIP bytes; a repacked ZIP naturally has a different outer SHA, even if its internal statepoint bytes and saved summary SHA remain identical.

**P0 acceptance is NOT met by this archive alone:** genuinely executed cold transport is established; actual source stationarity, qualified hot materials, local nuclear deposition and historical NERVA fidelity remain missing. No reactor parameters were changed in this audit.


## 6. Stage 21 independent genuine HDF5 audit — verified CI

On Stage 21 source commit b1299171d4a170137384deef426758d17aba3906, the separate actual-evidence workflow
https://github.com/dubakoor13/NUKE/actions/runs/40129283923 **COMPLETED SUCCESS**.

- It downloaded the actual run 39933618638 through the GitHub Actions artifact API, not a mock OpenMC library.
- 8/8 real-artifact integrity and falsification tests passed in 2.836 s. Baseline test inspects all four final native statepoint HDF5 files and their exact SHA and k values.
- Deliberate changes to HDF5 source bytes, duplicate seed metadata, synthetic relabeling, wrong criticality/XS manifest and false hot-data qualification were rejected as expected.
- It independently recomputed the weighted cold k as 0.01169802857789661.
- Source stationarity=False; hot evaluated data qualification=False; physically usable local heating=False; historical NERVA validation=False.
- Uploaded generated lightweight audit JSON artifact under name nerva-executed-cold-core-audit-summary (artifact ID 12574985915, expires 2026-11-07) for reproducibility. The ORIGINAL larger four-seed HDF5 ZIP (artifact 12492499186) expires 2026-11-13.
- This is **physical source evidence verification**, NOT a new OpenMC particle transport execution or a completed hot reactor simulation.

Maintain EVID-05 (long-lived preservation of the real archive), EVID-06 (actual source stationarity), EVID-07 (nonzero qualified local neutron/photon heating), EVID-08 (hot evaluated materials) as OPEN.
