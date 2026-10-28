# Stage 47 — executed genuine multi-temperature neutron transport physics

**Status: verified actual original OpenMC C++ neutron transport and evaluated data** on 2026-10-28 UTC / 2026-10-29 Asia/Kolkata.

[**SUCCESSFUL original Stage47 GitHub Actions 40765287931**](https://github.com/dubakoor13/NUKE/actions/runs/40765287931) executed two separate original OpenMC v0.15.3 continuous-energy eigenvalue neutron transport runs using the actual **official OpenMC-processed ENDF/B-VII.1 multi-temperature HDF5 neutron library**, not the cold-only NNDC test library. All eight adversarial mathematical tests and both mandatory genuine nuclear-data/source-StatePoint HDF5 tests passed (10/10). An independent quick test of the pinned OpenMC C++ temperature sampling source also passed all eight mathematical tests: [fast workflow 40765680985](https://github.com/dubakoor13/NUKE/actions/runs/40765680985).

## Original provenance: authoritative isotope evaluations and solver

- **Official published library:** https://openmc.org/data/ — OpenMC-produced ENDF/B-VII.1, actual evaluated neutron temperatures 250 K, 293.6 K, 600 K, 900 K, 1200 K, 2500 K. Original archive obtained directly from the official https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz and extracted without altering evaluated neutron data or regenerating cross sections.
- **Original official archive SHA256:** `5bc4b62a602d569e3a801c3499f03378192e83623c3eb39500804178706781f9`.
- **Actual original `cross_sections.xml` SHA256:** `2789cb61537a56a14b555ca97f439ed14896bd6db9bfe6460e066db3c96843db`.
- **Original OpenMC C++:** `openmc/openmc:v0.15.3`; checked the exact published `openmc-dev/openmc` v0.15.3 `src/nuclide.cpp` Git blob SHA `5ae6e30ee24e4f16c31fd04dbf9b7c9b4afa5c3e`, including the stochastic temperature-dependent neutron cross-section table selection:
  `f = (kT - kTs_[i_temp]) / (kTs_[i_temp+1] - kTs_[i_temp]); if (f > prn(p.current_seed())) ++i_temp;`.
- **Global source-selection caveat handled:** OpenMC's original C++ can change the global interpolation method to NEAREST if an incident neutron isotope has only a single evaluated temperature. The new Stage47 original-data validator rejects missing kT brackets for **every physical isotope used by the declared core materials**, not only U235 or the fuel.
- All **29 required neutron isotopes** had genuine evaluated temperature coverage at the material states exercised; 44 isotope/material requests were checked. Every 750 K request was strictly bracketed by physical evaluated tables at `600.0000049667 K` and `900.0000074501 K`. The observed upper-table stochastic selection probability from actual HDF5 kTs was `0.4999999793053` (roundoff from stored kT precision). Original **SHA256 of each evaluated isotope HDF5 file** is retained in the result JSON. All high-T neutron material conditions were covered by actual evaluated data.

## Real native OpenMC C++ continuous-energy neutron results

The current conceptual three-ring, 19-fuel-element, **very subcritical surrogate geometry, density and chemistry are unchanged**; only the neutron-physics material temperatures were changed. Both runs used fixed original densities and equal fuel/H2/reflector/moderator/coating/vessel temperatures for their respective snapshots. Requested neutron selector was `interpolation`, no evaluated-bound `S(alpha,beta)` claim. RNG seed 13007, 3000 starting neutron histories per generation, 24 batches, 8 inactive, 16 active; each run has an original signed/fingerprinted native `statepoint.24.h5`, particle source bank and scores.

| Original actual C++ neutron observable | 750 K intermediate | 900 K evaluated reference |
| --- | ---: | ---: |
| Actual cross-section selection | stochastic 600/900 K interpolation | evaluated table at 900 K |
| All 29 isotope kT data support | **Verified** | **Verified** |
| Original OpenMC HDF5 | **Completed** | **Completed** |
| Active-generation mean k (DESCRIPTIVE, not calibrated) | **0.010164112510559657** | **0.010149467059102043** |
| Final neutron source entropy | **5.152459961057226** | **5.492616601918809** |
| Native original sourcebank SHA256 | `90dad29e09acf93297f879f8186d2fb0d65bf6239d7bd8d9b4d2d32287642f28` | `c9b727b5854fdd3dd7026f9a236498f4877c9ab97b57061316562ea72f5d485d` |
| Original native `statepoint.24.h5` SHA256 | `afc6b3dfb73fe9fddd58231a966992f443f87c45b7c6da220dd780fe0ff9a081` | `a90221c0e14c464d29ac677d393026a8ca70fc0a038c4d2d501aac5f63e82d6c` |

**Observed maximum `k_generation` difference between the two original source histories = 0.004321550763485714**; original complete 3000-neutron source banks differ. A small difference in the *arithmetic means* of these two short correlated eigenvalue histories **is NOT a temperature feedback coefficient or statistically reliable physical hot-reactor effect**.

**First genuine multi-temperature neutron-transport milestone achieved:** unlike Stage45's interpolation request that downgraded to a single cold table, this original physical experiment actually had two independently evaluated neutron temperatures bracketing each 750 K material/isotope request, and the C++ solver source is pinned to the correct stochastic sampling implementation. The data-availability and operator-execution barriers are resolved for this idealized fixed-density uniform material test.

## Rigorous limitations and next high-value physics work

- **Two short single-seed runs are not a calibrated neutron source stationary distribution or reactivity-temperature coefficient.** The same-seed run at distinct physical temperatures cannot be assumed to be common-random-number paired after source-path divergence. Do not infer temperature worth from mean k without independent runs, burn-in/entropy/source convergence and uncertainty coverage.
- **The model is not a verified historical NASA NERVA reactor.** The fuel composition, enrichment, 19-element layout and coatings remain conceptual, and the surrogate k~0.01 is highly subcritical.
- **Fixed material mass densities** do not implement coupled thermal-fluid hydrogen pressure/EOS, flow acceleration, thermal expansion, or energy conservation under a reactor power boundary condition.
- **Bound-scattering thermal kernels remain unqualified at hot temperature.** This test explicitly uses the existing free-gas scattering surrogate; no hot S(alpha,beta) fidelity is claimed.
- **Heating/power and neutronic-thermal feedback have not been validated.** Original deposited-energy tallies if present are normalized per Monte Carlo source, not reactor W/cm3. No historical hot NERVA power level or physical operating condition is inferred.
- **Most valuable next step:** run independent-seed source-convergence and uncertainty checks at the now genuinely available multi-temperature neutron conditions, then validate thermal-scattering/solid material state and conservative solid/hydrogen thermal feedback. Avoid further rounds of narrow inactive-generation diagnostics whose C++ keff-feedback mechanism was already established at Stage44.

## Archived original scientific evidence (GitHub Actions)

- [**Full real per-isotope evaluated neutron kT table SHA, 750/900 K source physics JSON, artifact 12849228640**](https://github.com/dubakoor13/NUKE/actions/runs/40765287931/artifacts/12849228640).
- [**Both original HDF5 StatePoints, XML inputs, actual nuclear-data index/archive SHA, official C++ file, artifact 12848739297**](https://github.com/dubakoor13/NUKE/actions/runs/40765287931/artifacts/12848739297).
- [**Original accepted workflow**](https://github.com/dubakoor13/NUKE/actions/runs/40765287931); all 10 tests passed.
- [**Fast separate 8/8 mathematical/upstream C++ tests**](https://github.com/dubakoor13/NUKE/actions/runs/40765680985).

The large external evaluated library itself is **not stored in GitHub**; its authoritative origin, original archive SHA, nuclear-data index SHA and each required isotope file SHA are preserved to enable independently fetching/repeating the exact original data. GitHub Action artifacts expire around **2026-11-18**.
