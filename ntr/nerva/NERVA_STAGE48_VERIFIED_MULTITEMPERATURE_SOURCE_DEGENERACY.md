# Stage 48 — verified evaluated-multi-temperature independent-seed neutron source physics

**Scientific status: original OpenMC C++ transport completed and verified; reactor-source stationarity NOT demonstrated.**

[**Genuine original OpenMC Stage48 Actions 40767503400 — SUCCESS**](https://github.com/dubakoor13/NUKE/actions/runs/40767503400) executed **eight separate continuous-energy OpenMC C++ v0.15.3 eigenvalue neutron transport histories**, at **750 K and 900 K, four independent RNG seeds per temperature** (13007,23011,33013,43019), 3,000 initial source particles per generation, 100 generations (20 inactive). **16 original native HDF5 source StatePoints** (60,100) plus evaluated neutron XS archive/index SHA were verified. **Eight mathematical and two mandatory actual original HDF5 tests passed (10/10).** This report uses the real machine-readable stage48 numerical observations; no synthetic nuclear cross sections or source banks.

## Actual physical nuclear data and controlled reactor configuration

- Genuine official OpenMC-processed ENDF/B-VII.1 **evaluated multi-temperature neutron HDF5** (original [OpenMC release listing](https://openmc.org/data/) and upstream Box archive) as qualified in Stage47. **Archive SHA256**: `5bc4b62a602d569e3a801c3499f03378192e83623c3eb39500804178706781f9`; **evaluated cross_sections.xml SHA256**: `2789cb61537a56a14b555ca97f439ed14896bd6db9bfe6460e066db3c96843db`. 29 required neutron isotopes covered; 750 K is physically bracketed by evaluated 600/900 K tables and original C++ stochastic temperature selection, 900 K is a tabulated temperature.
- **Exactly unchanged geometry, fuel composition, particle/source settings, material densities, initial source shape and neutron scoring scheme** across all eight runs. Explicit material neutron temperatures for fuel, H2, Be/graphite/ZrH, ZrC, metal and B4C at 750 or 900 K. No fuel enrichment, size, drum, control or spectrum optimization.
- Original `settings.xml`, `materials.xml`, `geometry.xml`, native StatePoint identity, all source checkpoint SHA256, evaluated-data index SHA256 and actual source-bank physical bounds were checked. Four original independent source seeds per temperature, each with its own 60/100 HDF5. Particle-source spatial histograms used a fixed physically defined OpenMC entropy-source mesh (4×4×10 scoring bins), **not sample-min/sample-max bounds**, and preserved all source-bank site weight.

## Actual original C++ numerical observations

| Real original HDF5 observable | **750 K** | **900 K** |
| --- | ---: | ---: |
| Independent original OpenMC C++ seed runs | 4 | 4 |
| Native original HDF5 checkpoints | 8 | 8 |
| Mean of four seed-level active-generation `k` means | **0.010234905521526992** | **0.010146287964579587** |
| Empirical SD of four seed-level active mean `k` values | 0.00003752724273007928 | 0.00015031209810943355 |
| Seed-level SEM (descriptive only) | 0.00001876362136503964 | 0.00007515604905471677 |
| Illustrative t(3) 95% interval around mean k | [0.0101751913, 0.0102946197] | [0.0099071079, 0.0103854681] |
| **Mean source-bank total variation between batch60 and batch100** | **0.8315833333333333** | **0.7935000000000001** |
| **Mean pairwise different-seed source-bank TV at batch100** | **0.7900000000000000** | **0.8121666666666667** |
| Largest different-seed source-bank TV at batch100 | 0.883 | 0.921 |
| Exact distinct source positions per 3000-source-particle bank (batch 60 and 100 combined observed range) | **62–89** | **73–88** |

**Stage49 interpretation correction (verified against all 16 ORIGINAL HDF5):** Stage48's counts of 62–89 are **unique fission birth coordinates**, NOT the number of distinct complete neutron source states or independent fission ancestors. Stage49 audited position, direction, energy and all other actual source fields and found **2,250–3,000 distinct complete source neutron records out of 3,000**, while many records share a birth position. Thus Stage48's statement about strong **spatial** clustering remains valid but cannot by itself establish 97–98% identical complete source records, exact genealogical clone counts, or independently effective source sample size. See [executed Stage49 source-state audit](NERVA_STAGE49_EXECUTED_FULL_SOURCE_PHASE_SPACE_AND_REFERENCE.md).

**Particularly important physical/numerical result:** at batch100, only about **2–3% of the nominal 3,000 source sites** had distinct exact positions (source bank cloning), and source spatial TV remains enormous between checkpoints and independent RNG seeds. These are actual OpenMC saved bank coordinates; exact duplicate positions do not independently prove genealogical independence for all other source sites either. This is severe neutron-source population degeneracy in a **strongly subcritical** (`k≈0.01`) cold-composition conceptual reactor surrogate, not evidence that the true NERVA reactor source behaves this way.

Per-seed original `k` estimates from native C++ generation histories:

| RNG seed | k(750 K, batches21–100) | k(900 K, batches21–100) | Source bank TV(60→100), 750 K | Source bank TV(60→100), 900 K |
| --- | ---: | ---: | ---: | ---: |
| 13007 | 0.010187715616 | 0.010302105243 | 0.889667 | 0.854333 |
| 23011 | 0.010222712789 | 0.009941553309 | 0.779333 | 0.780333 |
| 33013 | 0.010271161257 | 0.010184700549 | 0.824333 | 0.778333 |
| 43019 | 0.010258032423 | 0.010156792757 | 0.833000 | 0.761000 |

**Observed four-seed mean difference:** `mean_k(900 K) − mean_k(750 K) = −0.00008861755694740577`. Conservative **illustrative** seed-level 95% difference band **[−0.0003875118651, +0.0002102767512]**, *includes zero*. This is not a statistically calibrated temperature worth, an independent-particle error bound, or a physical nuclear reactor negative temperature coefficient. Independence of particles within a shared eigenvalue source genealogy is not established, nor is source stationarity.

## Why the Stage47 short-run temperature comparison is not a reactor coefficient

The source bank is the population used to initiate subsequent eigenvalue generations. The present model's original HDF5 indicates large source drift/cloning across 40 extra generations. A seemingly stable arithmetic `k` mean across four RNG seeds does **not** validate the underlying stationary fission source; k bias and source history autocorrelation remain possible. Identical seed labels at two different material temperatures also cannot be assumed to provide paired common-random-number trajectories, because the neutron cross-section draws and subsequent source genealogies differ.

**None of these original fixed-density evaluated neutron runs establish:**
- A physical historical NASA NERVA operating-core eigenvalue or criticality, actual reactor geometry, fuel enrichment, control drum behavior or thermal power.
- A stationary fundamental eigenmode source / source-distribution convergence, independently calibrated generation-level uncertainty, or statistically defensible temperature-reactivity coefficient.
- Physical temperature-dependent bound `S(alpha,beta)` scattering for hot graphite/ZrH, hydrogen EOS/density/pressure evolution, coupled solid–hydrogen heat transfer or mass/energy feedback.
- Thermal watts, W/cm³, physical normalized hydrogen outlet temperature, or nuclear engine thrust.

## Immediate next **core physics and numerics** target

1. **Fix the physical core reference first**, rather than hiding `k≈0.01` source degeneracy with larger particle counts, arbitrary extrapolation or criticality tuning. Identify differences against actual NASA NERVA historical fuel element geometry, chemical composition, atomic densities, reflectors/moderation and the evaluated nuclear material specification. Do not simply change enrichment until k is high.
2. Quantify whether OpenMC's source diversity can be retained for the intended nuclear physics problem by using a physically justified, independently specified reference benchmark (without assuming the present surrogate should be critical). The verified population cloning diagnostics can reject unqualified source-converged claims.
3. Only after physically credible source calculations, run independent high-temperature neutron/photon energy deposition and couple conservative power normalization to hydrogen/solid thermal solver, with qualified scattering, fluid EOS, and stated uncertainty.
4. Treat each original evaluated isotope HDF5, nuclear XS index and simulator code as immutable scientific evidence; archive SHA256 and compare exact native C++ StatePoints.

## Original reproducibility artifacts (Actions)

- [**Stage48 accepted original physics execution, 10/10 tests**](https://github.com/dubakoor13/NUKE/actions/runs/40767503400).
- [**All 16 genuine original neutron C++ HDF5 source banks, XML physical inputs and data provenance, artifact 12852585389**](https://github.com/dubakoor13/NUKE/actions/runs/40767503400/artifacts/12852585389).
- [**Native evaluated-neutron independent-seed k and source genealogy JSON, artifact 12851975768**](https://github.com/dubakoor13/NUKE/actions/runs/40767503400/artifacts/12851975768).

GitHub's original scientific HDF5 artifacts expire approximately **2026-11-18**. Preserve them before expiration. The evaluated nuclear library is not copied into the repository; its original upstream download and exact index/archive SHA256 are recorded.
