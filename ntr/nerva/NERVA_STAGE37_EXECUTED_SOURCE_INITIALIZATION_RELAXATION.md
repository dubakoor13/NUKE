# Stage 37 — executed OpenMC source-initialization relaxation, 100 generations

**Nuclear core physics and numerical result — verified 2026-10-29 (2026-10-28 UTC).**
This is a genuine OpenMC C++ v0.15.3 coupled neutron/photon source-relaxation experiment,
**not** historical hot NERVA or a physically critical reactor.

## Original transport evidence

[Successful Stage37 original-HDF5 GitHub Actions run 40747291683](https://github.com/dubakoor13/NUKE/actions/runs/40747291683)
executed eight real independent neutron/photon OpenMC source histories (4 seed labels
13007, 23011, 33013, 43019 in each of two initial-source modes), each with:

- **3,000 starting neutrons per generation**; **100 total batches**, including
  **20 inactive**. Native saved source banks at batches **030, 060 and 100**.
- The same evaluated neutron/photon data index, 3-ring all-fuel cold demonstrator,
  geometry, materials, tally definitions and 12×12×10 physical heating mesh.
  OpenMC `geometry.xml`, `materials.xml`, `tallies.xml` SHA256 must match for
  all eight runs. `settings.xml` differs for the **initial source box only**.
- `legacy-broad`: original initial fissionable-only source box spans
  ±0.8 of the core source radius and half axial extent.
  `central-compact`: initial box spans ±0.35 of those same extents
  (minimum physical x/y half-width of 0.5 cm, not active here).
  Both are **initial-source guesses**, not changes to physical reactor design.
- Four raw source `k_generation` and Shannon entropy histories per mode, each
  required to have exactly matching saved-checkpoint history prefixes,
  genuine HDF5 source presence, RNG seed, 3,000 particles/generation, active
  realizations, valid finite numerical tallies and original source SHA256.
- **24 real original neutron/photon statepoints**; active nonoverlapping heating
  windows 21–30 (10 active batches), 31–60 (30), and 61–100 (40).
  Numerical reconstruction uses original cumulative energy tally `sum` differences
  instead of subtracting correlated standard errors. Each original window
  checks full energy mesh vs all material and neutron/photon/electron/positron
  `heating` partition conservation.
- **12/12 adversarial numerical tests and 2/2 genuine 24-HDF5 tests PASSED.**

## Actual native OpenMC observations (not simulated predictions)

| Observable | Legacy broad source | Central compact source |
| --- | ---: | ---: |
| Four-seed mean source-bank spatial TV, batch 30→100, 4×4×10 | **0.6315833** | **0.5533333** |
| Four-seed mean 3D deposited-heating shape TV, active 21–30→61–100, 12×12×10 | **0.2196655** | **0.2714739** |
| Four-seed mean *axial* deposited-heating TV, same windows, 1×1×10 | **0.1513287** | **0.2350324** |
| Active 21–30 mean heating, eV per Monte Carlo start | 908558.8314 | 866974.2752 |
| Active 31–60 mean heating, eV per Monte Carlo start | 863875.7782 | 830029.0556 |
| Active 61–100 mean heating, eV per Monte Carlo start | 864457.5252 | 864523.7593 |
| Active 61–100 total-heating sample SD between four seeds, eV per Monte Carlo start | 10484.2137 | 11490.6131 |

**Spatial distance between the two four-seed-AVERAGED fission-source site distributions**
at native 4×4×10 bank histogram resolution:

| Saved native generation | Between-initialization source-bank TV |
| --- | ---: |
| 30 | **0.2939167** |
| 60 | **0.3213333** |
| 100 | **0.2868333** |

At active window 61–100, between-initialization **four-seed-mean deposited-heating**
spatial-fraction total variation was **0.06563360** at original 12×12×10,
**0.05451314** at conservatively integrated 6×6×10, **0.05182488** at
3×3×10, and **0.04763559** for axial-only 1×1×10. The required
TV non-increase under conservative mesh coarsening passed.

Both eigenvalue source histories remained **strongly subcritical**
(observed generation multiplication factors near `k ≈ 0.012`), and
saved final fission source banks exhibited extensive position
repetition (native exact distinct-position fractions from **0.0217 to 0.0287**).
These position statistics are **not counts of independent neutron genealogies**.

### Physical interpretation

- Changing only the *initial fission-source guess* leads to different observed
  source-bank distributions at generation 100, while the between-case
  source-site TV at generation 30/60/100 is **not decreasing monotonically**.
  The experiment **does not establish a stationary eigenvalue fission source**.
- The compact-source runs exhibit *less* observed bank TV within a seed than
  broad-source runs in this experiment, but *more* temporal change in normalized
  **3D deposited heating**. This is why source-bank TV, Shannon entropy, `k`
  and heating must be assessed as **distinct observables**.
- The late-window four-seed **total heating means are close**, whereas
  spatial heat distributions still differ. Integrated energy conservation,
  agreement of total deposited energy, and seemingly stable `k`
  **do not imply convergence of the 3D heating field**.
- Four independently seeded runs per initialization and correlated fission
  descendants cannot provide calibrated frequentist coverage, unbiased
  source-entropy stationarity, reactor reactivity/criticality certification,
  or operating `W`/`W cm^-3`. Report only source-normalized **eV per
  Monte Carlo starting neutron**, not physical thermal reactor power.
- This cold small all-fuel demonstrator is **not** a geometrically and
  temperature-validated historical NASA NERVA reactor; evaluated hot
  scattering/temperature data and independent NASA benchmarks remain pending.

## Evidence retained from successful original execution

- [All 24 original OpenMC neutron+photon HDF5 source-bank statepoints, XML,
  evaluated-data provenance, SHA256, source/k/entropy checks](https://github.com/dubakoor13/NUKE/actions/runs/40747291683/artifacts/12841948627)
  (GitHub artifact ID **12841948627**, originally compressed ≈19.5 MB).
- [Full machine-readable original-observation JSON and compressed
  mode×seed×time source and heating arrays](https://github.com/dubakoor13/NUKE/actions/runs/40747291683/artifacts/12841274755)
  (artifact ID **12841274755**, original NPZ+JSON ≈250 KB).

Original GitHub Actions artifacts have limited retention (listed to
**2026-11-18**) and should be archived along with original evaluated
nuclear data provenance before they expire.

**Next core-numerics gate:** vary inactive generation counts and retain
closely spaced source-generation checkpoints in more independent long
histories, measure source-bank ancestral dependence, establish independent
source convergence and spatial tally bias diagnostics. Do not convert
cold-source-normalized heating into hot NERVA fuel loads without separately
validated reactor geometry, temperature-resolved nuclear data and thermal feedback.
