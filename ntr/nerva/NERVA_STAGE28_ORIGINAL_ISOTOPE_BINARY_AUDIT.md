# NUKE Stage 28 — reverify ORIGINAL neutron isotope HDF5 binary identity

**Date:** 2026-10-26  
**Stacked parent:** Stage 27 [PR #140](https://github.com/dubakoor13/NUKE/pull/140).  
**Original transport source:** 2026-10-23 real four-seed cold neutron OpenMC 0.15.3 [Actions run 39933618638](https://github.com/dubakoor13/NUKE/actions/runs/39933618638), binary artifact 12492499186.

## Why this is a physical-evidence improvement, not another guessed solver

The Stage 25 HDF5 material audit verified the published **claims** of 29 evaluated isotope HDF5 filename/size/digest records. The source artifact retains actual OpenMC `summary.h5`, statepoints and reports, but **does not include these nuclear-data binaries**. Until Stage 28, an independent re-reader could not recompute any claimed isotope HDF5 digest. The original cold geometry uses 13 of the 29 indexed isotopes, and its only evaluated isotope temperature is approximately 293.605913 K.

Stage 28 compares independently located **actual HDF5 bytes** to the original transport-run per-file SHA256 and size claims. It uses the same public `openmc/openmc:v0.15.3` container image as the original GitHub Actions run, without re-optimizing, modifying, or rerunning the reactor. The exact candidate Docker image digest/ID is retained as run evidence.

The read-only code `ntr/nerva/verify_original_xs_binary.py` checks:
1. Original four-seed statepoint/source and native summary/material identity via the existing Stage 21–26 evidence gates.
2. Candidate `cross_sections.xml` bytes against the original full SHA-256 recorded in the genuine transport-run summary.
3. Exact indexed 29 isotope filenames, byte sizes, file paths and **recomputed actual binary SHA256** against `verified_xs_files.json` from the original run, with no replacement, omission or fallback.
4. Opens every reverified true HDF5 through `openmc.data.IncidentNeutron.from_hdf5`, verifies the native isotope identity and compares every original temperature knot to actual `kTs/K_BOLTZMANN`.
5. Independently inspects native actual MT901 KERMA curve presence and numeric usability using the existing `mt901_numeric.py`, reporting genuine missing or zero local-energy coefficient data rather than generating substitutes.
6. Records which isotopes were physically root-reachable in the original transport, explicitly retains hot-core, source stationarity and material-resolved nuclear heating **BLOCKED**.

Seven adversarial unit tests verify actual SHA rejection, missing isotope bytes, duplicate/malformed manifests, candidate-index/path differences, and correct match behavior. These unit byte fixtures are **not physical nuclear data**. The independent Docker test is the only place genuine real isotope HDF5 is claimed.

## Unresolved scientific requirements

Matching an original HDF5 digest does **not** validate that the dataset is a certified ENDF release, matches historical NERVA hot fuel or bound scattering, contains suitable evaluated gamma transport/heating data, or supports physical watt normalization. All original model limits remain: **cold surrogate far below critical**, nonstationary/cloned fission source, absent useful `heating-local` energy scores, zero certified hot material temperatures, and no NASA reactor hold-out match.

The original archive still lacks the original isotope binaries even after this stage. CI keeps a bounded JSON evidence report and a Docker image identity manifest rather than redistributing third-party nuclear HDF5 files. Archive preservation and data-release licensing remain separate tasks.

**Status at creation:** original-data CI not yet executed. Report the check PASS only after new [workflow](../../.github/workflows/nerva-original-xs-binary.yml) completes against genuine archive and image.


## Verified actual OpenMC isotope-binary execution — 2026-10-26

**[GitHub Actions run 40303149266](https://github.com/dubakoor13/NUKE/actions/runs/40303149266) completed SUCCESS**, exact source-archive artifact digest `cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274`. The publicly retrievable OpenMC `v0.15.3` Docker image download reported manifest digest `sha256:dd73b9c7a0e1a1c956a05706ec813e43549f9bd6fc4cc32d4fbf94c676a60978` (Docker-manifest digest, distinct from raw HDF5 SHA).

- **7/7** deliberately altered-byte/missing-file/wrong-index local tests passed.
- **29/29 actual native isotope HDF5 files** independently SHA256-and-byte-size matched the original executed-run `verified_xs_files.json` records; the original evaluated `cross_sections.xml` binary SHA was also matched.
- All **29 original isotope HDF5** identities and temperature knots were reopened through the installed Docker OpenMC `IncidentNeutron.from_hdf5` API.
- The archived original geometry's **13 physically root-connected isotopes** were identified and checked.
- **Complete native MT901 local KERMA numerical coverage for the 13 root-used isotopes: FALSE**, reconfirming a material power-deposition precondition is not satisfied. There was NO artificial MT901, `heating-local`, neutron/photon energy or reactor-watt source substitution.
- Original evaluated release *edition* and license provenance remain uncertified; root-used data are cold only (~293.605913 K), no independently stationary fission source, no suitable hot NERVA data or measured historical core comparison.
- Compact observed evidence was uploaded as artifact `nuke-stage28-original-neutron-data-binary-reverification` (ID **12641970366**, reported expiry **2027-01-24**). This holds digests/results, **not** the third-party neutron HDF5 bytes. The original OpenMC statepoint artifact's separate reported expiry remains **2026-11-13**.

**What this changes:** the Stage 25 gap "isotope file SHA strings not independently checked against original bytes" is now experimentally resolved **for the present downloaded `v0.15.3` Docker image**. Long-term actual binary preservation, evaluated release certification and hot material physics are NOT resolved.

**New critical next step:** acquire independently licensed/identified evaluated neutron+photon data appropriate to the specified operating state, verify material-specific temperatures and MT901/photon energy accounting, and only then rerun real OpenMC transport. An unqualified neutron-only cold statepoint cannot be turned into locally deposited watts by statistical postprocessing.


### Exact original material-reachable isotope MT901 observation (actual HDF5)

After Stage 28 succeeded, the saved [actual JSON artifact 12641970366](https://github.com/dubakoor13/NUKE/actions/runs/40303149266/artifacts/12641970366) was downloaded and inspected independently. Each of the **13** root-reachable real evaluated neutron HDF5 isotope records had exactly **one** cold temperature knot and reported **`native_MT901_present: false`** (therefore `native_MT901_numerically_usable: false`):

```text
Be9, C0, H1, H2, U234, U235, U236, U238,
Zr90, Zr91, Zr92, Zr94, Zr96
```

Thus the original cold test library's local KERMA missing coverage is **13/13 actual root-used neutron isotopes**, not merely an incomplete row or uncertainty from absent file bytes. This is now a direct observation from re-opened real isotope binaries (cross-index and SHA verified), **not a fabricated reactor heating substitute**.

The justified engineering conclusion is **no neutron-only MT901-dependent local heat normalization from this dataset**. The original `heating` and `kappa-fission` signals are not automatically a material-disjoint neutron+photon deposited energy map. An independently qualified data library with correct reaction/temperature/phase/photon coverage is a mandatory upstream prerequisite for the next hot-core physics step.
