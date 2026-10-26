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
