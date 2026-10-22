# Emrich (2023) → NUKE core physics: traceable reference and validation plan

**Source:** William J. Emrich Jr., *Principles of Nuclear Rocket Propulsion*, **second edition** (Butterworth-Heinemann / Elsevier, 2023). PDF already committed in the repository root: [Emrich 2023 PDF](../../William%20J.%20Emrich%20Jr.%20-%20Principles%20of%20Nuclear%20Rocket%20Propulsion-Butterworth-Heinemann%20%282023%29.pdf).

**ISBN:** 978-0-323-90030-0 (print); 978-0-323-90031-7 (electronic). Publisher edition listing: https://www.educate.elsevier.com/book/details/9780323900300. The printed chapter-opening page numbers in the table below are taken from the **2023 second-edition public table of contents**: https://books.google.com/books?id=haCSEAAAQBAJ.

**Important:** The printed page number in the book is **NOT** necessarily the same as the PDF viewer's 1-based page number. The repository's `emrich_reference_audit.py` opens the **actual checked-in PDF**, records SHA-256 and all chapter-heading page candidates, and stores a machine-readable `emrich_2023_pdf_chapter_audit.json` as a GitHub Actions artifact. Use its page numbers to open and visually inspect any quoted equation or figure; **do not invent equation, figure, table, or PDF page numbers**. No equations or numerical historical test datasets have yet been certified as extracted.

## What this source can and cannot validate

| Level | Evidence needed | Validating role of Emrich |
|---|---|---|
| Governing neutron equations | Derivation plus unit/boundary/normalization checks | Secondary theoretical reference; inspect actual equations before citing specific equation numbers |
| Algorithm numerics | Independent benchmark with known numerical reference and tolerance | May provide example problems; textbook examples are **not** a NASA reactor benchmark |
| Evaluated nuclear data | Nuclide HDF5, temperature grids, bound S(a,b), ENDF reaction data, SHA-256 | Describes theory/interpretation; **not** a substitute for evaluated physical cross sections |
| Historical reactor measurements | Traceable NASA report, test article/configuration, operating point, uncertainty | Appendix III compiles selected NERVA data; trace each value back to NASA primary records |
| Genuine code execution | Actual OpenMC C++ outputs with statepoint provenance and independent seed checks | Does **not** replace actual transport evidence |
| Reactor acceptance | Verified full-core geometry, fuel inventory, source convergence, material temperatures, power deposition, test data | **Not satisfied** merely by referencing this book |

**Document version caution:** A 2016 *first edition* is widely indexed online and its pagination/sections differ from the 2023 second edition. Never copy equation or page citations from the 2016 edition without checking the committed 2023 PDF.

## Core-neutron physics chapter crosswalk

| Book (2023 printed opening page) | NUKE source/check | Current evidence | Important **unfinished** physical checks |
|---|---|---|---|
| Ch. 5 — Basic nuclear structure and processes, p. 59 | `audit_nuclear_data_temperatures.py`, `heating_data_qualification.py`, `executed_material_kerma_qualification.py` | Genuine evaluated NNDC HDF5 inspection; actual 13 root-reachable isotopes; cold ~293.6 K data | Hot-temperature neutron evaluations, physically paired S(a,b), neutron-local heating MT901 |
| Ch. 6 — Neutron flux energy distribution, p. 97 | `neutron_energy_groups.py`, `source_energy_audit.py`, `source_phase_space_audit.py` | Source energy/phase-space distributions from actual four-seed statepoints | Neutron field spectrum validation against independent data; no inference of temperature-qualified source from a cold spectrum |
| Ch. 7 — Neutron balance equation and transport theory, p. 107 | `core_neutron_balance.py`, `core_neutron_leakage_history.py`, `core_neutron_economy_validation.py` | Native neutron absorption, vacuum leakage, multiplicative scattering and *next-generation* banked fission traced to 20 actual HDF5 statepoints; conservation passed | Resolve nonstationary/cloned fission source, independent ancestry and valid uncertainty; no equivalence to a critical NERVA core |
| Ch. 8 — Multigroup neutron diffusion equations, p. 115 | `source_resolution_audit.py`, `core_criticality_physical_qualification.py` | Genuine continuous-energy OpenMC exists; separate multigroup diffusion benchmark of same state is **not** demonstrated | Independently derived group constants, leakage/boundary conditions, geometry fidelity, numerical residuals and comparison limits. **Do not** invent group constants or claim diffusion equals transport |
| Ch. 9 — Thermal fluid aspects of nuclear rockets, p. 135 | `heating_data_qualification.py`, `thermal.py` | Thermal/flow solvers exist; executed local deposited neutron heat is still blocked | Evaluated MT901/heat balance, physically coupled hot temperatures, conservation and experimental temperature measurements |
| Ch. 11 — Nuclear reactor kinetics, p. 187 | `openmc_transient.py` | Software exists, historical reactor kinetics not qualified | Qualified delayed-neutron parameters and source kinetics; do not claim transient reactor operation from cold static transport |
| Ch. 13 — Fuel burnup and transmutation, p. 251 | `model.py`, documented depletion planning | No physical depletion validation of current cold surrogate | Source-verified fission yields, flux histories, evolving nuclides and depletion benchmark; defer until core state and nuclear data qualified |
| Ch. 15 — Materials for nuclear thermal rockets, p. 287 | `core_material_temperature_qualification.py`, `physical_scattering_pair_qualification.py` | Four actual transported materials from nine definitions, one explicit material temperature | Reference-backed historical fuel composition/loading and material state; temperature-appropriate free/bound neutron data |
| Ch. 16 — Nuclear rocket engine testing, p. 305 | `physical_core_reference_validation.py`, `NASA_FIDELITY_MATRIX.md` | NASA-derived geometric element comparisons, not historical full-core validation | Identical test-article configuration, primary NASA operating/test records, measured uncertainty and independent benchmark |
| Appendix III — Selected data from NERVA tests, *printed page to be verified in PDF* | NASA references and benchmark data registry | Appendix existence supported by publisher contents; **test values not yet transcribed or independently verified** | Page/table/test identity audit of 2023 book **then** independent NASA primary-source validation. Keep XE-Prime, 1972 design and later SNRE concepts distinct |

## Work sequence: evidence-based physics first

1. **Pin textbook provenance (this change).** Verify the actual repository PDF's edition, SHA-256, chapter-heading locations and printed-page/PDF-page distinction. Publish only a small metadata JSON artifact, **not** a redistributed copy of the book.
2. **Reconcile chapter 7 with existing genuine OpenMC conservation checks.** The physical *one-generation* identity is neutron absorption + vacuum leakage − net neutron multiplicative scattering = one starting source neutron; fission daughters enter the *next* bank. Numerical closure alone is insufficient to establish genealogy/source stationarity.
3. **Audit chapter 6 against actual energy tallies.** Compare source and material flux spectra using exact energy thresholds without interpolating unknown within-group spectra. Annotate what is a directly scored reaction vs a source-bank histogram.
4. **Chapter 8: independent approximation validation plan.** Determine whether a diffusion-theory comparison has adequately qualified transport/diffusion boundary assumptions and independently evaluated group constants. Do **not** choose constants to force agreement with OpenMC or k=1.
5. **Chapter 5 and 15: nuclear data/material completeness.** Establish hot physical material temperatures, evaluated neutron data at those temperatures, genuinely material/phase matched thermal scattering, and neutron-local heat deposition. Currently *not physically qualified*.
6. **Appendix III and Ch.16: measured-history crosswalk.** Create a source-cited list of NERVA test article, configuration, measurement type and uncertainty **only after page/table inspection**. Trace historical values to independent NASA technical reports before using any as validation.
7. **Final independent reactor validation.** Require full-core geometry/material reference, converged independent neutron source, physical energy deposition and independent NASA benchmark; do not confuse a cold 19-element smoke model with a historical critical reactor.

## Scope and scientific non-claims

The executed NUKE reference job is a **cold, three-ring, 19-all-fuel-element surrogate**, not a configured full historical Rover/NERVA test reactor. Genuine OpenMC runs remain strongly subcritical (k≈0.0117) and have highly cloned spatial fission-source positions. The four active physical neutron materials are the fuel surrogate, hydrogen coolant, ZrC coating and beryllium reflector. Unplaced graphite, zirconium-hydride, Inconel, absorber and aluminum definitions are not evidence of transported tie tubes, drums or vessel. Missing local MT901 data prevent qualified neutron-local deposited heat.

**Nothing in this crosswalk changes any reactor configuration, materials, enrichment, control drums, source settings, nuclear evaluations or solver tolerances.** No historical numeric test value, equation number, or figure content is asserted to have been verified until the actual 2023 PDF and original NASA test report have been inspected.
