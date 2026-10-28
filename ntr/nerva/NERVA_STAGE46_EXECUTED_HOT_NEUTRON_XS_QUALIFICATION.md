# Stage 46 — verified physical hot neutron XS qualification gate

**Accepted numerical result (2026-10-29 IST)**

NUKE Stage46 implemented an actual OpenMC core-neutronics nuclear-data gate, not another cold-source convergence study. **[GitHub Actions 40764016808 PASSED](https://github.com/dubakoor13/NUKE/actions/runs/40764016808)**: **13/13** tests passed, including **2/2 tests loading genuine OpenMC v0.15.3 evaluated NNDC neutron isotope HDF5**. The real `run_openmc_transport` CLI deliberately tested a hot uniform snapshot, rejected it before OpenMC C++ could run neutron transport and wrote the actual isotope failures.

## Core physics that is now implemented

1. `NervaConfig.structural_solid_temperature_k`, exposed as `--structural-solid-temperature-k`, assigns explicit *neutron-physics material temperatures* to the **ZrC coating, Inconel tube, boron-carbide absorber, and aluminum vessel**. Existing explicit `fuel_temperature_k`, `hydrogen_temperature_k`, `bound_solid_temperature_k` already apply to fuel, H2, and the Be/graphite/ZrH solid materials respectively. The actual OpenMC materials are updated; this is not a JSON-only temperature field. **No material chemistry, concentration, density or geometry was changed**.
2. `hot_neutron_temperature_data_qualification.py` validates the original nuclear-data audit's **actual `IncidentNeutron.kTs` per-isotope HDF5**. Every material temperature must be explicit and its nuclides must have at least two evaluated neutron temperatures and actual tabulated support at the requested state. The audited selector must agree with OpenMC's requested interpolation. It records all missing nuclides and material-specific defects; no extrapolation or fabricated Doppler broadening.
3. `run_openmc_transport --require-physical-hot-neutron-xs` now writes a **complete read-only hot-XS qualification JSON before OpenMC transport**. If unsupported, the command exits nonzero with `HOT OPENMC NEUTRON TRANSPORT REJECTED`, preserving the report. Existing cold baseline behavior is unchanged when the flag is absent. A user-supplied **legitimate multi-temperature evaluated neutron HDF5 library** can pass this data-availability gate if it supplies every needed isotope/temperature, but that **still does not qualify a hot reactor**, bound scattering, fluid EOS or thermal feedback.

## Executed genuine evaluated-neutron physics result

Stage46 explicitly requested for the current three-ring demonstration core:

| Uniform material family | Temperature requested |
| --- | ---: |
| Conceptual fuel solid | 900 K |
| Hydrogen coolant | 600 K |
| Be/graphite/ZrH solid materials | 600 K |
| ZrC coating, metal tubes, boron carbide, Al vessel | 600 K |
| Neutron XS selector | OpenMC interpolation |
| Available data | Genuine official NNDC OpenMC neutron HDF5 |

**Actual evaluated nuclear-data inventory: 29 required neutron isotopes, each with exactly ONE evaluated temperature, 293.6059129982851 K.** None of the 29 provided genuine multi-temperature kT support for that hot request. The cold-only evaluated data index was not silently reused as a hot material evaluation.

All 29 isotopes with inadequate evaluated hot-temperature coverage:

`Al27, B10, B11, Be9, C0, Cr50, Cr52, Cr53, Cr54, Fe54, Fe56, Fe57, Fe58, H1, H2, Ni58, Ni60, Ni61, Ni62, Ni64, U234, U235, U236, U238, Zr90, Zr91, Zr92, Zr94, Zr96`.

Original verified evidence:
- **All 13/13 physical numeric tests passed**, including actual evaluated isotope HDF5 tests and explicit real OpenMC material temperature assignments.
- **Real `run_openmc_transport` hot request failed closed**, exactly as required, before export/transport. The job **proved that no `statepoint.24.h5` file was produced** for the invalid hot model.
- [Complete original per-isotope kT HDF5 audit, nuclear-data report, failed-hot-request log, Docker image provenance, artifact 12848619839](https://github.com/dubakoor13/NUKE/actions/runs/40764016808/artifacts/12848619839).

The data gate is **deliberately conservative**: it checks every **declared** NERVA model material rather than only root-reachable cells. That prevents accidental implicit default temperatures but may require isotope tables for unplaced materials in a small-core experiment; this is not an assertion that all nine material families occur in every geometry.

## Distinguish data availability from physically validated hot NERVA

A true multi-temperature neutron library is a **necessary**, not sufficient, condition. The model continues to use conceptual fuel chemistry and a highly subcritical surrogate. Its uniform material temperatures and fixed-density parameterization do **not** solve thermal expansion, hydrogen EOS/pressure changes, nonuniform hot-channel thermal transport, validated S(α,β) scattering at operating temperature, thermal feedback of evaluated composition, fission-source stationarity, or historical NASA hot NERVA benchmark physics.

OpenMC offers published multi-temperature evaluated libraries for incident neutrons; for example, its official ENDF/B-VII.1 releases list **250, 293.6, 600, 900, 1200 and 2500 K** nuclear data, while the installed NNDC *test* library includes only a cold ~293.6 K set. Source: https://openmc.org/data/ . Full published libraries can be large; **Stage46 did not download, alter, fabricate or run** them.

**Next core-physics execution milestone:** supply and independently fingerprint a genuine multi-temperature evaluated neutron + photoatomic library with required isotope coverage. Prove actual **two-table neutron XS interpolation at a temperature strictly between evaluated tables**, rather than a single-table OpenMC fallback. Then compare neutron transport over supported material snapshots with multiple independent seeds, while clearly distinguishing fixed-density temperature sensitivity from fully converged neutronic–thermal–hydraulic feedback.

**No unverified physical reactor watts or hot NERVA qualification was claimed.**
