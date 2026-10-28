# Stage 45 — physically evaluated neutron temperature selection in real OpenMC

**Focus:** actual neutron transport physics and numerical selection of evaluated incident-neutron cross sections. This stage stops at the real available nuclear-data temperature coverage; it does **not** synthesize missing high-temperature evaluations or alter reactor design.

## Implemented neutron-physics functionality

The physical NERVA-derived core's `NervaConfig` now provides `neutron_temperature_method` equal to `nearest` or `interpolation`. `build_model` explicitly exports this to the **real OpenMC C++ temperature selector** in `settings.xml`. The transport CLI records the method, which is separately audited against all physically requested nuclides in the actual evaluated `cross_sections.xml`.

The new `resolve_evaluated_neutron_temperature` uses the *available neutron HDF5 `kT` tables*, not just a temperature range or a user label. For a nearest-table request it checks the exact selected table and tolerance; for interpolation it requires **two genuine adjacent evaluated tables bracketing the requested material temperature**, with a numerically computed upper-table weight `(T-Tlower)/(Tupper-Tlower)`. An exact existing one-temperature match is allowed (no interpolation is necessary), while out-of-range temperatures are rejected even if a large tolerance was requested. **No hot neutron XS extrapolation.**

The code fails closed for `interpolation` plus unqualified bound neutron `S(a,b)` kernel interpolation, and leaves the default nearest-table production behavior intact while binding its effective tolerance explicitly. Full material isotope masses, geometry, core enrichment and reactor controls are unchanged.

## Genuine v0.15.3 OpenMC C++ neutron transport and evaluated data

**[Successful original Stage45 physical transport 40761641564](https://github.com/dubakoor13/NUKE/actions/runs/40761641564)**: actual official `openmc/openmc:v0.15.3`, evaluated nuclear-data index SHA256 `71c6be06e5584e5bd23de27f148286f6a6d21d9c771e03ede629ba66741841e0`, two **genuine original neutron eigenvalue simulations** of identical cold three-ring core/materials, explicit fuel and hydrogen temperatures **293.6059129982851 K**, identical random seed 13007, 3000 source neutrons/gen, 24 total batches and 8 inactive. Original native HDF5 statepoints both saved and validated, with identical geometry and materials input XML.

**10 mathematical/model XML tests plus 2 tests loading actual evaluated neutron HDF5 passed.** The code was also checked against the existing nuclear-data regression suite.

| Original OpenMC observable | Nearest request | Interpolation request |
| --- | ---: | ---: |
| Temperature requested [K] | 293.6059129982851 | 293.6059129982851 |
| Actual evaluated neutron temperature support | **One cold table ≈294 K** | **One cold table ≈294 K** |
| C++ executed neutron history | Yes | Yes |
| Source bank at batch 24, native SHA256 | `3e8c376b34...` | **Exactly same** |
| Active-generation k arithmetic mean | **0.01083503021721516** | **0.01083503021721516** |
| Final source Shannon entropy | **5.5155276730** | **5.5155276730** |
| Maximum absolute difference between the two k-generation histories | **5.204170427930421×10⁻¹⁸ (pair comparison)** | **5.204170427930421×10⁻¹⁸ (pair comparison)** |

**Critical numerical observation:** Official OpenMC **explicitly warns** for the interpolation-request run that the C0 cross sections exist **at only one temperature**, and the C++ solver **reverts to its nearest temperature method**. The two equal C++ neutron histories therefore **do not demonstrate interpolation between distinct evaluated neutron temperatures**. The XML correctly accepts the interpolation request, but physical **multi-temperature interpolation remains unexercised** with this cold-only evaluated library. The recorded results and Stage45 JSON must distinguish *requested* versus *physically executed* method.

Existing hot-reactor temperature, thermal feedback reactivity sensitivity and coupled material feedback are **NOT qualified**, because a sole ≈294 K neutron evaluation cannot represent a hot solid fuel or hydrogen neutron spectrum correctly. A 1800 K fuel request must not be silently treated as a verified hot neutron model.

## Evidence and source code

- [Successful original C++ neutron transport and XS tests, run 40761641564](https://github.com/dubakoor13/NUKE/actions/runs/40761641564).
- [Native original evaluated neutron transport HDF5, original XML, and SHA provenance artifact 12848228686](https://github.com/dubakoor13/NUKE/actions/runs/40761641564/artifacts/12848228686).
- [Original C++ original temperature-selection/k/source physics JSON artifact 12847059817](https://github.com/dubakoor13/NUKE/actions/runs/40761641564/artifacts/12847059817).
- `model.py`, `config.py`, `audit_nuclear_data_temperatures.py`, `run_openmc_transport.py`, `real_evaluated_neutron_temperature_selector.py`, `validate_stage45_evaluated_temperature_selector.py` contain actual implemented physics, not merely documentation.

## Remaining physical qualification

1. Obtain **genuine, independently sourced evaluated neutron cross sections for the required fuel, hydrogen, reflector and other *transport-reachable* isotopes** over the physically relevant temperature interval. In particular, a valid two-temperature table bracket must exist for requested intermediate fuel temperatures.
2. Qualify high-temperature crystalline/chemically bound `S(a,b)` kernels separately. Do not apply graphite kernels blindly to mixed-phase fuel and do not extrapolate a cold Be/ZrH thermal-scattering kernel.
3. Execute **independent cold/hot OpenMC C++ source histories at two supported evaluated temperatures**, with verified material/cross-section coverage, source-convergence and uncertainty assessment. Only then estimate genuine temperature-dependent neutron response; avoid inventing reactor k or a temperature-feedback coefficient.
4. Couple scored neutron/photon deposited energy to the existing resolved solid–hydrogen energy solver with explicit externally supplied power or source normalization, preserving all heat partitions, then validate thermal state against the same evaluated material temperatures. No current results qualify power W or historical NERVA hot operating behavior.

**Nuclear core model remains a cold highly subcritical conceptual surrogate**. The temperature-selection operator is implemented and validated at a supported evaluated table, but physically realistic hot-reactor neutron transport is unfinished.
