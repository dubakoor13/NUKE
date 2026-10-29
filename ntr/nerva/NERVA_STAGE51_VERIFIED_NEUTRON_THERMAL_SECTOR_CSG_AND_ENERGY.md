# Stage 51 — verified neutron-transport to thermal-sector geometry and energy conservation

**Status: genuine OpenMC v0.15.3 C++ neutron geometry-debug execution PASS; 12/12 new tests PASS; legacy and Stage50 CSG regression PASS.** This is a real geometry/thermal-discretization physics correction, not a nuclear design optimization.

- [**Original full successful Stage51 job, 40821015466**](https://github.com/dubakoor13/NUKE/actions/runs/40821015466) — all 10 numerical tests + 2 true OpenMC CSG tests passed, legacy `validate_layout`, `validate_fuel_sector_geometry`, `validate_fuel_sectors` and 2 original Stage50 material CSG tests passed; then **real native C++ OpenMC geometry-debug run** executed, producing original `statepoint.24.h5`, confirmed valid neutron eigenvalue generation history, XML, SHA and cold nuclear-data reports.
- [Independent PR original 12-test verification, 40821047672](https://github.com/dubakoor13/NUKE/actions/runs/40821047672) — also successful.
- [Original numerical/CSG/energy physics JSON and C++ SHA artifacts (12874443112)](https://github.com/dubakoor13/NUKE/actions/runs/40821015466/artifacts/12874443112).
- [Actual C++ geometry-debug OpenMC neutron HDF5 and full original XML/XS audit (12874662418)](https://github.com/dubakoor13/NUKE/actions/runs/40821015466/artifacts/12874662418).

## Two independently discovered coupled neutronic–thermal defects

**1. Wrong fuel thermal mass in the pre-existing 19 Voronoi thermal sector model.** Stage50 established that the physically modeled inner hex is a vertical-flat `openmc.model.HexagonalPrism(orientation="y")` and six outer channel-coating circular footprints intersect the inner prism boundary. The original `fuel_sector_fuel_areas_cm2` blindly removed **19 full coated circles** from 19 inner-hex Voronoi sectors even though parts of six outer circles lie outside the fuel matrix region. That omitted **0.000899530469757881 cm²** of actual OpenMC fuel matrix area (not a change in nuclear geometry), corresponding to **0.118738022008 cm³** over the 132-cm element.

Stage51's new `fuel_sector_coated_channel_areas_cm2` calculates **exact per-hole circular-segment intersections** with all six true vertical-flat faces, rejecting overlaps, bore escape, touching the outer prism and ambiguous two-face intersections. The fuel-sector thermal areas subtract the **actual 19 intersection footprints**, with full area for 13 channels and clipped areas for six outer channels.

**2. Actual OpenMC coating cells overlapped in six exterior caps.** The physical ZrC **material** union had been correct, but `build_fuel_element_universe` gave each full circular channel-coating CSG cell a region `(+inner_cylinder & -outer_cylinder)` with no `-inner_hex` clipping, while the exterior-coating cell was `+inner_hex`. Both CSG cells thus owned the same small cap positions, despite identical ZrC composition. Stage50's *first-match* `Universe.find` material classification could not reveal two simultaneous cell owners. Stage51 fixes the **cell ownership** by intersecting each channel ZrC ring with `-inner_hex`. The remaining cap is owned by the external ZrC cell; no nuclear material composition, coating/bore union, physical dimensions, or enrichment changes.

## Original observed exact geometry and thermal volumes

| Physical / thermal observable | Old inaccurate thermal sector | New real OpenMC CSG and thermal-sector value |
| --- | ---: | ---: |
| Fuel matrix cross-sectional area [cm²] | 2.0500847261282797 | **2.0509842565980376** |
| Fuel matrix volume at 132 cm [cm³] | 270.611183848933 | **270.729921870941** |
| Missing fuel material volume [cm³] | **0.118738022008** | **0** |
| Relative old-to-correct fuel mass correction | — | **+0.0438777%** |
| Sum of 19 thermal sector areas versus independent OpenMC fuel area | **Incorrect** | **Exact to machine precision** |
| Equivalent annular thermal cell volumes [m³] | 0.000270611183848933 | **0.000270729921870941** |
| Clipped channel annuli | Ignored | **6** |
| Channel-ZrC/exterior-ZrC double-owned CSG cap cells | **6 overlapping caps** | **0 overlap at controlled and sampled points** |

The per-sector **equivalent annulus** represents the same solid volume, mass and constant-property heat capacity as the real 19-channel fuel. It **does not** faithfully reproduce noncircular local 2D conduction paths, exact hexagonal wall heat-transfer coefficients or actual reactor thermal power.

### Independent OpenMC material-cell and C++ neutron verification

Actual `openmc.Universe.find` **AND the membership of every individual `Cell.region`** were checked at **3,607 distinct actual element points**: 3,500 deterministic-seed random uniform points inside the real y-hex plus controlled hydrogen/channel/interface positions, including independently constructed points at all six previously overlapping ZrC cap locations. All sampled positions belonged to **exactly one OpenMC CSG cell**; all original fuel, hydrogen and ZrC materials were present. Source field identity and material assignment remain the existing conceptual NERVA-derived model.

- Original physical material hit counts: **2,249 fuel, 1,226 hydrogen, 132 ZrC**.
- Six original cap channel indices: **14, 15, 16, 17, 18, 19**.
- Actual neutron C++ engine: official `openmc/openmc:v0.15.3`, **geometry debugging ENABLED**.
- Nuclear state: original supported evaluated **293.6059129982851 K** library, 3,000 particles/generation, 24 batches, 8 inactive and original seed 13007. **Native `statepoint.24.h5` verified** with positive finite generation-`k` and full original XML isotope/temperature provenance.
- Native descriptive generation mean `k` over batches 9–24: **0.011340110260039851**. This is solely a diagnostic for a strongly subcritical, non-source-qualified surrogate. No operating reactor multiplication, source stationarity, watts or historical NERVA behavior is claimed.

### Numerical thermal energy-conservation demonstration (explicitly SYNTHETIC)

The corrected sector fuel volumes were passed directly to the existing **conservative, sparse implicit solid/hydrogen conjugate thermal solver** via its `ConjugateCoreGeometry` volume definition. As a mathematical unit test, a *synthetic* uniform `1e6 W/m³` forcing is supplied to all 19 modeled equivalent fuel annuli for 1 s, with **synthetic constant** solid properties `ρ=1000 kg/m³`, `cp=1000 J/(kg·K)`, no coolant flow or wall conductance.

- Corrected exact total fuel thermal volume: **0.000270729921870941 m³**.
- Fuel temperatures increase from 300 to **301.00000000000003 K**, all 19 cells, as required by **1-K analytic energy storage**.
- Hydrogen stays 300 K (no applied hydrogen energy or wall conductance).
- Actual signed thermal energy conservation residual **6.650680006714538×10⁻¹² J**.
- There is **no OpenMC heating tally or externally qualified source-power normalization** in this test; neither the synthetic 1e6 W/m³ nor its total may be interpreted as physical NERVA reactor watts or temperatures.

## NASA physical fidelity and remaining work

The original component source remains [NASA NTRS 20150021278](https://ntrs.nasa.gov/api/citations/20150021278/downloads/20150021278.pdf), element figures and original 132 cm component. The 19-element NUKE surrogate is not a complete historical NERVA reactor, and the 1.905/1.913 cm drawing ambiguity remains unresolved. Even with the correct CSG material partition and thermal fuel mass, the model still lacks source-stationary and historically validated hot core neutron/photon energy deposition, realistic temperature-dependent bound scattering and gas EOS, real flow/pressure boundary conditions and full historical fuel inventory.

**Next real physics step:** retain the existing exact actual OpenMC geometry/thermal volume fingerprint and independently calibrate the **2D/3D thermal conduction operator and channel wall/contact areas** on a geometry-resolved representative fuel section. Volume-equivalent radial annuli preserve heat capacity but do not reproduce the conduction operator of a hexagonal multi-channel solid. Do not overstate a source-normalized C++ heating score as physical reactor watts without independently specified source normalization.
