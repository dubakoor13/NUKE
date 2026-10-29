# Stage 50 — NASA 2015 real OpenMC fuel-element and tie-tube CSG material/volume physics

**Status: real OpenMC CSG verified against independent analytical geometry**, original model unchanged. [Stage50 complete **13/13** tests plus the existing fast layout validator, GitHub Actions 40816631372 (PASS)](https://github.com/dubakoor13/NUKE/actions/runs/40816631372). The old layout helper was corrected to use the same hex orientation as OpenMC; an additional regression verifies the **-0.00363 cm** real inner-coating clearance and a positive **+0.00137 cm** hydrogen bore-to-inner-hex clearance.

## Source-specific NASA geometry pedigree (no mixed reactor baselines)

**Primary original reference:** Fittje, Borowski and Schnitzler, *Revised Point of Departure Design Options for Nuclear Thermal Propulsion*, NASA NTRS **20150021278**, [official PDF](https://ntrs.nasa.gov/api/citations/20150021278/downloads/20150021278.pdf), PDF **pages 6–8**, Figs. 4–6, Section III.A.

This is a **reference for NERVA-derived component geometry**, not evidence that NUKE's complete three-ring 19-element surrogate is a reconstructed experimental nuclear reactor. The paper distinguishes **89 cm** (small criticality-limited concept) from **132 cm** (111 kN / 25 klbf class) active lengths; Stage50 uses only the 132-cm component reference. It describes nominal **19 hydrogen holes per fuel element**, **0.2565 cm bore**, **0.40894 cm pitch**, and **50 micrometre (0.005 cm) ZrC external coating**.

**Source drawing ambiguity preserved:** Fig. 5 labels the across-flats width **1.913 cm**, while Fig. 6 labels **1.905 cm**. The unchanged NUKE model uses **1.905 cm**, but this is NOT a verified fabrication dimension/tolerance.

The Fig. 5 tie-tube section independently identifies inner/outer Inconel tube walls, ZrH moderator, ZrC sleeve, hydrogen supply and return circuits, cold void gaps, graphite filler, and outer coating. The source explicitly says its small illustrated radial gaps describe **cold**, not hot operating conditions.

## Actual unmodified OpenMC 19-hole fuel-element cross section

**Substantial real CSG finding:** OpenMC `model.HexagonalPrism(orientation="y")` has two sides parallel to the **y-axis (vertical flats)**. The previous helper `layout.outer_channel_margin_cm` used normals for a hex with horizontal flats and falsely reported a **positive 0.105945-cm** margin. The original OpenMC y-oriented geometry has six outermost coated-channel rings crossing the **inner hex** boundary by **0.003630 cm (0.03630 mm)**. The adjacent exterior ZrC coating material cell owns the tiny cut-off segments. The H2 bores remain contained within the inner prism, and each complete coated circle stays within the outer hex.

The overlap area, calculated by the **exact circular-segment formula** and confirmed by real OpenMC material CSG classification, is **0.0008995304697577301 cm²** summed across six channel rings. The original model's fuel material area is therefore slightly larger than the earlier naive full-annulus subtraction; the effective channel-coating area **inside the inner hex** is correspondingly smaller. **No physical geometry was rotated or resized**. The existing fast layout helper and test now report the correct orientation and coating-clipping warning instead of giving a false-positive geometric clearance.

| Exact current-model component | Cross section (cm²) | Volume over 132 cm (cm³) |
| --- | ---: | ---: |
| Fuel matrix inside actual OpenMC cells | **2.050984256598** | **270.729921870941** |
| 19 original hydrogen bores | **0.9817891339999** | **129.596165687991** |
| ZrC channel annuli within the inner hex | **0.0771454850270** | **10.1832040235702** |
| Outer continuous hexagonal ZrC coating | **0.0329089653438** | **4.34398342538271** |
| **Total physical element** | **3.142827840969** | **414.853275007885** |

Exactly closed to machine precision in cm²/cm³; all four material assignments are distinct CSG regions even though the same ZrC material fills two categories. Nominal coolant-area fraction **31.2390%**, fuel-matrix fraction **65.2592%**. Total 19-channel wetted H2 bore perimeter **15.31055 cm per cross section** and actual H2-wetted axial wall area **2020.99284 cm² per 132-cm element**; these are **geometric reference values**, not fluid heat-transfer coefficients or reactor power.

Additional source-region checks: minimum *coated-channel-to-coated-channel* ligament **0.14244 cm**, minimum hydrogen bore-to-inner-hex clearance **0.00137 cm**, minimum coated-channel-to-outer-hex clearance **0.00137 cm**, and the six clipped channel ring inner-hex clearance **−0.00363 cm**.

## Actual original OpenMC 11-region coaxial tie tube

Independently closed the current CSG tie-tube stack with analytical annular shell areas, including the noncircular graphite-to-inner-hex region and outer coating. Each of 11 radial/surface zones has nonzero, nonoverlapping CSG area, including separate **H2 supply, H2 return, two Inconel tubes, ZrH moderator, ZrC sleeve, graphite filler, three cold void gaps and outer hex ZrC coating**. Physical numeric values:

- H2 supply area **0.1378852869642 cm²**, H2 return **0.3726808533101 cm²**; total H2 flow cross section **0.5105661402743 cm²**.
- Sum of the three *source-described cold-condition void gaps* **0.07166444081736 cm²**.
- ZrH moderator cross section **0.8483360452213 cm²**; graphite filler **1.0334235209744 cm²**; ZrC sleeve **0.4819745055339 cm²**.
- All 11 shell/hex regions sum to the same **3.142827840969 cm²** outer hex, within (10^{-12}) cm², with positive shell widths and proper radial ordering.
- This is not a verified hot tie-tube expansion/thermal stress model. It must not be used to infer operating clearances.

## Genuine OpenMC CSG numerical validation

The official `openmc/openmc:v0.15.3` Python geometry objects used by NUKE's transport are inspected **directly**, not only input-file dimensions: `build_fuel_element_universe`, `build_tie_tube_universe`, and actual `openmc.Universe.find`. An independent analytic material classifier checks deterministic radial points at all 19 coolant passage centers and annular boundaries, all tie-shell boundaries, exterior flats, and additional fixed-seed sample points **inside the true y-oriented hex**.

**Original accepted numerical result:** **2,025** fuel-element actual CSG queries plus **1,841** tie-tube CSG queries, **all matched** the independent actual geometric containment/identification calculation, including H2, fuel, ZrC, Inconel, ZrH, graphite and true void regions. The tests also reject nonphysical channel overlap/hex escape, inverted tie-shell ordering, and historical reactor overclaiming.

[Original successful Stage50 CSG/material physics job, 13/13 PASS](https://github.com/dubakoor13/NUKE/actions/runs/40816631372); [exact numerical material-volume/geometry CSG JSON artifact 12872696110](https://github.com/dubakoor13/NUKE/actions/runs/40816200340/artifacts/12872696110).

## Model physics limitations and next priority

The **current three-ring, 19-element all-fuel NUKE core remains a conceptual demonstrator**, not either of the complete 2015 NASA NERVA-derived reactor configurations and not a tested historical NERVA core. Its source-element geometry dimensions alone do not verify fuel atom fractions, the source-specific UC/ZrC/graphite composite distribution, uranium inventory, tie-tube population, reflector, control drum k worth, hot hydrogen density/EOS, power density, fission source convergence or reactor criticality.

**Future geometry improvement:** if and only if source-specific geometry orientation is established as required, separately compare an **x-oriented hex** (horizontal flats) consistent with the NASA drawing and ring/sector placement, and quantify resulting physical effects with actual OpenMC. Do not silently rotate the existing model or mix the 1.913 cm and 1.905 cm dimensions. The geometry/thermal contact assumptions at the six clipped coatings merit explicit thermal-interface review. **No enrichment, geometry or control changes were introduced by Stage50.**
