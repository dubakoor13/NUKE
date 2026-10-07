# Public references used for the NERVA-derived model

## Primary geometry reference

James E. Fittje, Stanley K. Borowski, and Bruce Schnitzler,
**"Revised Point of Departure Design Options for Nuclear Thermal Propulsion"**,
NASA Glenn Research Center / Oak Ridge National Laboratory.

NASA NTRS document:
https://ntrs.nasa.gov/citations/20150021278

The paper states for its NERVA-derived Monte Carlo model:

- active fuel length of 132 cm (52 in) for the 111 kN class engine;
- 19 propellant coolant channels;
- 0.2565 cm (0.101 in) coolant borehole diameter;
- 0.40894 cm (0.161 in) borehole pitch;
- 50 micrometer ZrC outer surface coating;
- a typical arrangement in which each tie tube is surrounded by six fuel
  elements;
- tie tubes with the same outer hexagonal cross section as the fuel elements.

Its tie-tube cross-section figure gives the dimensions used by this
demonstrator:

- graphite filler structure: 16.26 mm ID;
- ZrC sleeve: 16.13 mm OD x 14.10 mm ID;
- outer tie tube: 13.97 mm OD x 0.205 mm wall;
- ZrH moderator: 11.68 mm OD x 5.33 mm ID;
- inner tie tube: 5.21 mm OD x 0.51 mm wall.

The figure also identifies small cold-condition gaps between several concentric
layers. In this model those gaps are represented explicitly as void regions.

## Heritage fuel-element dimensions

NASA publications describing Rover/NERVA heritage also identify the standard
fuel element as approximately 0.75 in (1.905 cm) across flats with 19 axial
coolant channels and a 52 in (1.32 m) length.

These references are used only as public historical geometry sources. The
present OpenMC material loading and simplified Inconel/ZrH material definitions
are intentionally not a historical criticality reproduction.


## Reactor periphery and control topology

The same NASA point-of-departure paper shows the NERVA-derived reactor core
surrounded by partial filler elements, a beryllium radial reflector containing
circumferential control drums, and an aluminum-alloy pressure vessel. The
control drums use a reflector/moderator portion and a neutron absorber portion
and rotate to vary reactivity.

The public cross-section figure shows twelve control drums around the reflector.
This demonstrator therefore uses 12 drums as its default topology.

Important: the present control-drum radius, absorber-shell thickness, reflector
thickness, and vessel wall thickness are parameterized engineering surrogates.
They are not claimed to reproduce a specific NERVA/SNRE test article until a
matching public drawing or dimensions are selected for calibration.
