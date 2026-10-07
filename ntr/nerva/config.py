"""Configuration for the NERVA-derived demonstration model."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NervaConfig:
    """Geometry and Monte Carlo controls for the demonstration model."""

    fuel_flat_to_flat_cm: float = 1.905
    active_length_cm: float = 132.0
    coolant_bore_diameter_cm: float = 0.2565
    coolant_bore_pitch_cm: float = 0.40894
    zrc_coating_thickness_cm: float = 0.0050

    # Public NERVA-derived tie-tube dimensions.
    tie_graphite_filler_id_cm: float = 1.626
    tie_zrc_outer_diameter_cm: float = 1.613
    tie_zrc_inner_diameter_cm: float = 1.410
    tie_outer_tube_outer_diameter_cm: float = 1.397
    tie_outer_tube_wall_cm: float = 0.0205
    tie_zrh_outer_diameter_cm: float = 1.168
    tie_zrh_inner_diameter_cm: float = 0.533
    tie_inner_tube_outer_diameter_cm: float = 0.521
    tie_inner_tube_wall_cm: float = 0.051

    core_rings: int = 3
    reflector_thickness_cm: float = 5.0

    # Reactor-periphery engineering surrogates. The public NASA cross section
    # establishes the topology (12 drums in a Be reflector, Al-alloy vessel),
    # but not all dimensions below are tied to a single historical article.
    control_drum_count: int = 12
    control_drum_radius_cm: float = 1.20
    control_absorber_thickness_cm: float = 0.20
    control_drum_angle_deg: float = 0.0
    pressure_vessel_thickness_cm: float = 0.50

    uranium_enrichment_wt_percent: float = 5.0
    fuel_matrix_density_g_cm3: float = 1.85
    hydrogen_density_g_cm3: float = 0.0030
    hydrogen_temperature_k: float = 600.0
    reflector_density_g_cm3: float = 1.85
    zrc_density_g_cm3: float = 6.5
    graphite_density_g_cm3: float = 1.70
    zrh_density_g_cm3: float = 5.60
    inconel_density_g_cm3: float = 8.19
    b4c_density_g_cm3: float = 2.52
    aluminum_density_g_cm3: float = 2.70

    batches: int = 80
    inactive: int = 20
    particles: int = 4000

    @property
    def fuel_edge_length_cm(self) -> float:
        return self.fuel_flat_to_flat_cm / math.sqrt(3.0)

    @property
    def channel_radius_cm(self) -> float:
        return 0.5 * self.coolant_bore_diameter_cm

    @property
    def coated_channel_radius_cm(self) -> float:
        return self.channel_radius_cm + self.zrc_coating_thickness_cm

    @property
    def inner_fuel_flat_to_flat_cm(self) -> float:
        return self.fuel_flat_to_flat_cm - 2.0 * self.zrc_coating_thickness_cm

    @property
    def inner_fuel_edge_length_cm(self) -> float:
        return self.inner_fuel_flat_to_flat_cm / math.sqrt(3.0)

    @property
    def tie_graphite_inner_radius_cm(self) -> float:
        return 0.5 * self.tie_graphite_filler_id_cm

    @property
    def tie_zrc_outer_radius_cm(self) -> float:
        return 0.5 * self.tie_zrc_outer_diameter_cm

    @property
    def tie_zrc_inner_radius_cm(self) -> float:
        return 0.5 * self.tie_zrc_inner_diameter_cm

    @property
    def tie_outer_tube_outer_radius_cm(self) -> float:
        return 0.5 * self.tie_outer_tube_outer_diameter_cm

    @property
    def tie_outer_tube_inner_radius_cm(self) -> float:
        return self.tie_outer_tube_outer_radius_cm - self.tie_outer_tube_wall_cm

    @property
    def tie_zrh_outer_radius_cm(self) -> float:
        return 0.5 * self.tie_zrh_outer_diameter_cm

    @property
    def tie_zrh_inner_radius_cm(self) -> float:
        return 0.5 * self.tie_zrh_inner_diameter_cm

    @property
    def tie_inner_tube_outer_radius_cm(self) -> float:
        return 0.5 * self.tie_inner_tube_outer_diameter_cm

    @property
    def tie_inner_tube_inner_radius_cm(self) -> float:
        return self.tie_inner_tube_outer_radius_cm - self.tie_inner_tube_wall_cm

    @property
    def cluster_radius_cm(self) -> float:
        return self.fuel_flat_to_flat_cm + self.fuel_edge_length_cm

    @property
    def cluster_reflector_outer_radius_cm(self) -> float:
        return self.cluster_radius_cm + self.reflector_thickness_cm

    @property
    def core_radius_cm(self) -> float:
        center_radius = (self.core_rings - 1) * self.fuel_flat_to_flat_cm
        return center_radius + self.fuel_edge_length_cm

    @property
    def reflector_outer_radius_cm(self) -> float:
        return self.core_radius_cm + self.reflector_thickness_cm

    @property
    def control_drum_center_radius_cm(self) -> float:
        return self.core_radius_cm + 0.5 * self.reflector_thickness_cm

    @property
    def control_absorber_inner_radius_cm(self) -> float:
        return self.control_drum_radius_cm - self.control_absorber_thickness_cm

    @property
    def pressure_vessel_outer_radius_cm(self) -> float:
        return self.reflector_outer_radius_cm + self.pressure_vessel_thickness_cm

    def validate(self) -> None:
        if self.core_rings < 1:
            raise ValueError("core_rings must be >= 1")
        if self.active_length_cm <= 0.0:
            raise ValueError("active_length_cm must be positive")
        if self.coolant_bore_diameter_cm <= 0.0:
            raise ValueError("coolant_bore_diameter_cm must be positive")
        if self.coolant_bore_pitch_cm <= self.coolant_bore_diameter_cm:
            raise ValueError("coolant bore pitch must exceed bore diameter")
        if self.zrc_coating_thickness_cm <= 0.0:
            raise ValueError("ZrC coating thickness must be positive")
        if self.inner_fuel_flat_to_flat_cm <= 0.0:
            raise ValueError("outer coating is thicker than the fuel element")
        if not (0.0 < self.uranium_enrichment_wt_percent < 20.0):
            raise ValueError("demonstration enrichment must remain between 0 and 20 wt% U-235")
        if self.inactive >= self.batches:
            raise ValueError("inactive batches must be less than total batches")
        if self.particles < 100:
            raise ValueError("particles should be at least 100")
        if self.control_drum_count < 3:
            raise ValueError("control_drum_count must be at least 3")
        if self.control_drum_radius_cm <= 0.0:
            raise ValueError("control drum radius must be positive")
        if not (0.0 < self.control_absorber_thickness_cm < self.control_drum_radius_cm):
            raise ValueError("control absorber thickness must be between 0 and drum radius")
        if self.pressure_vessel_thickness_cm <= 0.0:
            raise ValueError("pressure vessel thickness must be positive")
        inner_drum_edge = self.control_drum_center_radius_cm - self.control_drum_radius_cm
        outer_drum_edge = self.control_drum_center_radius_cm + self.control_drum_radius_cm
        if inner_drum_edge <= self.core_radius_cm:
            raise ValueError("control drums intrude into the cylindrical core envelope")
        if outer_drum_edge >= self.reflector_outer_radius_cm:
            raise ValueError("control drums protrude beyond the Be reflector")

        radii = (
            self.tie_inner_tube_inner_radius_cm,
            self.tie_inner_tube_outer_radius_cm,
            self.tie_zrh_inner_radius_cm,
            self.tie_zrh_outer_radius_cm,
            self.tie_outer_tube_inner_radius_cm,
            self.tie_outer_tube_outer_radius_cm,
            self.tie_zrc_inner_radius_cm,
            self.tie_zrc_outer_radius_cm,
            self.tie_graphite_inner_radius_cm,
        )
        if any(radius <= 0.0 for radius in radii):
            raise ValueError("tie-tube radii must all be positive")
        if any(a >= b for a, b in zip(radii, radii[1:])):
            raise ValueError("tie-tube radial layers must be strictly ordered")
        if self.tie_graphite_inner_radius_cm >= 0.5 * self.inner_fuel_flat_to_flat_cm:
            raise ValueError("tie-tube circular stack does not fit inside the hexagonal filler")
