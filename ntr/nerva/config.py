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

    core_rings: int = 3
    reflector_thickness_cm: float = 5.0

    uranium_enrichment_wt_percent: float = 5.0
    fuel_matrix_density_g_cm3: float = 1.85
    hydrogen_density_g_cm3: float = 0.0030
    hydrogen_temperature_k: float = 600.0
    reflector_density_g_cm3: float = 1.85
    zrc_density_g_cm3: float = 6.5

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
    def core_radius_cm(self) -> float:
        center_radius = (self.core_rings - 1) * self.fuel_flat_to_flat_cm
        return center_radius + self.fuel_edge_length_cm

    @property
    def reflector_outer_radius_cm(self) -> float:
        return self.core_radius_cm + self.reflector_thickness_cm

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
