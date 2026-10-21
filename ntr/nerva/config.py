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
    fuel_temperature_k: float | None = None  # None retains legacy default OpenMC behavior
    # 'evaluated-bound' enables chemically bound thermal scattering in
    # specific PURE moderators/reflectors, but only when all required
    # evaluated S(alpha,beta) HDF5 kernels exist in cross_sections.xml.
    # NNDC test data include the four named kernels at about 296 K,
    # but cold incident-neutron XS are at about 293.6 K: exact solid-
    # temperature coverage must be checked independently. Free-gas
    # remains a surrogate, never "NASA validated".
    thermal_scattering_treatment: str = "free-gas"
    bound_solid_temperature_k: float | None = None
    # User-selected nearest nuclear-data temperature tolerance [K].
    # A small finite tolerance <=5 K does NOT replace hot evaluated data.
    bound_nearest_temperature_tolerance_k: float = 0.0
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

    # OpenMC diagnostics / transport controls. These increase observability
    # only; they do not alter enrichment, geometry, or control settings.
    photon_transport: bool = False
    # Only mesh TALLY discretization: geometry and materials remain unchanged.
    heating_mesh_dimension: tuple[int, int, int] = (48, 48, 24)
    diagnostic_energy_groups: int = 80
    axial_mesh_bins: int = 96
    entropy_mesh_xy: int = 16
    entropy_mesh_z: int = 8

    # Fine-grained repeated-cell diagnostics.
    element_instance_tallies: bool = True
    channel_instance_tallies: bool = True
    instance_axial_tallies: bool = True

    # Optional tally-driven numerical convergence. Disabled by default so
    # existing run lengths remain deterministic unless explicitly requested.
    tally_rel_err_trigger: float | None = None
    trigger_max_batches: int = 500
    trigger_batch_interval: int = 5

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
        if (
            not math.isfinite(self.bound_nearest_temperature_tolerance_k)
            or not 0.0 <= self.bound_nearest_temperature_tolerance_k <= 5.0
        ):
            raise ValueError("bound nearest temperature tolerance must be within 0..5 K")
        if (self.thermal_scattering_treatment == "free-gas"
                and self.bound_nearest_temperature_tolerance_k != 0.0):
            raise ValueError("nearest temperature tolerance only applies to evaluated-bound")
        if self.thermal_scattering_treatment not in ("free-gas", "evaluated-bound"):
            raise ValueError(
                "thermal_scattering_treatment must be 'free-gas' or "
                "'evaluated-bound'; missing evaluated data cannot be approximated"
            )
        if self.bound_solid_temperature_k is not None and (
            not math.isfinite(self.bound_solid_temperature_k)
            or self.bound_solid_temperature_k <= 0
        ):
            raise ValueError("bound solid temperature must be positive finite K")
        if (self.thermal_scattering_treatment == "evaluated-bound"
                and self.bound_solid_temperature_k is None):
            raise ValueError(
                "evaluated-bound thermal scattering requires explicit "
                "bound_solid_temperature_k and matching evaluated S(a,b)"
            )
        if self.fuel_temperature_k is not None and (
            not math.isfinite(self.fuel_temperature_k) or self.fuel_temperature_k <= 0
        ):
            raise ValueError("fuel temperature must be positive and finite")
        if not math.isfinite(self.hydrogen_temperature_k) or self.hydrogen_temperature_k <= 0:
            raise ValueError("hydrogen temperature must be positive and finite")
        if not math.isfinite(self.hydrogen_density_g_cm3) or self.hydrogen_density_g_cm3 <= 0:
            raise ValueError("hydrogen density must be positive and finite")
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
        dims = self.heating_mesh_dimension
        if (
            not isinstance(dims, (tuple, list))
            or len(dims) != 3
            or any(
                isinstance(dim, bool)
                or not isinstance(dim, int)
                or not 4 <= dim <= 256
                for dim in dims
            )
        ):
            raise ValueError(
                "heating_mesh_dimension must contain three integers "
                "between 4 and 256"
            )
        if self.diagnostic_energy_groups < 8:
            raise ValueError("diagnostic_energy_groups must be >= 8")
        if self.axial_mesh_bins < 8:
            raise ValueError("axial_mesh_bins must be >= 8")
        if self.entropy_mesh_xy < 2 or self.entropy_mesh_z < 2:
            raise ValueError("entropy mesh dimensions must be >= 2")
        if self.tally_rel_err_trigger is not None:
            if not (0.0 < self.tally_rel_err_trigger < 1.0):
                raise ValueError(
                    "tally_rel_err_trigger must lie in (0, 1)"
                )
            if self.trigger_max_batches <= self.batches:
                raise ValueError(
                    "trigger_max_batches must exceed batches"
                )
            if self.trigger_batch_interval < 1:
                raise ValueError(
                    "trigger_batch_interval must be >= 1"
                )
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
