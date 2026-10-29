"""Stage75: physically disjoint outer-fuel and inner-H2 ZrC radial contacts.

Each interface is the SAME Stage73 exact angular arc, assigned to exactly
one radial coating volume: the OUTER shell sees fuel, INNER shell sees H2.
No heat is deposited in the intervening shell or added twice. Internal
radial conduction is delegated to the conservative Stage74 FVM.

The caller must supply independent physical thermal material properties.
Shell-centre logarithmic interface distances are finite-volume
approximations; the coating's hex-clipped volumes and exact boundary arcs
are inherited from Stages 70–74. No genuine OpenMC transport is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .coating_angular_interfaces import AngularCoatingInterfaces
from .coating_radial_fvm import RadialCoatingMesh
from .fuel_cutcell_axial_transport import AxialCutCellModel
from .fuel_cutcell_nonlinear_enthalpy import NonlinearThermalProperties
from .resolved_coating_thermal import ResolvedCoatingModel


@dataclass(frozen=True)
class RadialInterfacePower:
    fuel_to_coating_by_fuel_w: np.ndarray
    fuel_to_coating_by_radial_w: np.ndarray
    coating_to_hydrogen_by_radial_w: np.ndarray
    coating_to_hydrogen_by_channel_w: np.ndarray
    fuel_antisymmetric_residual_w: float
    hydrogen_antisymmetric_residual_w: float


def radial_interface_power(
    coating: ResolvedCoatingModel,
    solid: AxialCutCellModel,
    laws: NonlinearThermalProperties,
    contacts: AngularCoatingInterfaces,
    radial: RadialCoatingMesh,
    fuel_temperature_k: np.ndarray,
    coating_temperature_k: np.ndarray,
    hydrogen_temperature_k: np.ndarray,
) -> RadialInterfacePower:
    """Return separate fuel and H2 signed watts on the correct shell faces."""
    nzone,nfuel=fuel_temperature_k.shape
    na=contacts.sector_count
    nr=len(radial.radius_edges_m)-1
    expected=(nzone,19,na,nr)
    if (coating_temperature_k.shape!=expected
            or radial.sector_mass_kg.shape!=expected
            or hydrogen_temperature_k.shape!=(nzone,19)
            or nfuel!=solid.transverse.n_fuel
            or contacts.fuel_outer_full_m2.shape!=(nfuel,19*na)
            or contacts.h2_inner_full_m2.shape!=(19,na)
            or not np.all(np.isfinite(fuel_temperature_k))
            or not np.all(np.isfinite(coating_temperature_k))
            or not np.all(np.isfinite(hydrogen_temperature_k))):
        raise ValueError("radial ZrC interface geometry and temperatures mismatch")
    if nr<2 or np.any(np.diff(radial.radius_edges_m)<=0):
        raise ValueError("radial ZrC requires at least two positive shell widths")
    edges=radial.radius_edges_m
    if (abs(edges[0]-coating.inner_radius_m)>1e-13
            or abs(edges[-1]-coating.outer_radius_m)>1e-13):
        raise ValueError("radial ZrC shell surface radii are incompatible")
    kf=laws.fuel_k_w_m_k.at(fuel_temperature_k)
    kc=laws.coating_k_w_m_k.at(coating_temperature_k)
    film=laws.hydrogen_film_w_m2_k.at(hydrogen_temperature_k)
    if (np.any(kf<=0) or np.any(kc<=0) or np.any(film<0)):
        raise ValueError("invalid fuel/coating conductivity or hydrogen film")
    length=float(solid.axial_edges_m[-1]-solid.axial_edges_m[0])
    dz=np.diff(solid.axial_edges_m)
    if length<=0 or len(dz)!=nzone or np.any(dz<=0):
        raise ValueError("fuel and coating axial extent mismatch")
    # Physical outer fuel->coating arc split by actual Stage53 fuel-cell
    # edge AND Stage72 angular sector. ONLY shell nr-1 gets its watts.
    rim=contacts.fuel_outer_full_m2.tocoo()
    cell=rim.row.astype(int)
    channel=(rim.col//na).astype(int)
    angle=(rim.col%na).astype(int)
    arc=dz[:,None]*rim.data[None,:]/length
    outer_mid=math.sqrt(edges[-2]*edges[-1])
    outer_res=(coating.half_fuel_distance_m/kf[:,cell]
               +edges[-1]*math.log(edges[-1]/outer_mid)
               /kc[:,channel,angle,-1]
               +coating.contact_resistance_m2_k_w[:,channel])
    if np.any(outer_res<=0) or not np.all(np.isfinite(outer_res)):
        raise ArithmeticError("nonphysical fuel/outer ZrC radial contact")
    flow=arc/outer_res*(
        fuel_temperature_k[:,cell]-coating_temperature_k[:,channel,angle,-1])
    fuel_out=np.zeros((nzone,nfuel))
    to_radial=np.zeros(expected)
    for iz in range(nzone):
        np.add.at(fuel_out[iz],cell,flow[iz])
        np.add.at(to_radial[iz,:,:,-1],(channel,angle),flow[iz])
    # H2 is attached to actual INNER bore arc, only at shell 0.
    inner=contacts.h2_inner_full_m2[None,:,:]*dz[:,None,None]/length
    inner_mid=math.sqrt(edges[0]*edges[1])
    coat_inner=coating_temperature_k[:,:,:,0]
    radial_res=edges[0]*math.log(inner_mid/edges[0])/kc[:,:,:,0]
    film3=film[:,:,None]
    denom=radial_res+np.where(
        film3>0,1./np.maximum(film3,1e-300),np.inf)
    inner_flow=np.zeros_like(coat_inner)
    np.divide(inner,denom,out=inner_flow,where=film3>0)
    inner_flow*=coat_inner-hydrogen_temperature_k[:,:,None]
    to_hydrogen=np.zeros(expected)
    to_hydrogen[:,:,:,0]=inner_flow
    gas_by_channel=inner_flow.sum(axis=2)
    balance_f=float(np.sum(fuel_out)-np.sum(to_radial))
    balance_h=float(np.sum(to_hydrogen)-np.sum(gas_by_channel))
    if (abs(balance_f)>1e-10*max(1.,abs(float(fuel_out.sum())))
            or abs(balance_h)>1e-10*max(1.,abs(float(gas_by_channel.sum())))):
        raise ArithmeticError("fuel/ZrC/H2 radial interface energy not antisymmetric")
    return RadialInterfacePower(
        fuel_to_coating_by_fuel_w=fuel_out,
        fuel_to_coating_by_radial_w=to_radial,
        coating_to_hydrogen_by_radial_w=to_hydrogen,
        coating_to_hydrogen_by_channel_w=gas_by_channel,
        fuel_antisymmetric_residual_w=balance_f,
        hydrogen_antisymmetric_residual_w=balance_h,
    )
