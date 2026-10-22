"""Actual OpenMC fixed-source neutron spatial-response audit.

Use the eight genuine cold-core fission ON/OFF HDF5 statepoints. Independently
reconcile whole-core 3D and axial tracklength neutron flux, fission and
nu-fission integrals, then compare the four paired-seed spatial responses.
No parameter tuning, eigenvalue inference or historical-NERVA validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

SCORES = ("flux", "fission", "nu-fission")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(x) -> str:
    if not isinstance(x, (bytes, np.bytes_)):
        raise ValueError("expected a native OpenMC HDF5 byte-string")
    return bytes(x).decode("utf-8").rstrip("\x00")


def _mesh(state: h5py.File, tally_name: str, mesh_name: str, n: int) -> dict:
    collection = state["tallies"]
    matches = [
        v for k, v in collection.items()
        if k.startswith("tally ") and isinstance(v, h5py.Group)
        and "name" in v and _label(v["name"][()]) == tally_name
    ]
    if len(matches) != 1:
        raise ValueError("missing or duplicate actual neutron mesh tally: " + tally_name)
    t = matches[0]
    if (_label(t["estimator"][()]) != "tracklength"
            or int(t["n_realizations"][()]) != n
            or int(t["n_filters"][()]) != 1):
        raise ValueError("wrong actual neutron tracklength estimator or realizations")
    ids = np.asarray(t["filters"][()], dtype=int).reshape(-1)
    if ids.size != 1:
        raise ValueError("native neutron tally has incorrect mesh filter count")
    filt = collection["filters"][f"filter {int(ids[0])}"]
    if _label(filt["type"][()]) != "mesh":
        raise ValueError("neutron transport tally is not mesh filtered")
    mid = np.asarray(filt["bins"][()]).reshape(-1)
    if mid.size != 1:
        raise ValueError("actual OpenMC mesh identifier missing")
    mesh = collection["meshes"][f"mesh {int(mid[0])}"]
    if (_label(mesh["name"][()]) != mesh_name
            or _label(mesh["type"][()]) != "regular"):
        raise ValueError("actual physical neutron mesh identity mismatch")
    dims = np.asarray(mesh["dimension"][()], dtype=int)
    lo = np.asarray(mesh["lower_left"][()], dtype=float)
    hi = np.asarray(mesh["upper_right"][()], dtype=float)
    if (dims.shape != (3,) or np.any(dims <= 0)
            or lo.shape != (3,) or hi.shape != (3,)
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(hi <= lo)
            or int(filt["n_bins"][()]) != int(np.prod(dims))):
        raise ValueError("invalid actual physical neutron mesh domain")
    scores = [_label(x) for x in t["score_bins"][()]]
    if len(scores) != len(set(scores)) or any(x not in scores for x in SCORES):
        raise ValueError("neutron flux or fission-production score absent")
    result = np.asarray(t["results"][()], dtype=float)
    if (result.shape != (int(np.prod(dims)), len(scores), 2)
            or not np.isfinite(result).all()):
        raise ValueError("invalid native OpenMC tracklength neutron moments")
    data = result[:, [scores.index(x) for x in SCORES], :]
    if np.any(data < -1e-10):
        raise ValueError("negative physical neutron flux or fission moments")
    return {
        "dims": dims.tolist(), "lower": lo.tolist(), "upper": hi.tolist(),
        "integrals": data[:, :, 0] / n,
    }


def reconcile_axial_three_d(axial: np.ndarray, volume: np.ndarray,
                            *, tolerance: float = 1e-8) -> dict:
    """A tracklength tally is an extensive reaction/path integral, not flux/cm3."""
    a = np.asarray(axial, dtype=float)
    v = np.asarray(volume, dtype=float)
    if (a.ndim != 2 or v.ndim != 2 or a.shape[1] != 3 or v.shape[1] != 3
            or a.shape[0] < 4 or v.shape[0] < a.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(v).all()
            or np.any(a < -1e-12) or np.any(v < -1e-12)
            or not 0 < tolerance <= 1e-4):
        raise ValueError("invalid neutron spatial reaction input")
    total_a = a.sum(axis=0, dtype=np.float64)
    total_v = v.sum(axis=0, dtype=np.float64)
    outcomes = {}
    for k, score in enumerate(SCORES):
        x, y = float(total_a[k]), float(total_v[k])
        if x <= 0 or y <= 0 or abs(x-y) / max(x,y) > tolerance:
            raise ValueError("independent physical neutron mesh integrals do not close: " + score)
        shape = a[:, k] / x
        outcomes[score] = {
            "axial_and_3d_integral_per_external_source": x,
            "relative_mesh_integral_error": abs(x-y) / max(x,y),
            "normalized_axial_distribution": shape.tolist(),
            "axial_mirror_total_variation": 0.5 * float(np.abs(shape-shape[::-1]).sum()),
            "axial_centroid_normalized_to_length": float(np.dot(
                shape, (np.arange(shape.size, dtype=float)+0.5)/shape.size-0.5)),
        }
    return outcomes


def read_actual_spatial_statepoint(path: Path, daughters: bool) -> dict:
    path = Path(path)
    settings = ET.parse(path.parent / "settings.xml").getroot()
    if (settings.tag != "settings"
            or settings.findtext("run_mode") != "fixed source"
            or settings.findtext("create_fission_neutrons") != ("true" if daughters else "false")
            or settings.findtext("inactive", default="0") != "0"):
        raise ValueError("wrong physical externally driven fission setting")
    with h5py.File(path, "r") as h:
        if (_label(h.attrs["filetype"]) != "statepoint"
                or _label(h["run_mode"][()]) != "fixed source"
                or int(h.attrs.get("photon_transport", -1)) != 0
                or int(h.attrs.get("tallies_present", -1)) != 1):
            raise ValueError("not an actual neutron-only OpenMC fixed-source statepoint")
        n = int(h["current_batch"][()])
        seed = int(h["seed"][()])
        particles = int(h["n_particles"][()])
        if (seed < 1 or n < 20 or particles < 100
                or int(h["n_realizations"][()]) != n
                or ("n_inactive" in h and int(h["n_inactive"][()]) != 0)):
            raise ValueError("invalid real neutron source batch/seed data")
        axial = _mesh(h, "nerva_axial_transport", "nerva_axial_mesh", n)
        three_d = _mesh(h, "nerva_3d_neutronics", "nerva_core_mesh", n)
    if (axial["dims"] != [1,1,three_d["dims"][2]]
            or any(v < 2 for v in three_d["dims"][:2])
            or not np.allclose(axial["lower"], three_d["lower"], rtol=0, atol=1e-9)
            or not np.allclose(axial["upper"], three_d["upper"], rtol=0, atol=1e-9)):
        raise ValueError("native neutron axial and 3D domains differ")
    geometry = path.parent / "geometry.xml"
    root = ET.parse(geometry).getroot()
    boundaries = [x for x in root.findall("surface")
                  if x.attrib.get("boundary") == "vacuum"]
    endcaps = sorted(float(x.attrib["coeffs"]) for x in boundaries
                     if x.attrib.get("type") == "z-plane")
    radial = [x for x in boundaries if x.attrib.get("type") == "z-cylinder"]
    if root.tag != "geometry" or len(boundaries) != 3 or len(endcaps) != 2 or len(radial) != 1:
        raise ValueError("invalid actual whole-core vacuum geometry")
    cx, cy, radius = (float(x) for x in radial[0].attrib["coeffs"].split())
    lo, hi = axial["lower"], axial["upper"]
    if (not all(map(math.isfinite, [cx,cy,radius,*endcaps])) or radius <= 0
            or not np.allclose(endcaps,[lo[2],hi[2]],rtol=0,atol=1e-7)
            or lo[0] > cx-radius+1e-7 or hi[0] < cx+radius-1e-7
            or lo[1] > cy-radius+1e-7 or hi[1] < cy+radius-1e-7):
        raise ValueError("real neutron tally meshes clip exterior reactor vacuum surfaces")
    scored = reconcile_axial_three_d(axial["integrals"], three_d["integrals"])
    return {
        "seed": seed, "batches": n, "particles_per_batch": particles,
        "daughters_transported": daughters, "scores": scored,
        "statepoint_sha256": _sha(path),
        "geometry_sha256": _sha(geometry),
        "materials_sha256": _sha(path.parent/"materials.xml"),
        "tallies_sha256": _sha(path.parent/"tallies.xml"),
        "physical_bounds_cm": {"lower":lo,"upper":hi},
        "axial_dimensions":axial["dims"], "three_d_dimensions":three_d["dims"],
    }


def audit_four_seed_spatial_response(root: Path, *, batches: int = 30,
        seeds: tuple[int,...] = (13007,23011,33013,43019)) -> dict:
    root = Path(root)
    if (len(seeds) != 4 or len(set(seeds)) != 4
            or any(type(x) is not int or x <= 0 for x in seeds)):
        raise ValueError("four independent actual OpenMC seed pairs required")
    paired = []
    for seed in seeds:
        off = read_actual_spatial_statepoint(
            root/"suppressed_fission_control"/f"seed_{seed}"/f"statepoint.{batches}.h5",
            False)
        on = read_actual_spatial_statepoint(
            root/"fission_daughters_transported"/f"seed_{seed}"/f"statepoint.{batches}.h5",
            True)
        identity = ("geometry_sha256","materials_sha256","tallies_sha256",
                    "physical_bounds_cm","axial_dimensions","three_d_dimensions")
        if (off["seed"] != on["seed"] or on["seed"] != seed
                or off["batches"] != on["batches"] or on["batches"] != batches
                or off["particles_per_batch"] != on["particles_per_batch"]
                or off["statepoint_sha256"] == on["statepoint_sha256"]
                or any(off[k] != on[k] for k in identity)):
            raise ValueError("actual physical neutron tally/model inputs differ between modes")
        paired.append({"seed":seed,"fission_off":off,"fission_on":on})
    metrics = {}
    for score in SCORES:
        before = np.array([x["fission_off"]["scores"][score]
            ["axial_and_3d_integral_per_external_source"] for x in paired])
        after = np.array([x["fission_on"]["scores"][score]
            ["axial_and_3d_integral_per_external_source"] for x in paired])
        difference = after-before
        off_profile = np.array([x["fission_off"]["scores"][score]
            ["normalized_axial_distribution"] for x in paired])
        on_profile = np.array([x["fission_on"]["scores"][score]
            ["normalized_axial_distribution"] for x in paired])
        dp = on_profile-off_profile
        metrics[score] = {
            "control_mean_per_external_source": float(before.mean()),
            "daughter_mean_per_external_source": float(after.mean()),
            "four_pair_response_mean": float(difference.mean()),
            "paired_seed_response_standard_error": float(difference.std(ddof=1)/2.),
            "relative_response": float(difference.mean()/before.mean()),
            "per_seed_response": difference.tolist(),
            "axial_profile_change_mean": dp.mean(axis=0).tolist(),
            "axial_profile_change_between_pair_sem": (dp.std(axis=0,ddof=1)/2.).tolist(),
        }
    return {
        "audit": "GENUINE_OPENMC_FIXED_SOURCE_FISSION_DAUGHTER_SPATIAL_NEUTRON_RESPONSE",
        "independent_seed_pairs": len(paired),
        "real_hdf5_statepoints": 2*len(paired),
        "real_whole_core_axial_and_3d_flux_fission_nufission_conserved": True,
        "physically_identical_geometry_materials_tallies": True,
        "paired_response": metrics, "seeds": paired,
        "criticality_or_eigenvalue_source_stationarity_validated": False,
        "hot_nuclear_energy_deposition_or_historical_nerva_validated": False,
        "formal_confidence_coverage_established": False,
        "interpretation": (
            "Source-driven, noncritical cold conceptual reactor physics only. "
            "All reaction rates are integrated whole-core tracklength scores "
            "per external neutron, NOT flux per unit volume. Axial profiles "
            "are normalized independently; four paired random seeds do not "
            "establish rigorous Monte Carlo uncertainty coverage."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=audit_four_seed_spatial_response(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Genuine 8-statepoint fixed-source axial/3D neutron tracklength conservation: PASS")
    print("Paired flux response:",result["paired_response"]["flux"]["four_pair_response_mean"])
    print("NERVA eigenvalue, source stationarity, and local heating: NOT QUALIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
