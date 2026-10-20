"""Numerical tests for engine-level mass, thrust and Isp conservation."""
import math
from .engine_stream_balance import (
    G0, engine_stream_balance, infer_bleed_from_engine_reference
)
from .historical_presets import XE_PRIME


def main() -> int:
    pure = engine_stream_balance(25.0, 8100.0)
    assert pure.total_flow_kg_s == pure.main_flow_kg_s
    assert math.isclose(pure.engine_isp_s, pure.nozzle_isp_s)
    mixed = engine_stream_balance(25.0, 8100.0, 3.5, 1200.0)
    assert math.isclose(mixed.total_flow_kg_s, 28.5)
    assert math.isclose(mixed.total_thrust_n, 206700.0)
    assert math.isclose(
        mixed.engine_isp_s, 206700.0/(28.5*G0), rel_tol=1e-13)
    assert mixed.engine_isp_s < mixed.nozzle_isp_s
    inv = infer_bleed_from_engine_reference(
        mixed.total_thrust_n, mixed.engine_isp_s, mixed.main_flow_kg_s,
        mixed.nozzle_effective_velocity_m_s
    )
    assert math.isclose(inv.inferred_bleed_flow_kg_s, 3.5, rel_tol=1e-12)
    assert math.isclose(inv.required_bleed_effective_velocity_m_s, 1200.0,
                        rel_tol=1e-11)
    # XE-Prime inverse demonstrates that net engine Isp and listed main
    # channel H2 flow must not be confused. Main flow is from NASA preset.
    target = infer_bleed_from_engine_reference(
        XE_PRIME.thrust_n, XE_PRIME.isp_s,
        XE_PRIME.hydrogen_mass_flow_kg_s, 7600.0
    )
    assert target.inferred_bleed_flow_kg_s > 0
    assert target.inferred_total_flow_kg_s > XE_PRIME.hydrogen_mass_flow_kg_s
    for kwargs in (
        dict(main_flow_kg_s=0.0, nozzle_effective_velocity_m_s=5000),
        dict(main_flow_kg_s=1.0, nozzle_effective_velocity_m_s=5000,
             bleed_flow_kg_s=-1.0),
        dict(main_flow_kg_s=1.0, nozzle_effective_velocity_m_s=5000,
             bleed_effective_velocity_m_s=100),
    ):
        try:
            engine_stream_balance(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid engine-stream input accepted")
    print("NERVA multi-stream mass/thrust/Isp accounting: PASS")
    print(f"XE-Prime mass-flow difference (inferred, not measured): "
          f"{target.inferred_bleed_flow_kg_s:.3f} kg/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
