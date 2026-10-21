"""Regress physical MT=901 preflight wiring, without running reactor transport.

A real OpenMC/NNDC evaluation in the nuclear-data workflow separately proves
that neutron-ONLY thermal normalization is correctly blocked for test data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import run_full_openmc_analysis as full
from . import run_openmc_transport as transport


class _CapturedCommand(RuntimeError):
    def __init__(self, arguments):
        super().__init__("command captured before any transport executes")
        self.arguments = arguments


def _capture_first_command(arguments):
    raise _CapturedCommand(arguments)


def _run_full_once(root: Path, *, photon: bool) -> list[str]:
    argv = [
        "run_full_openmc_analysis", "--cross-sections", "unopened-data.xml",
        "--assembly", "core", "--rings", "3",
        "--power-mw", "1",
        "--fuel-mass-flow-kg-s", "1",
        "--tie-mass-flow-kg-s", "1",
        "--inlet-temperature-k", "300",
        "--inlet-pressure-mpa", "1",
        "--output-root", str(root),
    ]
    if photon:
        argv.append("--photon-transport")
    with patch.object(sys, "argv", argv):
        with patch.object(full, "_run", _capture_first_command):
            try:
                full.main()
            except _CapturedCommand as exc:
                return exc.arguments
    raise AssertionError("full analysis did not prepare actual transport command")


def main() -> int:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        neutron = _run_full_once(root / "neutron", photon=False)
        assert "ntr.nerva.run_openmc_transport" in neutron
        assert "--require-local-heating-data" in neutron
        assert "--photon-transport" not in neutron
        coupled = _run_full_once(root / "coupled", photon=True)
        assert "ntr.nerva.run_openmc_transport" in coupled
        assert "--require-local-heating-data" not in coupled
        assert "--photon-transport" in coupled

    # The transport API must never interpret neutron-only MT901 KERMA
    # coverage as validation of explicitly transported photon heating.
    args = [
        "run_openmc_transport", "--assembly", "core",
        "--rings", "3", "--photon-transport",
        "--require-local-heating-data",
    ]
    with patch.object(sys, "argv", args):
        try:
            transport.main()
        except ValueError as exc:
            assert "neutron-only local heating" in str(exc), str(exc)
        else:
            raise AssertionError("coupled transport incorrectly accepted neutron-only MT901 gate")

    print("NERVA full-run neutron-only local KERMA physical preflight wiring: PASS")
    print("  mock command capture only; actual evaluated data checked in real CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
