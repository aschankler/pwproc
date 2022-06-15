from pathlib import Path

from typing import Iterable

import pytest

import pwproc.geometry
from pwproc.geometry.cell import MovableFlags
from tests.geometry.util import (
    Geometry,
    Trajectory,
    canonical_order,
    check_geometry,
)

# pylint: disable=import-outside-toplevel,redefined-outer-name,no-self-use


@pytest.fixture
def geometry_directory() -> Path:
    return Path(__file__).parent.joinpath("data", "poscar")


@pytest.fixture
def geometry_ref(request, monkeypatch) -> Geometry:
    data_dir = Path(__file__).parent.joinpath("data")
    monkeypatch.syspath_prepend(data_dir)
    geometry_test_data = pytest.importorskip("geometry_test_data")

    data = geometry_test_data.GEOMETRY_DATA[request.param]
    basis = pwproc.geometry.cell.Basis(data["basis"].copy())
    species = pwproc.geometry.cell.Species(data["species"])
    tau = pwproc.geometry.cell.Tau(data["tau"].copy())
    return basis, species, tau


# Todo: tests cell
# ------------------
# Test `cell` module
# ------------------


# Todo: tests xsf
# ------------------
# Test XSF module
# ------------------


class TestXSF:
    @pytest.fixture
    def file(self, request, geometry_directory: Path) -> Iterable[str]:
        geom_file = geometry_directory.joinpath(request.param + ".xsf")
        with open(geom_file) as f:
            return f.readlines()

    @pytest.mark.skip
    def test_read_xsf(self, geometry_ref: Geometry, file: Iterable[str]) -> None:
        geom_test = pwproc.geometry.xsf.read_xsf(file, axsf_allowed=False)
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.skip
    def test_read_axsf(self, trajectory_ref: Trajectory, file: Iterable[str]) -> None:
        ...

    @pytest.mark.skip
    def test_gen_xsf(self):
        ...

    @pytest.mark.skip
    def test_gen_axsf(self):
        ...


# ------------------
# Test POSCAR module
# ------------------


class TestPoscar:
    geom_read_data = [
        {"id": "BN_crys", "comment": "Cubic BN", "scale": 3.57, "geom": "BN"},
        {"id": "BN_cart", "comment": "Cubic BN", "scale": 3.57, "geom": "BN"},
        {"id": "BN_crys_sd", "comment": "Cubic BN", "scale": 3.57, "geom": "BN"},
        {"id": "BN_cart_sd", "comment": "Cubic BN", "scale": 3.57, "geom": "BN"},
        {"id": "BTO_cart", "comment": "BaTiO3", "scale": 1.0, "geom": "BTO"},
        {"id": "BTO_cart_scale", "comment": "BaTiO3", "scale": 4.004456, "geom": "BTO"},
        {"id": "BTO_crys", "comment": "BaTiO3", "scale": 1.0, "geom": "BTO"},
        {"id": "BTO_crys_scale", "comment": "BaTiO3", "scale": 4.004456, "geom": "BTO"},
        {"id": "LiNbO3_cart", "comment": "Li2Nb2O6", "scale": 1.0, "geom": "LiNbO3"},
        {
            "id": "LiNbO3_cart_scale",
            "comment": "Li2Nb2O6",
            "scale": 5.592399,
            "geom": "LiNbO3",
        },
        {
            "id": "LiNbO3_direct",
            "comment": "Li2Nb2O6",
            "scale": 5.592399,
            "geom": "LiNbO3",
        },
    ]

    @pytest.fixture
    def file(self, request, geometry_directory: Path) -> Iterable[str]:
        geom_file = geometry_directory.joinpath("POSCAR_" + request.param)
        with open(geom_file) as f:
            return f.readlines()

    @pytest.mark.parametrize(
        ["comment", "file"],
        [
            pytest.param(dat["comment"], dat["id"], id=dat["id"])
            for dat in geom_read_data
        ],
        indirect=["file"],
    )
    def test_comment(self, comment: str, file: Iterable[str]) -> None:
        comment_test = pwproc.geometry.poscar.read_poscar_comment(file)
        assert comment == comment_test

    @pytest.mark.parametrize(
        ["scale", "file"],
        [pytest.param(dat["scale"], dat["id"], id=dat["id"]) for dat in geom_read_data],
        indirect=["file"],
    )
    def test_scale(self, scale: float, file: Iterable[str]) -> None:
        scale_test = pwproc.geometry.poscar.read_poscar_scale(file)
        assert pytest.approx(scale) == scale_test

    @pytest.mark.parametrize("coord", ["crystal", "angstrom", "bohr"])
    @pytest.mark.parametrize(
        ["geometry_ref", "file"],
        [pytest.param(dat["geom"], dat["id"], id=dat["id"]) for dat in geom_read_data],
        indirect=["geometry_ref", "file"],
    )
    def test_read(
        self, geometry_ref: Geometry, file: Iterable[str], coord: str
    ) -> None:
        basis, species, tau = pwproc.geometry.poscar.read_poscar(file, out_type=coord)
        tau = pwproc.geometry.cell.convert_positions(tau, basis, coord, "crystal")
        geom_test = (basis, species, tau)
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.parametrize("coord", ["direct", "cartesian"])
    @pytest.mark.parametrize("scale", [1.0, 3.57])
    @pytest.mark.parametrize(
        ["geometry_ref", "comment"],
        [("BN", "BN"), ("BN_rev", "BN"), ("BTO", "BaTiO3"), ("LiNbO3", "LiNbO3")],
        indirect=["geometry_ref"],
    )
    def test_write(
        self, geometry_ref: Geometry, comment: str, scale: float, coord: str
    ) -> None:
        """Test writer assuming reader is correct."""
        out_lines = pwproc.geometry.poscar.gen_poscar(
            *geometry_ref,
            comment=comment,
            scale=scale,
            in_type="crystal",
            coordinate_type=coord,
        )
        out_lines = "".join(out_lines).split("\n")
        assert comment == pwproc.geometry.poscar.read_poscar_comment(out_lines)
        assert scale == pwproc.geometry.poscar.read_poscar_scale(out_lines)
        geom_test = pwproc.geometry.poscar.read_poscar(out_lines, out_type="crystal")
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.parametrize(
        "file,dynamics",
        [
            pytest.param("BN_cart", (None, None), id="BN_cart"),
            pytest.param("BN_crys", (None, None), id="BN_crys"),
            pytest.param(
                "BN_cart_sd",
                ((True, True, False), (False, False, False)),
                id="BN_cart_sd",
            ),
            pytest.param("BN_crys_sd", ((True, False, True), None), id="BN_crys_sd"),
        ],
        indirect=["file"],
    )
    def test_read_dynamics(self, file: Iterable[str], dynamics: MovableFlags) -> None:
        result = pwproc.geometry.poscar.read_poscar_selective_dynamics(file)
        assert len(result) == len(dynamics)
        for test, ref in zip(result, dynamics):
            assert test == ref

    @pytest.mark.parametrize(
        "geometry_ref,dynamics",
        [
            pytest.param("BN_crystal", None, marks=pytest.mark.xfail),
            ("BN", (None, None)),
            ("BN", (None, (True, True, True))),
            ("BN", (None, (False, True, True))),
            ("BN", ((False, False, False), (True, True, True))),
            ("BN_rev", (None, None)),
            ("BN_rev", (None, (True, True, True))),
            ("BN_rev", (None, (False, True, True))),
            ("BN_rev", ((False, False, False), (True, True, True))),
        ],
        indirect=["geometry_ref"],
    )
    def test_write_dynamics(
        self, geometry_ref: Geometry, dynamics: MovableFlags
    ) -> None:
        out_lines = pwproc.geometry.poscar.gen_poscar(
            *geometry_ref, selective_dynamics=dynamics
        )
        out_lines = "".join(out_lines).split("\n")
        geom_out = pwproc.geometry.poscar.read_poscar(out_lines)
        dynamics_out = pwproc.geometry.poscar.read_poscar_selective_dynamics(out_lines)
        assert len(dynamics) == len(dynamics_out)
        # Define a canonical order to sort the dynamics flags
        idx_ref = canonical_order(geometry_ref[2])
        idx = canonical_order(geom_out[2])
        assert [dynamics[i] for i in idx_ref] == [dynamics_out[i] for i in idx]


# Test with species mixed up so that they must be re-ordered
# Error tests (incl. wrong shape input basis/position to the writer, incorrect inputs
# to other parameters, malformed files). Also xsf files with forces.


# Test espresso
# Test data module
