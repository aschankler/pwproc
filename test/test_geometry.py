import pytest
from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
import pwproc.geometry
from pwproc.geometry.cell import Basis, Species, Tau


Geometry = Tuple[Basis, Species, Tau]

GEOMETRY_DATA = {
    "BN_crystal": {
        "basis": 3.57 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        "species": ("B", "N"),
        "tau": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    },
}


def check_geometry(geom: Geometry, geom_ref: Geometry) -> None:
    basis, species, tau = geom
    basis_ref, species_ref, tau_ref = geom_ref
    assert (basis == basis_ref).all()
    assert species == species_ref
    assert (tau == tau_ref).all()


@pytest.fixture
def geometry_directory() -> Path:
    return Path(__file__).parent.joinpath("data", "geometry")


@pytest.fixture
def geometry_ref(request) -> Geometry:
    data = GEOMETRY_DATA[request.param]
    basis = pwproc.geometry.cell.Basis(data["basis"].copy())
    species = pwproc.geometry.cell.Species(data["species"])
    tau = pwproc.geometry.cell.Tau(data["tau"].copy())
    return basis, species, tau


# Todo: test format utils
# Todo: test cell
# Todo: test xsf

# ------------------
# Test POSCAR module
# ------------------


class TestPoscar:
    geom_read_data = [
        {"id": "BN_crys", "comment": "Cubic BN", "scale": 3.57, "geom": "BN_crystal"},
        {"id": "BN_cart", "comment": "Cubic BN", "scale": 3.57, "geom": "BN_crystal"},
        {
            "id": "BN_crys_sd",
            "comment": "Cubic BN",
            "scale": 3.57,
            "geom": "BN_crystal",
        },
        {
            "id": "BN_cart_sd",
            "comment": "Cubic BN",
            "scale": 3.57,
            "geom": "BN_crystal",
        },
    ]

    @pytest.fixture
    def file(self, request, geometry_directory) -> Iterable[str]:
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
        assert scale == scale_test

    @pytest.mark.parametrize(
        ["geometry_ref", "file"],
        [pytest.param(dat["geom"], dat["id"], id=dat["id"]) for dat in geom_read_data],
        indirect=["geometry_ref", "file"],
    )
    def test_read(self, geometry_ref: Geometry, file: Iterable[str]) -> None:
        geom_test = pwproc.geometry.poscar.read_poscar(file, out_type="crystal")
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.parametrize("coord", ["direct", "cartesian"])
    @pytest.mark.parametrize("scale", [1.0, 3.57])
    @pytest.mark.parametrize(
        ["geometry_ref", "comment"],
        [(k, k) for k in GEOMETRY_DATA],
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

    def test_read_dynamics(self):
        ...

    def test_write_dynamics(self):
        ...


# writer (direct)
# writer (cart)
# writer (scale)
# reader (selective_dynamics)
# writer (selective_dynamics)
# writer/reader (something with more atoms)
# Error tests


# Test espresso
# Test data module
