from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pytest

import pwproc.geometry
from pwproc.geometry.cell import Basis, CellParam, MovableFlags, Species, Tau

# pylint: disable=import-outside-toplevel,redefined-outer-name,no-self-use


@contextmanager
def no_exception() -> Generator[None, None, None]:
    yield


Geometry = Tuple[Basis, Species, Tau]

GEOMETRY_DATA = {
    "BN_crystal": {
        "basis": 3.57 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        "species": ("B", "N"),
        "tau": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    },
    "BN_crystal_rev": {
        "basis": 3.57 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        "species": ("N", "B"),
        "tau": np.array([[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]]),
    },
    "BTO": {
        "basis": 4.0044569
        * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.04899014]]),
        "species": ("Ba", "O", "O", "O", "Ti"),
        "tau": np.array(
            [
                [0.5, 0.5, 0.004136000],
                [0.0, 0.5, 0.479348987],
                [0.5, 0.0, 0.479348987],
                [0.0, 0.0, 0.958854020],
                [0.0, 0.0, 0.524312973],
            ]
        ),
    },
    "LiNbO3": {
        "basis": 5.5923991
        * np.array(
            [
                [1.000000000, 0.000000000, 0.000000000],
                [0.563873923, 0.825860883, 0.000000000],
                [0.563873923, 0.297774270, 0.770309472],
            ]
        ),
        "species": ("Li", "Li", "Nb", "Nb", "O", "O", "O", "O", "O", "O"),
        "tau": np.array(
            [
                [0.717167974, 0.717167974, 0.717167974],
                [0.217168003, 0.217168003, 0.217168003],
                [0.999230981, 0.999230981, 0.999230981],
                [0.499231011, 0.499231011, 0.499231011],
                [0.639684021, 0.890367985, 0.279949009],
                [0.890367985, 0.279949009, 0.639684021],
                [0.279949009, 0.639684021, 0.890367985],
                [0.390368015, 0.139684007, 0.779949009],
                [0.139684007, 0.779949009, 0.390368015],
                [0.779949009, 0.390368015, 0.139684007],
            ]
        ),
    },
}


def check_basis(basis: Basis, basis_ref: Basis, **kw: Any) -> None:
    assert np.isclose(basis, basis_ref, **kw).all()


def canonical_order(tau: Tau) -> Sequence[int]:
    """Return an argsort based on a canonical order for the positions."""
    from functools import total_ordering

    @total_ordering
    class _VectorApproxOrder:
        def __init__(self, obj: np.ndarray) -> None:
            if obj.shape != (3,):
                raise ValueError
            self.obj = obj

        def __eq__(self, other: object) -> bool:
            if isinstance(other, np.ndarray):
                return np.isclose(self.obj, other).all()
            if isinstance(other, _VectorApproxOrder):
                return np.isclose(self.obj, other.obj).all()
            return NotImplemented

        @staticmethod
        def _vec_approx_lt(x: np.ndarray, y: np.ndarray) -> bool:
            for xi, yi in zip(x, y):
                if np.isclose(xi, yi):
                    continue
                return xi < yi
            # All components equal
            return False

        def __lt__(self, other: object) -> bool:
            if isinstance(other, np.ndarray):
                if other.shape != (3,):
                    return NotImplemented
                return self._vec_approx_lt(self.obj, other)
            if isinstance(other, _VectorApproxOrder):
                return self._vec_approx_lt(self.obj, other.obj)
            return NotImplemented

    decorated = [(_VectorApproxOrder(x), i) for i, x in enumerate(tau)]
    decorated.sort()
    return [i for _, i in decorated]


def check_geometry(geom: Geometry, geom_ref: Geometry) -> None:
    basis, species, tau = geom
    basis_ref, species_ref, tau_ref = geom_ref
    check_basis(basis, basis_ref)

    # Put both in canonical order
    def _order_geom(_species: Species, _tau: Tau) -> Tuple[Species, Tau]:
        idx = canonical_order(_tau)
        return tuple(_species[i] for i in idx), _tau[idx]

    species, tau = _order_geom(species, tau)
    species_ref, tau_ref = _order_geom(species_ref, tau_ref)

    # Now geometry should match exactly
    assert species == species_ref
    assert np.isclose(tau, tau_ref).all()


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


# Todo: test cell
# ------------------
# Test `cell` module
# ------------------

"""
def convert_positions(
    positions: Tau,
    basis: Basis,
    in_type: str,
    out_type: str,
    *,
    basis_units: str = "angstrom",
    alat: Optional[float] = None,
) -> Tau:

def to_crystal(
    positions: Tau, basis: Basis, in_type: str, alat: Optional[float]
) -> Tau:

def from_crystal(
    positions: Tau, basis: Basis, out_type: str, alat: Optional[float]
) -> Tau:


def wrap_positions(
    positions: Tau,
    basis: Basis,
    *,
    pos_type: str = "angstrom",
    basis_units: str = "angstrom",
    alat: Optional[float] = None,
) -> Tau:

def minimum_distance(
    pos1: Tau,
    pos2: Tau,
    basis: Basis,
    *,
    pos_type: str = "angstrom",
    basis_units: str = "angstrom",
    alat: Optional[float] = None,
) -> Sequence[float]:
"""


@pytest.mark.parametrize(
    "basis,in_type,out_type,alat,basis_ref,expect_raise",
    [
        (np.eye(3), "bohr", "bohr", None, np.eye(3), no_exception()),
        (np.eye(3), "angstrom", "angstrom", None, np.eye(3), no_exception()),
        (np.eye(3), "alat", "alat", 1.0, np.eye(3), no_exception()),
        (np.eye(3), "alat", "alat", None, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "bohr", "alat", 1.0, np.eye(3), no_exception()),
        (np.eye(3), "bohr", "alat", 2.0, 0.5 * np.eye(3), no_exception()),
        (np.eye(3), "alat", "bohr", 2.0, 2.0 * np.eye(3), no_exception()),
        (np.eye(3), "angstrom", "bohr", None, 1.88972 * np.eye(3), no_exception()),
        (np.eye(3), "bohr", "angstrom", None, 0.52917892 * np.eye(3), no_exception()),
        (np.eye(3), "alat", "angstrom", 2.0, 1.058357 * np.eye(3), no_exception()),
        (np.eye(3), "angstrom", "alat", 2.0, 0.94486 * np.eye(3), no_exception()),
        (np.eye(3), "alat", "bohr", None, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "bohr", "alat", None, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "alat= 2.0", "bohr", None, 2.0 * np.eye(3), no_exception()),
        (np.eye(3), "alat = 2.0", "bohr", None, 2.0 * np.eye(3), no_exception()),
        (np.eye(3), "bohr", "alat= 2.0", None, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "alat= 2.0", "bohr", 3.0, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "bohr", "crystal", None, np.eye(3), pytest.raises(ValueError)),
        (np.eye(3), "crystal", "alat", None, np.eye(3), pytest.raises(ValueError)),
    ],
)
def test_convert_basis(
    basis: Basis,
    in_type: str,
    out_type: str,
    alat: Optional[float],
    basis_ref: Basis,
    expect_raise: ContextManager,
) -> None:
    with expect_raise:
        basis_test = pwproc.geometry.cell.convert_basis(
            basis, in_type, out_type, alat=alat
        )
        check_basis(basis_test, basis_ref)


@pytest.mark.parametrize(
    "units,alat",
    [("bohr", None), ("angstrom", None), ("alat", 1.0), ("alat= 1.0", None)],
)
@pytest.mark.parametrize(
    "test_method",
    (
        pwproc.geometry.cell.cell_alat,
        pwproc.geometry.cell.cell_volume,
        pwproc.geometry.cell.cell_parameters,
        pwproc.geometry.cell.normalize_alat,
    ),
)
def test_cell_unit_passthrough(
    test_method: Callable, units: str, alat: float, monkeypatch
) -> None:
    """Ensure that args are passed through to `convert_basis`."""
    # Patch convert function
    class _MockConvertBasis:
        def __init__(self) -> None:
            self.called = False
            self.n_calls = 0
            self.in_type: Optional[str] = None
            self.out_type: Optional[str] = None
            self.alat: Optional[float] = None

        def __call__(
            self,
            basis: Basis,
            in_type: str,
            out_type: str,
            *,
            alat: Optional[float] = None,
        ) -> Basis:
            self.n_calls += 1
            if not self.called:
                self.called = True
                self.in_type = in_type
                self.out_type = out_type
                self.alat = alat
            return basis

    mocked_fn = _MockConvertBasis()
    monkeypatch.setattr(pwproc.geometry.cell, "convert_basis", mocked_fn)

    # Call method
    nul_basis = np.eye(3)
    test_method(nul_basis, units=units, alat=alat)
    assert mocked_fn.called
    assert mocked_fn.in_type == units
    assert mocked_fn.alat == alat


@pytest.mark.parametrize(
    "basis,units,alat_ref",
    [
        (np.eye(3), "bohr", 1.0),
        (np.eye(3), "angstrom", pytest.approx(1.88972, 1e-3)),
        (np.eye(3), "alat= 1.0", 1.0),
        (np.eye(3), "alat= 2.0", 2.0),
        (np.diag((1.0, 2.0, 3.0)), "bohr", 1.0),
        (np.diag((4.0, 2.0, 3.0)), "bohr", 4.0),
        (
            np.array([[1.0, 1.0, 0.0], [0, 1, 0], [0, 0, 1]]),
            "bohr",
            pytest.approx(1.4142, 1e-3),
        ),
    ],
)
def test_cell_alat(basis: Basis, units: str, alat_ref: float) -> None:
    """This relies on the `convert_basis` tests to exercise the alat argument."""
    alat_test = pwproc.geometry.cell.cell_alat(basis, units)
    assert pytest.approx(alat_test) == alat_ref


@pytest.mark.parametrize(
    "basis,volume",
    [
        (np.eye(3), 1.0),
        (2 * np.eye(3), 8.0),
        (np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]), 0.25),
        (np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]), 0.5),
    ],
)
def test_cell_volume(basis: Basis, volume: float) -> None:
    """Relies on `convert_basis` tests to exercise unit conversion code paths."""
    volume_test = pwproc.geometry.cell.cell_volume(basis)
    assert pytest.approx(volume_test) == volume


@pytest.mark.parametrize(
    "basis,ref_params",
    [
        (np.eye(3), (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)),
        (3.0 * np.eye(3), (3.0, 3.0, 3.0, 90.0, 90.0, 90.0)),
        (np.diag([1.0, 3.0, 2.0]), (1.0, 3.0, 2.0, 90.0, 90.0, 90.0)),
        (
            np.array([[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0, 0, 1]]),
            (pytest.approx(0.707107), pytest.approx(0.707107), 1.0, 90.0, 90.0, 90.0),
        ),
        (
            np.array([[-0.5, 0.5, 0.0], [0, 1, 0], [0, 0, 1]]),
            (pytest.approx(0.707107), 1, 1, 90, 90, 45.0),
        ),
        (
            np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1]]),
            (pytest.approx(0.707107), 1, 1, 90, 45.0, 90),
        ),
    ],
)
def test_cell_parameters(basis: Basis, ref_params: CellParam) -> None:
    """Relies on `convert_basis` tests to exercise unit conversion code paths."""
    test_params = pwproc.geometry.cell.cell_parameters(basis)
    assert pytest.approx(test_params) == ref_params


@pytest.mark.parametrize(
    "basis,units,ref_alat,ref_basis",
    [
        (np.eye(3), "alat= 1", 1.0, np.eye(3)),
        (np.eye(3), "alat= 2", 2.0, np.eye(3)),
        (2.0 * np.eye(3), "alat= 1", 2.0, np.eye(3)),
        (np.diag([1.0, 2.0, 3.0]), "alat= 1", 1.0, np.diag([1.0, 2.0, 3.0])),
        (np.diag([2.0, 2.0, 4.0]), "alat= 1", 2.0, np.diag([1.0, 1.0, 2.0])),
        (np.diag([2.0, 2.0, 4.0]), "alat= 1.5", 3.0, np.diag([1.0, 1.0, 2.0])),
        (
            np.array([[3.0, 4.0, 0.0], [0, 1, 0], [0, 0, 5]]),
            "alat= 2",
            10.0,
            np.array([[0.6, 0.8, 0.0], [0, 0.2, 0], [0, 0, 1]]),
        ),
    ],
)
def test_normalize_alat(
    basis: Basis, units: str, ref_alat: float, ref_basis: Basis
) -> None:
    new_alat, new_basis = pwproc.geometry.cell.normalize_alat(basis, units)
    assert pytest.approx(new_alat) == ref_alat
    check_basis(new_basis, ref_basis)


"""
def format_basis(
    basis: Basis,
    *,
    min_space: int = 3,
    left_pad: Optional[int] = None,
    precision: int = LATTICE_PRECISION,
) -> List[str]:

def format_positions(
    species: Species,
    tau: Tau,
    *,
    min_space: int = 3,
    left_pad: int = 0,
    precision: int = POSITION_PRECISION,
) -> List[str]:
"""


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
        {"id": "BTO_cart", "comment": "BaTiO3", "scale": 1.0, "geom": "BTO"},
        {
            "id": "BTO_cart_scale",
            "comment": "BaTiO3",
            "scale": pytest.approx(4.004456),
            "geom": "BTO",
        },
        {
            "id": "BTO_direct",
            "comment": "BaTiO3",
            "scale": pytest.approx(4.004456),
            "geom": "BTO",
        },
        {"id": "LiNbO3_cart", "comment": "Li2Nb2O6", "scale": 1.0, "geom": "LiNbO3"},
        {
            "id": "LiNbO3_cart_scale",
            "comment": "Li2Nb2O6",
            "scale": pytest.approx(5.592399),
            "geom": "LiNbO3",
        },
        {
            "id": "LiNbO3_direct",
            "comment": "Li2Nb2O6",
            "scale": pytest.approx(5.592399),
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
        assert scale == scale_test

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
            ("BN_crystal", (None, None)),
            ("BN_crystal", (None, (True, True, True))),
            ("BN_crystal", (None, (False, True, True))),
            ("BN_crystal", ((False, False, False), (True, True, True))),
            ("BN_crystal_rev", (None, None)),
            ("BN_crystal_rev", (None, (True, True, True))),
            ("BN_crystal_rev", (None, (False, True, True))),
            ("BN_crystal_rev", ((False, False, False), (True, True, True))),
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


# writer/reader (something with more atoms) BTO (tetrahedral), LiNbO3,
# Error tests


# Test espresso
# Test data module
