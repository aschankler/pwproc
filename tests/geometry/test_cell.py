"""Tests for the :py:mod:`pwproc.geometry.cell` module."""

from typing import Callable, ContextManager, Optional

import numpy as np
import pytest

import pwproc.geometry
from pwproc.geometry import Basis
from pwproc.geometry.cell import CellParam
from tests.geometry.util import check_basis, no_exception

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
