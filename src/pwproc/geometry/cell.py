"""Property calculators and conversion utilities for unit cells."""

import re
from typing import List, NewType, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.constants  # type: ignore[import]

from pwproc.geometry.format_util import (
    LATTICE_PRECISION,
    POSITION_PRECISION,
    as_fixed_precision,
    columns,
)

# Vector of atomic species
Species = NewType("Species", Sequence[str])
# Crystal basis [3x3]
Basis = NewType("Basis", np.ndarray)
# Position matrix [n_atoms x 3]
Tau = NewType("Tau", np.ndarray)
# Represent whether atoms may be moved during relaxation/dynamics
MovableFlags = Optional[Sequence[Union[None, Tuple[bool, bool, bool]]]]
# Unit cell dimensions (a, b, c) in angstrom and angles (alpha, beta, gamma)
CellParam = NewType("CellParam", Tuple[float, float, float, float, float, float])


# Allowed output coordinate types
_BASIS_COORDINATES = ("angstrom", "bohr", "alat")
_POSITION_COORDINATES = ("angstrom", "bohr", "crystal", "alat")


def convert_positions(
    positions: Tau,
    basis: Basis,
    in_type: str,
    out_type: str,
    *,
    basis_units: str = "angstrom",
    alat: Optional[float] = None,
) -> Tau:
    """Convert the coordinate type of atomic positions.

    :param positions: Vector of positions shape (n_atoms, 3)
    :param basis: Basis of the unit cell
    :param in_type: Coordinate type of `positions`
    :param out_type: Coordinate type to return
    :param basis_units: Coordinate type of `basis`
    :param alat: lattice parameter in bohr
    :returns: Atomic positions in converted units
    """
    if out_type not in _POSITION_COORDINATES:
        raise ValueError(f"Invalid position coordinate type {out_type}.")
    # Short-circuit conversion
    if in_type == out_type:
        return positions
    # Normalize the basis
    basis = convert_basis(basis, basis_units, "angstrom")
    # Otherwise convert first to crystal
    positions = to_crystal(positions, basis, in_type, alat)
    # Then to desired type
    positions = from_crystal(positions, basis, out_type, alat)
    return positions


def to_crystal(
    positions: Tau, basis: Basis, in_type: str, alat: Optional[float]
) -> Tau:
    """Convert from arbitrary coordinates to crystal."""
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom

    if in_type == "crystal":
        return positions
    elif in_type == "alat":
        if alat is None:
            raise ValueError("Value of alat must be specified")
        return alat * bohr_to_ang * positions @ np.linalg.inv(basis)
    elif in_type == "bohr":
        return bohr_to_ang * positions @ np.linalg.inv(basis)
    elif in_type == "angstrom":
        return positions @ np.linalg.inv(basis)
    else:
        raise ValueError(f"Invalid position coordinate type {in_type}")


def from_crystal(
    positions: Tau, basis: Basis, out_type: str, alat: Optional[float]
) -> Tau:
    """Convert from crystal coordinates to arbitrary coordinate units."""
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")

    # lattice vectors are rows of the basis
    if out_type == "crystal":
        return positions
    elif out_type == "alat":
        if alat is None:
            raise ValueError("Value of alat must be specified")
        return (1 / alat) * ang_to_bohr * positions @ basis
    elif out_type == "bohr":
        return ang_to_bohr * positions @ basis
    elif out_type == "angstrom":
        return positions @ basis
    else:
        raise ValueError(f"Invalid position coordinate type {out_type}")


def wrap_positions(
    positions: Tau,
    basis: Basis,
    *,
    pos_type: str = "angstrom",
    basis_units: str = "angstrom",
    alat: Optional[float] = None,
) -> Tau:
    """Wrap vector of positions using periodic boundary conditions.

    :param positions: Vector of positions; shape (n_atoms, 3)
    :param basis: Basis of the unit cell
    :param pos_type: Coordinate type of `positions`
    :param basis_units: Coordinate type of `basis`
    :param alat: lattice parameter in bohr
    :returns: Positions wrapped so they are in the the first unit cell
    """
    crystal_pos = convert_positions(
        positions, basis, pos_type, "crystal", basis_units=basis_units, alat=alat
    )
    wrapped = crystal_pos.copy()
    wrapped[wrapped > 1.0] -= 1.0
    wrapped[wrapped < 0.0] += 1.0
    return convert_positions(
        wrapped, basis, "crystal", pos_type, basis_units=basis_units, alat=alat
    )


def _read_alat_unit(in_type: str, alat: Optional[float] = None) -> float:
    """Read value of alat if specified in the unit string."""
    alat_re = re.compile(r"^alat *= +([.\d]+)$")
    if in_type == "alat":
        if alat is None:
            raise ValueError("Must specify value for alat")
        return alat
    if match := alat_re.match(in_type):
        if alat is not None:
            raise ValueError("Specifying alat twice is ambiguous")
        return float(match.group(1))
    raise ValueError(f"Bad input units for basis: {in_type!r}")


def _basis_to_bohr(basis: Basis, in_type: str, alat: Optional[float]) -> Basis:
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")
    if in_type == "angstrom":
        return ang_to_bohr * basis
    elif in_type == "bohr":
        return basis
    elif in_type == "alat":
        if alat is None:
            raise ValueError("Value of alat must be specified")
        return alat * basis
    raise ValueError(f"Bad basis coordinates {in_type}")


def convert_basis(
    basis: Basis,
    in_type: str,
    out_type: str,
    *,
    alat: Optional[float] = None,
) -> Basis:
    """Scale basis to correct units.

    The value of `alat` may be read from the input unit string or from parameter `alat`.
    It must be set if either the input or output type is 'alat'.

    :param basis: Basis in input coordinates
    :param in_type: Input units of the basis
    :param out_type: Desired output coordinates
    :param alat: Value of alat, if not specified in the type
    :returns: Rescaled basis
    """
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom

    if out_type not in _BASIS_COORDINATES:
        raise ValueError(f"Invalid basis coordinate type {out_type}.")
    # Short-circuit coordinate conversion
    if in_type == out_type and not in_type.startswith("alat"):
        return basis

    if in_type.startswith("alat"):
        alat = _read_alat_unit(in_type, alat)
        in_type = "alat"

    # First convert to Bohr
    basis_bohr = _basis_to_bohr(basis, in_type, alat)

    # Convert to output coordinates
    if out_type == "angstrom":
        return bohr_to_ang * basis_bohr
    elif out_type == "bohr":
        return basis_bohr
    elif out_type == "alat":
        if alat is None:
            raise ValueError("Value of alat must be specified")
        return basis_bohr / alat
    else:
        raise ValueError(f"Bad basis coordinates {out_type}")


def cell_alat(
    basis: Basis, units: str = "angstrom", alat: Optional[float] = None
) -> float:
    """Calculate alat (defined as the first lattice parameter in bohr)."""
    basis_bohr = convert_basis(basis, in_type=units, out_type="bohr", alat=alat)
    return np.linalg.norm(basis_bohr[0])


def cell_volume(
    basis: Basis, units: str = "angstrom", alat: Optional[float] = None
) -> float:
    """Calculate cell volume in A^3."""
    basis = convert_basis(basis, in_type=units, out_type="angstrom", alat=alat)
    return float(np.linalg.det(basis))


def cell_parameters(
    basis: Basis, units: str = "angstrom", alat: Optional[float] = None
) -> CellParam:
    """Return unit cell dimensions and angles (in angstrom)."""

    def get_angle(vec1, vec2):
        # type: (np.ndarray, np.ndarray) -> float
        cos_ab = (
            np.abs(np.dot(vec1, vec2)) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        )
        return float(np.degrees(np.arccos(cos_ab)))

    basis = convert_basis(basis, in_type=units, out_type="angstrom", alat=alat)

    len_a = np.linalg.norm(basis[0])
    len_b = np.linalg.norm(basis[1])
    len_c = np.linalg.norm(basis[2])
    alpha = get_angle(basis[1], basis[2])
    beta = get_angle(basis[0], basis[2])
    gamma = get_angle(basis[0], basis[1])
    return CellParam((len_a, len_b, len_c, alpha, beta, gamma))


def normalize_alat(
    basis: Basis, units: str, alat: Optional[float] = None
) -> Tuple[float, Basis]:
    """Rescale the basis units so that alat is the length of the first lattice vector.

    This is useful if the unit cell has changed (eg. in a vc-relax), but the units of
    alat have not.

    Args:
        basis: Lattice vectors in `units`
        units: Units of `basis`
        alat: Lattice constant used to scale `basis` units

    Returns:
        new_alat: alat used to scale the basis
        new_basis: Basis rescaled using `new_alat`. The first lattice vector will have
            length one in these new units.
    """
    basis_bohr = convert_basis(basis, units, "bohr", alat=alat)
    new_alat = cell_alat(basis_bohr, units="bohr")
    new_basis = convert_basis(basis_bohr, "bohr", "alat", alat=new_alat)
    return new_alat, new_basis


def format_basis(
    basis: Basis,
    *,
    min_space: int = 3,
    left_pad: Optional[int] = None,
    precision: int = LATTICE_PRECISION,
) -> List[str]:
    """Format a basis as an aligned array.

    Args:
        basis: 3x3 array to format as a basis
        min_space: Number of spaces between columns
        left_pad: Initial space before first column of basis
        precision: Precision to format lattice vectors

    Returns:
        List of lines of the formatted basis
    """

    def _field_fmt(value: float) -> str:
        return as_fixed_precision(value, precision)

    return columns(basis, min_space=min_space, left_pad=left_pad, convert_fn=_field_fmt)


def format_positions(
    species: Species,
    tau: Tau,
    *,
    min_space: int = 3,
    left_pad: int = 0,
    precision: int = POSITION_PRECISION,
) -> List[str]:
    """Format a list of atomic positions preceded by their species.

    Args:
        species
        tau
        min_space: Number of spaces between columns
        left_pad: Initial space before first column
        precision: Precision to format atomic positions

    Returns:
        List of lines with one formatted atomic position per line
    """

    def _field_fmt(value: Union[str, float]) -> str:
        if isinstance(value, float):
            return as_fixed_precision(value, precision)
        return str(value)

    position_fields = [[name] + list(pos) for name, pos in zip(species, tau)]
    return columns(
        position_fields, min_space=min_space, left_pad=left_pad, convert_fn=_field_fmt
    )
