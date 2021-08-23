"""Geometry conversion utilities."""

import re
from typing import NewType, Tuple

import numpy as np
import scipy.constants  # type: ignore[import]

# Vector of atomic species
Species = NewType('Species', Tuple[str, ...])
# Crystal basis [3x3]
Basis = NewType('Basis', np.ndarray)
# Position matrix [n_atoms x 3]
Tau = NewType('Tau', np.ndarray)
# Unit cell dimensions (a, b, c) in angstrom and angles (alpha, beta, gamma)
CellParam = NewType("CellParam", Tuple[float, float, float, float, float, float])


def convert_coords(alat, basis, tau, in_type, out_type):
    # type: (float, Basis, Tau, str, str) -> Tau
    """Convert coordinate type.

    :param alat: lattice parameter in bohr
    :param basis: basis in angstrom
    :param tau: vector of positions shape (n_atoms, 3)
    :param in_type: coordinate type of `tau`
    :param out_type: coordinate type to return
    """
    # TODO: deal with coord types such as "alat = 3.2"; either here or in callers
    if in_type == out_type:
        return tau

    # Otherwise convert first to crystal
    tau = to_crystal(alat, basis, tau, in_type)

    # Then to desired type
    tau = from_crystal(alat, basis, tau, out_type)

    return tau


def to_crystal(alat, basis, tau, in_type):
    # type: (float, Basis, Tau, str) -> Tau
    """Convert from arbitrary coords to crystal."""
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom

    if in_type == 'crystal':
        return tau
    elif in_type == 'alat':
        return alat * bohr_to_ang * tau @ np.linalg.inv(basis)
    elif in_type == 'angstrom':
        return tau @ np.linalg.inv(basis)
    else:
        raise ValueError("Coord. type {}".format(in_type))


def from_crystal(alat, basis, tau, out_type):
    # type: (float, Basis, Tau, str) -> Tau
    """Convert from crystal coords to arbitrary coords."""
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")

    # lattice vectors are rows of the basis
    if out_type == 'crystal':
        return tau
    elif out_type == 'alat':
        return (1 / alat) * ang_to_bohr * tau @ basis
    elif out_type == 'angstrom':
        return tau @ basis
    else:
        raise ValueError("Coord. type {}".format(out_type))


def _basis_to_bohr(basis: Basis, in_type: str) -> Basis:
    """Scale basis to bohr units."""
    alat_re = re.compile(r"^alat *= +([.\d]+)$")
    ang_to_bohr = scipy.constants.angstrom / scipy.constants.value("Bohr radius")
    if in_type == "angstrom":
        return ang_to_bohr * basis
    elif in_type == "bohr":
        return basis
    elif alat_re.match(in_type):
        alat_in = float(alat_re.match(in_type).group(1))
        return alat_in * basis
    raise ValueError(f"Bad basis coordinates {in_type}")


def cell_alat(basis: Basis, in_type="bohr") -> float:
    """Calculate alat (defined as the first lattice parameter in bohr)."""
    basis_bohr = _basis_to_bohr(basis, in_type)
    return np.linalg.norm(basis_bohr[0])


def convert_basis(basis: Basis, in_type: str, out_type: str) -> Basis:
    """Scale basis to correct units.

    :param basis: Basis in input coordinates
    :param in_type: Input units. alat should contain value of alat.
    :param out_type: Desired output coordinates. alat output will redefine alat.
    :returns: Rescaled basis
    """
    bohr_to_ang = scipy.constants.value("Bohr radius") / scipy.constants.angstrom

    # First convert to Bohr
    basis_bohr = _basis_to_bohr(basis, in_type)

    # Convert to output coordinates
    if out_type == "angstrom":
        return bohr_to_ang * basis_bohr
    elif out_type == "bohr":
        return basis_bohr
    elif out_type == "alat":
        alat_out = cell_alat(basis_bohr)
        return basis_bohr / alat_out
    else:
        raise ValueError(f"Bad basis coordinates {out_type}")


def cell_volume(basis):
    # type: (Basis) -> float
    """Calculate cell volume in A^3."""
    return float(np.linalg.det(basis))


def cell_parameters(basis):
    # type: (Basis) -> CellParam
    """Return unit cell dimensions and angles (in angstrom)."""

    def get_angle(vec1, vec2):
        # type: (np.ndarray, np.ndarray) -> float
        cos_ab = (
            np.abs(np.dot(vec1, vec2)) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        )
        return np.arccos(cos_ab)

    len_a = np.linalg.norm(basis[0])
    len_b = np.linalg.norm(basis[1])
    len_c = np.linalg.norm(basis[2])
    alpha = get_angle(basis[1], basis[2])
    beta = get_angle(basis[0], basis[2])
    gamma = get_angle(basis[0], basis[1])
    return CellParam((len_a, len_b, len_c, alpha, beta, gamma))
