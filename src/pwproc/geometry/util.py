"""Geometry conversion utilities."""

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
