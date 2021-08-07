"""Geometry conversion utilities."""

from typing import NewType, Tuple
import numpy as np

# Vector of atomic species
Species = NewType('Species', Tuple[str, ...])
# Crystal basis [3x3]
Basis = NewType('Basis', np.ndarray)
# Position matrix [n_atoms x 3]
Tau = NewType('Tau', np.ndarray)


def convert_coords(alat, basis, tau, in_type, out_type):
    # type: (float, Basis, Tau, str, str) -> Tau
    """Convert coordinate type.

    :param alat: lattice parameter in bohr
    :param basis: basis in angstrom
    :param tau: vector of positions shape (n_atoms, 3)
    :param in_type: coordinate type of `tau`
    :param out_type: coordinate type to return
    """
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
    from scipy import constants
    bohr_to_ang = constants.value('Bohr radius') / constants.angstrom

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
    from scipy import constants
    ang_to_bohr = constants.angstrom / constants.value('Bohr radius')

    # lattice vectors are rows of the basis
    if out_type == 'crystal':
        return tau
    elif out_type == 'alat':
        return (1/alat) * ang_to_bohr * tau @ basis
    elif out_type == 'angstrom':
        return tau @ basis
    else:
        raise ValueError("Coord. type {}".format(out_type))
