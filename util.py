
import re
import numpy as np

def parse_vector(s, *_, num_re=re.compile(r"[\d.-]+")):
    # type: (Text) -> Tuple[float]
    """Convert a vector string into a tuple."""
    return tuple(map(float, num_re.findall(s)))


def convert_coords(alat, basis, tau, in_type, out_type):
    # type: (float, np.ndarray, np.ndarray, str, str) -> np.ndarray
    """Converts coordinate type.

    :param alat: lattice parameter in bohr
    :param basis: basis in angstrom
    :param tau: vector of posititons shape (natoms, 3)
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
    # type: (float, np.ndarray, np.ndarray, str) -> np.ndarray
    if in_type == 'crystal':
        return tau
    elif in_type == 'angstrom':
        return (np.linalg.inv(basis) @ tau.T).T
    else:
        raise ValueError("Coord. type {}".format(in_type))


def from_crystal(alat, basis, tau, out_type):
    # type: (float, np.ndarray, np.ndarray, str) -> np.ndarray
    if out_type == 'crystal':
        return tau
    elif out_type == 'angstrom':
        return (basis @ tau.T).T
    else:
        raise ValueError("Coord. type {}".format(in_type))
