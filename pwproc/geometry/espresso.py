"""Read/write for Quantum ESPRESSO input files."""

import numpy as np
from typing import Iterator, Tuple

Species = Tuple[str, ...]
Basis = np.ndarray      # Shape: (3, 3)
Tau = np.ndarray        # Shape: (natoms, 3)


def read_pwi(lines):
    raise NotImplementedError


def gen_pwi(basis, species, pos, coord_type, write_cell=True, write_pos=True):
    # type: (Basis, Species, Tau, str, bool, bool) -> Iterator[str]
    """Generate the `CELL_PARAMETERS` and `ATOMIC_POSITIONS` cards.

    Basis is assumed to be in angstroms and tau should agree with `coord_type`
    """
    from pwproc.geometry.format_util import format_basis, format_tau

    # Yield basis
    if write_cell:
        yield "CELL_PARAMETERS {}\n".format("angstrom")
        yield format_basis(basis)
        yield "\n\n"

    # Yield atomic positions
    if write_pos:
        yield "ATOMIC_POSITIONS {}\n".format(coord_type)
        yield format_tau(species, pos)