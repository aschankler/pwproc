"""Read/write for Quantum ESPRESSO input files."""

import numpy as np
from typing import Iterator, Optional, Sequence, Tuple, Union

Species = Tuple[str, ...]
Basis = np.ndarray      # Shape: (3, 3)
Tau = np.ndarray        # Shape: (natoms, 3)


def read_pwi(lines):
    raise NotImplementedError


def gen_pwi(basis: Basis, species: Species, pos: Tau, coord_type: str,
            write_cell: bool = True, write_pos: bool = True,
            if_pos: Optional[Sequence[Union[None, Sequence[bool]]]] = None
           ) -> Iterator[str]:
    """Generate the `CELL_PARAMETERS` and `ATOMIC_POSITIONS` cards.

    Basis is assumed to be in angstroms and tau should agree with `coord_type`
    """
    from itertools import starmap
    from pwproc.geometry.format_util import format_basis, format_tau

    # Yield basis
    if write_cell:
        yield "CELL_PARAMETERS {}\n".format("angstrom")
        yield format_basis(basis)
        yield "\n\n"

    # Yield atomic positions
    if write_pos:
        yield "ATOMIC_POSITIONS {}\n".format(coord_type)
        if if_pos is None:
            yield format_tau(species, pos)
        else:
            # Optionally add the position freeze
            assert(len(if_pos) == len(species))

            def if_string(ifp):
                if ifp:
                    return "   {}  {}  {}".format(*map(lambda b: int(b), ifp))
                else:
                    return ""

            pos_lines = format_tau(species, pos).split('\n')
            yield from starmap(lambda p, ifp: p + if_string(ifp) + '\n', zip(pos_lines, if_pos))
