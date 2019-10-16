"""
Read/write for POSCAR files.
"""

import numpy as np


# Species = Tuple[str, ...]
# Basis = np.ndarray[3, 3]
# Tau = np.ndarray[natoms, 3]

def read_poscar(lines):
    # type: (Iterable[Text]) -> Tuple[Text, Float, Basis, Species, Tau]
    from itertools import chain, repeat
    from pwproc.util import parse_vector

    # Read data from input
    lines = iter(lines)
    name = next(lines).strip()
    alat = float(next(lines))
    basis = tuple(next(lines) for _ in range(3))
    s_name = next(lines)
    s_num = next(lines)

    # Read atomic positions
    assert(next(lines).strip() == "Cartesian")
    pos = [l for l in lines]

    # parse the basis
    basis = alat * np.array(tuple(map(parse_vector, basis)))

    # Parse the species label
    species = tuple(zip(s_name.split(), map(int, s_num.split())))
    species = tuple(chain(*(repeat(s, n) for s, n in species)))

    # Parse positions
    pos = alat * np.array(tuple(map(parse_vector, pos)))

    return name, alat, basis, species, pos


def gen_poscar(basis, species, pos, name=None, alat=1.0):
    # type: (Basis, Species, Tau, Optional[Text], Float) -> Iterator[Text]
    from geometry.format_util import format_basis, columns, FORMAT_POS

    # Write the basis information
    name = "POSCAR" if name is None else name
    yield name + '\n'
    yield "{}".format(alat) + '\n'
    yield format_basis(basis / alat) + '\n'

    # Group the positions by species
    idx = tuple(i[0] for i in sorted(enumerate(species), key=lambda x:x[1]))
    pos = pos[(idx,)]

    # Count each unique species
    s_kinds = []
    s_counts = {}
    for s in species:
        if s in s_counts:
            s_counts[s] += 1
        else:
            s_kinds.append(s)
            s_counts[s] = 1
    s_kinds = sorted(s_kinds)

    # Write atomic species
    yield columns((s_kinds, tuple(str(s_counts[s]) for s in s_kinds)), 1, lspace=0) + '\n'
    yield 'Cartesian\n'
    yield columns(pos, 3, lspace=0, s_func=FORMAT_POS)
