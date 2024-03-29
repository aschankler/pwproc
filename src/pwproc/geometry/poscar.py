"""Read/write for POSCAR files."""

from typing import Iterable, Iterator, Optional, Tuple
import numpy as np

from pwproc.geometry import Basis, Species, Tau


def content_lines(lines):
    # type: (Iterable[str]) -> Iterator[str]
    """Only contains non-blank lines of the input."""
    for line in lines:
        line = line.strip()
        if line != '':
            yield line


def read_poscar(lines, out_type='angstrom'):
    # type: (Iterable[str], str) -> Tuple[str, float, Basis, Species, Tau]
    from itertools import chain, repeat
    from pwproc.util import parse_vector
    from pwproc.geometry import convert_coords

    # Read data from input
    lines = content_lines(lines)
    name = next(lines).strip()
    alat = float(next(lines))
    basis = tuple(next(lines) for _ in range(3))
    s_name = next(lines)
    s_num = next(lines)

    # Read atomic positions
    coord_line = next(lines).strip().lower()
    if coord_line == 'direct':
        in_type = 'crystal'
    elif coord_line == 'cartesian':
        in_type = 'angstrom'
    else:
        raise ValueError('Poscar error {}'.format(coord_line))

    pos = [line for line in lines]

    # parse the basis
    basis = alat * Basis(np.array(tuple(map(parse_vector, basis))))

    # Parse the species label
    species_pairs = tuple(zip(s_name.split(), map(int, s_num.split())))
    species = tuple(chain(*(repeat(s, n) for s, n in species_pairs)))
    species = Species(species)

    # Parse positions
    pos = alat * Tau(np.array(tuple(map(parse_vector, pos))))

    # Convert the input coordinates
    pos = convert_coords(alat, basis, pos, in_type, out_type)

    return name, alat, basis, species, pos


def gen_poscar(basis, species, pos, name=None, alat=1.0):
    # type: (Basis, Species, Tau, Optional[str], float) -> Iterator[str]
    from pwproc.geometry.format_util import format_basis, columns, FORMAT_POS

    # Write the basis information
    name = "POSCAR" if name is None else name
    yield name + '\n'
    yield "{}".format(alat) + '\n'
    yield format_basis(basis / alat) + '\n'

    # Group the positions by species
    idx = tuple(i[0] for i in sorted(enumerate(species), key=lambda x: x[1]))
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
