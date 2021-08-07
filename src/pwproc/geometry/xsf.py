"""Read/Write for XSF file format."""

import re
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union
import numpy as np

from pwproc.geometry import Basis, Species, Tau


def _get_next_line(lines):
    # type: (Iterator[str]) -> str
    """Consume items from `lines` until a non-comment,
    non-blank line is reached
    """
    while True:
        line = next(lines).strip()

        # Blank line
        if line == '':
            continue
        # Comment line
        if line[0] == '#':
            continue

        return line.strip()


def _match_primvec_header(line):
    # type: (str) -> Union[int, None]
    header_re = r"PRIMVEC([ \t]+[\d]+)?"

    # Match the header
    m = re.match(header_re, line)
    if not m:
        raise ValueError("PRIMVEC not found")
    else:
        s = m.group(1)
        if s is not None:
            s = int(s)
        return s


def _parse_primvec_content(lines):
    # type: (Iterator[str]) -> Basis
    from pwproc.util import parse_vector

    # Parse the basis
    basis = tuple(next(lines) for _ in range(3))
    basis = np.array(tuple(map(parse_vector, basis)))

    return Basis(basis)


def _read_primvec(lines, step=None):
    # type: (Iterator[str], Optional[int]) -> Basis
    # Match the header
    s = _match_primvec_header(_get_next_line(lines))
    assert(s == step)

    return _parse_primvec_content(lines)


def _read_first_primvec(lines):
    # type: (Iterator[str]) -> Tuple[Basis, bool]
    # Match the header
    s = _match_primvec_header(_get_next_line(lines))
    animate_cell = (s is not None)

    if animate_cell:
        assert(s == 1)

    return _parse_primvec_content(lines), animate_cell


def _read_primcoord(lines, step=None):
    # type: (Iterator[str], Optional[int]) -> Tuple[Species, Tau]
    # Match header
    header_re = r"PRIMCOORD([ \t]+[\d]+)?"
    m = re.match(header_re, _get_next_line(lines))
    if not m:
        raise ValueError("PRIMCOORD not found")
    else:
        s = m.group(1)
        if s is not None:
            s = int(m.group(1))
        assert(s == step)

    # Get number of atoms
    nat = int(re.match(r"([\d]+)[ \t]+1", _get_next_line(lines)).group(1))

    # Read coordinate lines
    coord_lines = [_get_next_line(lines) for _ in range(nat)]

    # Parse coordinates and species
    species = []
    tau = []
    for line in coord_lines:
        line = line.split()
        species.append(line[0])
        tau.append(tuple(map(float, line[1:])))

    species = tuple(species)
    tau = np.array(tau)

    assert(len(tau) == len(species) == nat)
    return Species(species), Tau(tau)


def _read_xsf_single(lines):
    # type: (Iterator[str]) -> Tuple[Basis, Species, Tau]
    # Get basis
    basis = _read_primvec(lines)
    species, tau = _read_primcoord(lines)

    return basis, species, tau


def _read_xsf_animate(lines, nsteps):
    # type: (Iterator[str], int) -> Tuple[Sequence[Basis], Species, Sequence[Tau]]
    # Decide if animating the cell
    b, animate_cell = _read_first_primvec(lines)

    # Read the first step
    species, t = _read_primcoord(lines, 1)

    # Initialize accumulators
    basis = [b]
    tau = [t]

    # Read the remaining steps
    for i in range(2, nsteps + 1):
        if animate_cell:
            b = _read_primvec(lines, i)

        s, t = _read_primcoord(lines, i)
        basis.append(b)
        assert(s == species)
        tau.append(t)

    return basis, species, tau


def read_xsf(lines):
    # type: (Iterable[str]) -> Tuple[Basis, Species, Tau]
    lines = iter(lines)
    line = _get_next_line(lines)

    animate_re = r"ANIMSTEPS[ \t]+([\d]+)"

    if re.match(animate_re, line):
        # Reading an animation
        n_steps = re.match(animate_re, line).group(1)
        n_steps = int(n_steps)
        line = _get_next_line(lines)
        assert(line == 'CRYSTAL')
        b, s, t = _read_xsf_animate(lines, n_steps)
    else:
        # Reading a single structure
        assert(line == 'CRYSTAL')
        b, s, t = _read_xsf_single(lines)

    return b, s, t


def gen_xsf(basis, species, tau, write_header=True, step=None):
    # type: (Basis, Species, Tau, bool, Optional[int]) -> Iterator[str]
    from pwproc.geometry.format_util import format_basis, format_tau

    nat = len(species)

    if write_header:
        yield 'CRYSTAL\n'

    step = ' {}'.format(step) if step is not None else ''

    yield 'PRIMVEC{}\n'.format(step)
    yield format_basis(basis) + '\n'
    yield 'PRIMCOORD{}\n'.format(step)
    yield "{} 1\n".format(nat)
    yield format_tau(species, tau) + '\n'


def gen_xsf_animate(basis, species, tau):
    # type: (Sequence[Basis], Species, Sequence[Tau]) -> Iterator[str]
    from itertools import chain

    nsteps = len(basis)
    yield 'ANIMSTEPS {}\n'.format(nsteps)
    yield 'CRYSTAL\n'

    yield from chain(*(gen_xsf(basis[i], species, tau[i],
                               write_header=False, step=(i+1))
                       for i in range(nsteps)))
