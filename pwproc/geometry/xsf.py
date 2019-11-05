"""
Read/Write for XSF file format
"""

# Species = Tuple[str, ...]
# Basis = np.ndarray[3, 3]
# Tau = np.ndarray[natoms, 3]


def _get_next_line(lines):
    # type: (Iterator[str]) -> str
    """Consume items from `lines` until a non-comment,
    non-blank line is reached
    """
    while True:
        l = next(lines).strip()
        blank = l == ''
        com = l[0] == '#'
        if not (blank or com):
            return l


def _read_xsf_single(lines):
    # type: (Iterator[str]) -> Tuple[Basis, Species, Tau]
    import re
    import numpy as np
    from pwproc.util import parse_vector

    basis_re = re.compile(r"PRIMVEC( [\d]+)?")
    tau_re = re.compile(r"PRIMCOORD( [\d]+)?")

    # Get basis
    l = _get_next_line(lines)
    assert(basis_re.match(l))
    basis = tuple(next(lines) for _ in range(3))
    basis = np.array(tuple(map(parse_vector, basis)))

    # Get coord
    l = _get_next_line(lines)
    assert(tau_re.match(l))

    # Get number of atoms
    nat = int(re.match(r"([\d]+) 1", _get_next_line(lines)).group(1))

    # Read coodinate lines
    coord_lines = [next(lines) for _ in range(nat)]

    # Parse coordinates and species
    species = []
    tau = []
    for l in coord_lines:
        l = l.split()
        species.append(l[0])
        tau.append(tuple(map(float, l[1:])))

    species = tuple(species)
    tau = np.array(tau)

    assert(len(tau) == len(species) == nat)

    return basis, species, tau


def _read_xsf_animate(lines):
    raise NotImplementedError


def read_xsf(lines):
    # type: (Iterable[Text]) -> Tuple[Basis, Species, Tau]
    import re

    lines = iter(lines)
    l = _get_next_line(lines)

    if re.match(r"ANIMSTEPS ([\d]+)", l):
        # Reading an animation
        # TODO: does not consider fixed cell animations
        nsteps = re.match(r"ANIMSTEPS ([\d]+)", l).group(1)
        nsteps = int(nsteps)
        l = _get_next_line(lines)
        assert(l.strip() == 'CRYSTAL')
        b, s, t = _read_xsf_animate(lines)
    else:
        # Reading a single structure
        assert(l.strip() == 'CRYSTAL')
        b, s, t = _read_xsf_single(lines)

    return b, s, t


def gen_xsf(basis, species, tau, write_header=True, step=None):
    # type: (Basis, Species, Tau, bool) -> Iterator[Text]
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
    # type: (Sequence[basis], Species, Sequence[Tau]) -> Iterator[Text]
    from itertools import chain

    nsteps = len(basis)
    yield 'ANIMSTEPS {}\n'.format(nsteps)
    yield 'CRYSTAL\n'

    yield from chain(*(gen_xsf(basis[i], species, tau[i],
                               write_header=False, step=(i+1))
                       for i in range(nsteps)))

