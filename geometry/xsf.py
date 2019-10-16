
# Species = Tuple[str, ...]
# Basis = np.ndarray[3, 3]
# Tau = np.ndarray[natoms, 3]


def read_xsf(lines):
    raise NotImplementedError


def gen_xsf(basis, species, tau, write_header=True, step=None):
    # type: (Basis, Species, Tau, bool) -> Iterator[Text]
    from geometry.format_util import format_basis, format_tau

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

    nsteps = len(basis)
    yield 'ANIMSTEPS {}\n'.format(nsteps)
    yield 'CRYSTAL\n'

    for i in range(nsteps):
        gen_xsf(basis[i], species, tau[i], write_header=False, step=(i+1))
