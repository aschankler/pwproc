"""
Parser for pw.x relax output.
"""

import numpy as np

from argparse import ArgumentParser
from parsers import parse_relax, get_basis
from format_util import format_basis, format_tau


def gen_xsf(basis, species, tau):
    # Check if we are animating
    animate = (type(tau) is not np.ndarray)

    if animate:
        yield 'ANIMSTEPS {}\n'.format(len(tau))

    yield 'CRYSTAL\n'
    yield 'PRIMVEC\n'
    yield format_basis(basis) + '\n'

    if not animate:
        yield 'PRIMCOORD\n'
        yield "{} 1\n".format(len(tau))
        yield format_tau(species, tau) + '\n'
    else:
        for i in range(len(tau)):
            yield 'PRIMCOORD {}\n'.format(i+1)
            yield "{} 1\n".format(len(species))
            yield format_tau(species, tau[i]) + '\n'


def parse_args(args):
    parser = ArgumentParser(prog='pwproc relax')

    parser.add_argument('in_file', action='store')
    parser.add_argument('--xsf', action='store', metavar='FILE')
    parser.add_argument('--energy', action='store', metavar='FILE')
    parser.add_argument('--final', action='store_true')

    return parser.parse_args(args)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])

    # Parse files
    final, relax = parse_relax(args.in_file, coord_type='angstrom')
    _, basis = get_basis(args.in_file)

    # Keep final or full relax data
    if args.final:
        if final is not None:
            energy, species, tau = final
            energy = (energy,)
        else:
            raise ValueError('Relaxation did not finish')
    else:
        energy, species, tau = relax

    # Write XSF file
    if args.xsf is not None:
        with open(args.xsf, 'w') as f:
            f.writelines(gen_xsf(basis, species, tau))

    # Write energy
    if args.energy is not None:
        with open(args.energy, 'w') as f:
            f.write(args.in_file + '\n')
            for e in energy:
                f.write("{}\n".format(e))
