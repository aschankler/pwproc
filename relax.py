"""
Parser for pw.x relax output.
"""

import numpy as np

from argparse import ArgumentParser
from parsers import parse_relax, get_basis, get_save_file
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

    parser.add_argument('in_file', action='store', nargs='+')
    parser.add_argument('--xsf', action='store', metavar='FILE')
    parser.add_argument('--energy', action='store', metavar='FILE')
    endpt = parser.add_mutually_exclusive_group()
    endpt.add_argument('--final', dest='endpoint', action='store_const', const='final')
    endpt.add_argument('--initial', dest='endpoint', action='store_const', const='initial')

    return parser.parse_args(args)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])

    relax_data = {}

    def join_relax(old_data, new_data):
        old_en, old_species, old_tau = old_data
        new_en, new_species, new_tau = new_data
        assert old_species == new_species
        return (old_en+new_en, old_species, old_tau+new_tau)

    # Parse files
    for p in args.in_file:
        # TODO: it would be nice to store all this data in a class...
        final, relax = parse_relax(p, coord_type='angstrom')
        _, basis = get_basis(p)
        prefix = get_save_file(p)

        # TODO: this merge is broken if a relax is split over >2 files
        if prefix in relax_data:
            old_basis, old_final, old_relax = relax_data[prefix]
            assert np.allclose(basis, old_basis)
            if final is not None:
                relax_data[prefix] = (basis, final, join_relax(old_relax, relax))
            elif old_final is not None:
                relax_data[prefix] = (basis, old_final, join_relax(relax, old_relax))
            else:
                # Join arbitrarily
                relax_data[prefix] = (basis, final, join_relax(old_relax, relax))
        else:
            relax_data[prefix] = (basis, final, relax)

    # Check final
    if args.endpoint == 'final':
        for pref, data in relax_data.items():
            if data[1] is None:
                raise ValueError('Relaxation did not finish for {}'.format(pref))

    # Write XSF file
    def save_xsf(path, data):
        basis, final_data, relax_data = data
        if args.endpoint is None:
            _, species, tau = relax_data
        elif args.endpoint == 'final':
            _, species, tau = final_data
        else:
            _, species, tau = relax_data
            tau = tau[0]
        with open(path, 'w') as f:
            f.writelines(gen_xsf(basis, species, tau))

    if args.xsf is not None:
        if '{PREFIX}' not in args.xsf:
            if len(relax_data) > 1:
                raise ValueError('Saving multiple structures to same file')
            else:
                # Grab the first (and only) entry
                data = next(iter(relax_data.values()))
                save_xsf(args.xsf, data)
        else:
            for prefix, data in relax_data.items():
                save_xsf(args.xsf.format(PREFIX=prefix), data)

    # Write energy
    if args.energy is not None:
        with open(args.energy, 'w') as f:
            for pref in relax_data:
                if args.endpoint == 'final':
                    energy = relax_data[pref][1][0]
                    f.write("{}: {}\n".format(pref, energy))
                elif args.endpoint == 'initial':
                    energy = relax_data[pref][2][0]
                    f.write("{}: {}\n".format(pref, energy[0]))
                else:
                    energy = relax_data[pref][2][0]
                    f.write(pref + '\n')
                    for e in energy:
                        f.write('{}\n'.format(e))
