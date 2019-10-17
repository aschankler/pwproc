"""
Parser for pw.x relax output.
"""

from geometry.data import GeometryData, RelaxData


def parse_file(path):
    from pwproc.parsers import parse_relax, get_save_file

    final, relax = parse_relax(path, coord_type='angstrom')
    prefix = get_save_file(path)

    if final is None:
        final_data = None
    else:
        final_data = GeometryData(prefix, final[1], final[2], final[3], energy=final[0])

    relax_data = RelaxData(prefix, relax[1], relax[2], relax[3], energy=relax[0])

    return (prefix, final_data, relax_data)


def parse_files(paths):
    data = {}

    for p in paths:
        prefix, final, relax = parse_file(p)

        if prefix not in data:
            data[prefix] = (final, relax)
        else:
            # We must merge several files
            # TODO: this merge is broken if a relax is split over >2 files
            old_final, old_relax = data[prefix]
            if final is not None:
                # `final` and `relax` are the more recent steps
                data[prefix] = (final, old_relax | relax)
            elif old_final is not None:
                # `old_final` and `old_relax` are more recent
                data[prefix] = (old_final, relax | old_relax)
            else:
                # Join arbitrarily
                data[prefix] = (final, old_relax | relax)

    return data


def write_energy(e_file, data):
    # type: (Text, Mapping[str, GeometryData]) -> None
    """Write energy data to file."""
    with open(e_file, 'w') as f:
        for prefix, dat in data.items():
            if type(dat) is RelaxData:
                f.write(prefix + '\n')
                for e in dat.energy:
                    f.write('{}\n'.format(e))
            else:
                f.write("{}: {}\n".format(prefix, dat.energy))


def write_xsf(xsf, data):
    # type: (Text, Mapping[str, GeometryData]) -> None
    """Write structure data to xsf files."""

    if '{PREFIX}' not in xsf:
        if len(data) > 1:
            raise ValueError('Saving multiple structures to same file')
        else:
            # Grab the first (and only) entry
            save_data = zip((xsf,), data.values())
    else:
        save_data = ((xsf.format(PREFIX=pref), geom_data)
                     for pref, geom_data in data.items())

    for path, geom_data in save_data:
        with open(path, 'w') as xsf_f:
            xsf_f.writelines(geom_data.to_xsf())


def parse_args(args):
    """Argument parser for `relax` subcommand."""
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='pwproc relax')

    parser.add_argument('in_file', action='store', nargs='+')
    parser.add_argument('--xsf', action='store', metavar='FILE')
    parser.add_argument('--energy', action='store', metavar='FILE')
    endpt = parser.add_mutually_exclusive_group()
    endpt.add_argument('--final', dest='endpoint', action='store_const', const='final')
    endpt.add_argument('--initial', dest='endpoint', action='store_const', const='initial')

    return parser.parse_args(args)


def relax(args):
    """Main function for `relax` subcommand."""
    # Parse the output files
    relax_data = parse_files(args.in_file)

    # Take the desired step
    for pref, data in relax_data.items():
        final, relax = data
        if args.endpoint == 'final':
            if data[0] is None:
                raise ValueError('Relaxation did not finish for {}'.format(pref))
            else:
                relax_data[pref] = final
        elif args.endpoint == 'initial':
            relax_data[pref] = relax.get_init()
        else:
            relax_data[pref] = relax

    # Write XSF file
    if args.xsf:
        write_xsf(args.xsf, relax_data)

    # Write energy
    if args.energy:
        write_energy(args.energy, relax_data)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    relax(args)

