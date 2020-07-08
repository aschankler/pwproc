"""Parser for pw.x relax output."""


def parse_file(path):
    from pwproc.parsers import parse_relax

    final_data, relax_data = parse_relax(path)
    prefix = relax_data.prefix

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
    # type: (TextIO, Mapping[str, GeometryData]) -> None
    """Write energy data to file."""
    from pwproc.geometry import RelaxData
    for prefix, dat in data.items():
        if type(dat) is RelaxData:
            e_file.write("{} {}\n".format(prefix, len(dat.energy)))
            for e in dat.energy:
                e_file.write('{}\n'.format(e))
        else:
            e_file.write("{}: {}\n".format(prefix, dat.energy))


def write_xsf(xsf, data):
    # type: (str, Mapping[str, GeometryData]) -> None
    """Write structure data to xsf files."""
    if '{PREFIX}' not in xsf:
        if len(data) != 1:
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
    import sys
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(prog='pwproc relax',
                            description="Parser for relax and vc-relax output")

    parser.add_argument('in_file', action='store', nargs='+',
                        help="List of pw.x output files")
    parser.add_argument('--xsf', action='store', metavar='FILE',
                        help="Write xsf structures to file. The key `{PREFIX}`"
                        " in FILE is replaced by the calculation prefix")
    parser.add_argument('--energy', nargs='?', action='store', type=FileType('w'),
                        const=sys.stdout, default=None, metavar='FILE',
                        help="Write energy to file (in Ry)")
    endpt = parser.add_mutually_exclusive_group()
    endpt.add_argument('--final', dest='endpoint', action='store_const',
                       const='final', help="Save data only for the final"
                       " structure. Warn if relaxation did not finish")
    endpt.add_argument('--initial', dest='endpoint', action='store_const',
                       const='initial', help="Save data only for the initial"
                       " structure")

    return parser.parse_args(args)


def relax(args):
    """Main function for `relax` subcommand."""
    # Parse the output files
    relax_data = parse_files(args.in_file)

    # Take the desired step
    out_data = {}
    for prefix, data in relax_data.items():
        final, relax = data
        if args.endpoint == 'final':
            if final is None:
                print('Relaxation did not finish for {}'.format(prefix))
            else:
                out_data[prefix] = final
        elif args.endpoint == 'initial':
            out_data[prefix] = relax.get_init()
        else:
            out_data[prefix] = relax

    # Write XSF file
    if args.xsf:
        write_xsf(args.xsf, out_data)

    # Write energy
    if args.energy:
        write_energy(args.energy, out_data)
        args.energy.close()


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    relax(args)
