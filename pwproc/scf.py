"""
`scf` subcommand parses single point energy calculations
"""


def get_scf_energy(path):
    # type: (str) -> float
    """Get the energy in Ry from pw.x output"""
    import re
    from pwproc.util import parser_one_line

    energy_re = re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry")
    energy_parser = parser_one_line(energy_re, lambda m: float(m.group(1)))

    with open(path) as f:
        return energy_parser(f)


def parse_scf(path, coord_type='crystal'):
    # type: (str, str) -> GeometryData
    """Parse pw.x output file

    :param path: path to pw.x output
    :param coord_type: coordinate type of output

    :returns: GeometryData
    """
    from pwproc.geometry import GeometryData
    from pwproc.parsers import get_save_file, get_init_basis, get_init_coord
    from pwproc.util import convert_coords

    # Run parsers on output to get geometry
    prefix = get_save_file(path)
    alat, basis = get_init_basis(path)
    ctype, species, pos = get_init_coord(path)

    # Parse file for energy
    energy = get_scf_energy(path)

    # Convert coordinates if needed
    pos = convert_coords(alat, basis, pos, ctype, coord_type)

    return GeometryData(prefix, basis, species, pos,
                        energy=energy, coord_type=coord_type)


def parse_args(args):
    """Argument parser for `scf` subcommand."""
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(prog='pwproc scf',
                            description="Parser for scf and nscf calculations")

    parser.add_argument('in_file', action='store', nargs='+',
                        help="List of pw.x output files")
    parser.add_argument('--xsf', action='store', metavar='FILE',
                        help="Write xsf structures to file. The key `{PREFIX}` in FILE" \
                        " is substituted for the calculation prefix")
    parser.add_argument('--energy', nargs='?', type=FileType('w'), const=sys.stdout,
                        metavar='FILE', help="Write energy to file (in Ry)")

    return parser.parse_args(args)



def scf(args):
    """Main program for `scf` subcommand"""
    data = {}
    for p in args.in_file:
        geom = parse_scf(p, coord_type='angstrom')
        if geom.prefix in data:
            raise ValueError("duplicate prefixes")
        else:
            data[geom.prefix] = geom

    if args.energy:
        args.energy.writelines("{}: {}\n".format(p, d.energy) for p, d in data.items())
        args.energy.close()

    if args.xsf:
        from pwproc.relax import write_xsf
        write_xsf(args.xsf, data)


if __name__ == '__main__':
    import sys

    args = parse_args(sys.argv[1:])
    scf(args)

