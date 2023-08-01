"""`scf` subcommand parses single point energy calculations."""

import sys

from pwproc.write_data import write_xsf, add_xsf_arg


def get_scf_energy(path):
    # type: (str) -> float
    """Get the energy in Ry from pw.x output."""
    import re
    from pwproc.util import parser_one_line

    energy_re = re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry")
    energy_parser = parser_one_line(energy_re, lambda m: float(m.group(1)))

    with open(path) as f:
        return energy_parser(f)


def parse_scf(path, coord_type='crystal'):
    # type: (str, str) -> GeometryData
    """Parse pw.x output file.

    :param path: path to pw.x output
    :param coord_type: coordinate type of output

    :returns: GeometryData
    """
    from pwproc.geometry import convert_positions, GeometryData
    from pwproc.parsers import get_save_file, get_init_basis, get_init_coord

    # Run parsers on output to get geometry
    prefix = get_save_file(path)
    alat, basis = get_init_basis(path)
    ctype, species, pos = get_init_coord(path)

    # Parse file for energy
    energy = get_scf_energy(path)

    # Convert coordinates if needed
    pos = convert_positions(pos, basis, ctype, coord_type, alat=alat)

    return GeometryData(prefix, basis, species, pos,
                        energy=energy, coord_type=coord_type)


def parse_args_scf(args):
    """Argument parser for `scf` subcommand."""
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(prog='pwproc scf',
                            description="Parser for scf and nscf calculations")

    parser.add_argument('in_file', action='store', nargs='+',
                        help="List of pw.x output files")

    add_xsf_arg(parser)

    parser.add_argument('--energy', nargs='?', type=FileType('w'),
                        const=sys.stdout, metavar='FILE',
                        help="Write energy to file (in Ry)")

    return parser.parse_args(args)


def run_scf(args):
    """Execute `scf` subcommand."""
    data = {}
    for p in args.in_file:
        geom = parse_scf(p, coord_type='angstrom')
        if geom.prefix in data:
            raise ValueError("duplicate prefixes")
        else:
            data[geom.prefix] = geom

    if args.energy:
        for p, d in data.items():
            args.energy.write("{}: {}\n".format(p, d.energy))
        args.energy.close()

    if args.xsf:
        write_xsf(args.xsf, data.values())


if __name__ == '__main__':
    args = parse_args_scf(sys.argv[1:])
    run_scf(args)
