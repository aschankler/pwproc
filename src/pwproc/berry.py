"""Extract polarization from pw.x calculation.

Phase [-pi, pi) is mapped to [-1/2, 1/2) if mod is 1, to [-1, 1) if mod is 2
Polarization direction vector is in cartesian coordinates
Polarization is in units of C/m^2
"""

from collections import namedtuple

Polarization = namedtuple('Polarization', ('gdir', 'phase', 'vec', 'pol'))


def make_berry_parser():
    import re
    from pwproc.util import parser_one_line

    def parse_vector(vec, *, num_re=re.compile(r"-?[.\d]+")):
        return tuple(map(float, num_re.findall(vec)))

    gdir_re = re.compile(r"^[ \w]* direction of vector ([123])$")
    phase_re = re.compile(r"^ +TOTAL PHASE: +(-?[.\d]+) \(mod (1|2)\)$")
    vec_re = re.compile(r"^ +The polarization direction is: +\(( *(?:-?[.\d]+[ ,]*){3})\)$")
    pol_re = re.compile(r"^ +P = +(-?[.\d]+) +\(mod +([.\d]+)\) +C/m\^2$")

    g_parser = parser_one_line(gdir_re, lambda m: int(m.group(1)))
    phase_parser = parser_one_line(phase_re, lambda m: (float(m.group(1)), int(m.group(2))))
    vec_parser = parser_one_line(vec_re, lambda m: parse_vector(m.group(1)))
    pol_parser = parser_one_line(pol_re, lambda m: tuple(map(float, m.group(1, 2))))

    def berry_parser(f_name):
        # type: (Path) -> Polarization
        with open(f_name) as f:
            gdir = g_parser(f)
            f.seek(0)
            phase = phase_parser(f)
            f.seek(0)
            vec = vec_parser(f)
            f.seek(0)
            pol = pol_parser(f)
        return Polarization(gdir, phase, vec, pol)

    return berry_parser


def parse_berry_output(in_files):
    # type: (Sequence[Path]) -> Mapping[str, Polarization]
    """Parse output from berry phase calculation."""
    from pwproc.parsers import get_save_file

    berry_parser = make_berry_parser()
    pol_data = []

    for p in in_files:
        name = get_save_file(p)
        pol = berry_parser(p)
        pol_data.append((name, pol))

    return pol_data


def write_indexed(pol_data, what, where):
    for dat in pol_data:
        k = "{} ({!s})".format(dat[0], dat[1].gdir)
        where.write(k + ": ")
        where.write(what(dat[1]))
        where.write('\n')


def parse_args_berry(args):
    """Argument parser for `berry` subcommand."""
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(prog='pwproc berry')
    parser.add_argument('in_file', nargs='+', type=Path)
    parser.add_argument('--phase', type=Path)
    parser.add_argument('--vec', type=Path)
    parser.add_argument('--pol', type=Path)
    parser.add_argument('--out', type=str)

    return parser.parse_args(args)


def run_berry(args):
    from pathlib import Path
    pol_data = parse_berry_output(args.in_file)

    # Sort output by name then gdir
    pol_data = sorted(pol_data, key=lambda d: (d[0], d[1].gdir))

    if args.out is not None:
        args.phase = Path.cwd().joinpath(args.out + "_phase.dat")
        args.vec = Path.cwd().joinpath(args.out + "_vec.dat")
        args.pol = Path.cwd().joinpath(args.out + "_pol.dat")
    if args.phase is not None:
        with open(args.phase, 'w') as f:
            write_indexed(pol_data, lambda pol: "{: 6.5f} (mod {!s})".format(*pol.phase), f)
    if args.vec is not None:
        with open(args.vec, 'w') as f:
            write_indexed(pol_data, lambda pol: "{: 6.5f} {: 6.5f} {: 6.5f}".format(*pol.vec), f)
    if args.pol is not None:
        with open(args.pol, 'w') as f:
            write_indexed(pol_data, lambda pol: "{: 6.5f} (mod {:6.5f})".format(*pol.pol), f)


if __name__ == '__main__':
    import sys

    args = parse_args_berry(sys.argv[1:])
    run_berry(args)
