"""Defines the `fermi` subcommand for extracting the fermi energy."""

from argparse import Namespace
from typing import TextIO
from pathlib import Path
from typing import Iterable, Sequence


def find_fermi(path):
    # type: (Path) -> Iterable[float]
    """Extract all records of the Fermi energy from pw.x output."""
    import re
    from pwproc.util import parser_one_line

    fermi_re = re.compile(r"[ \t]+the Fermi energy is[ \t]+(-?[.\d]+) ev")
    fermi_parser = parser_one_line(fermi_re, lambda m: float(m.group(1)), find_multiple=True)

    with open(path) as f:
        fe = fermi_parser(f)

    return fe


def write_fermi(outfile, data, write_all=False):
    # type: (TextIO, Iterable[str, Sequence[float]], bool) -> None
    """Write fermi energy to data file."""
    for prefix, fe in data:
        if write_all:
            # Write all FE for each file
            outfile.write("{} {}\n".format(prefix, len(fe)))
            for en in fe:
                outfile.write('{}\n'.format(en))
        else:
            # Only write the final FE
            outfile.write("{}: {}\n".format(prefix, fe[-1]))


def parse_args(args):
    # type: (Sequence[str]) -> Namespace
    """Parse comandline arguments for fermi subcommand."""
    import sys
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(prog='pwproc fermi')

    parser.add_argument('in_file', action='store', nargs='+')
    parser.add_argument('--out', action='store', type=FileType('w'), default=sys.stdout)
    parser.add_argument('--all', action='store_true')

    return parser.parse_args(args)


def fermi(args):
    # type: (Namespace) -> None
    """Execute fermi command."""
    from pwproc.parsers import get_save_file

    names = [get_save_file(p) for p in args.in_file]
    fe = [find_fermi(p) for p in args.in_file]

    write_fermi(args.out, zip(names, fe), write_all=args.all)
    args.out.close()


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    fermi(args)
