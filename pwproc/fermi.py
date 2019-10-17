
import re
from pwproc.parsers import get_save_file
from pwproc.util import parser_one_line


def find_fermi(path):
    fermi_re = re.compile(r"[ \t]+the Fermi energy is[ \t]+(-?[.\d]+) ev")
    fermi_parser = parser_one_line(fermi_re, lambda m: float(m.group(1)), find_multiple=True)

    with open(path) as f:
        fe = fermi_parser(f)

    return fe


def parse_args(args):
    from argparse import ArgumentParser

    parser = ArgumentParser(prog='pwproc fermi')

    parser.add_argument('in_file', action='store', nargs='+')

    return parser.parse_args(args)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])

    names = [get_save_file(p) for p in args.in_file]
    fe = [find_fermi(p)[-1] for p in args.in_file]

    for n, f in zip(names, fe):
        print("{}: {}".format(n, f))
