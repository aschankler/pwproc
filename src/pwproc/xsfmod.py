"""Module to read and process .xsf files."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np

from pwproc.geometry import GeometryData, RelaxData
from pwproc.geometry.xsf import read_xsf
from pwproc.write_data import add_data_args, write_data


def parse_file(xsf_path: Path) -> Union[GeometryData, RelaxData]:
    if xsf_path.suffix == ".xsf":
        prefix = xsf_path.stem
        is_axsf = False
        with open(xsf_path, encoding="utf8") as xsf_f:
            struct = read_xsf(xsf_f, axsf_allowed=False)
    elif xsf_path.suffix == ".axsf":
        prefix = xsf_path.stem
        is_axsf = True
        with open(xsf_path, encoding="utf8") as xsf_f:
            struct = read_xsf(xsf_f, axsf_expected=True)
    else:
        prefix = xsf_path.name
        with open(xsf_path, encoding="utf8") as xsf_f:
            struct = read_xsf(xsf_f)
        # Todo: there are better ways of checking this
        is_axsf = isinstance(struct[2], np.ndarray)

    if is_axsf:
        return RelaxData(prefix, *struct)
    return GeometryData(prefix, *struct)


def parse_args_xsf(args) -> Namespace:
    """Argument parser for `xsf` subcommand."""
    parser = ArgumentParser(prog="pwproc xsf", description="Parser for .xsf files")

    parser.add_argument(
        "in_file", action="store", nargs="+", help="List of .xsf and .axsf files"
    )

    add_data_args(parser, ("vol", "lat"))

    return parser.parse_args(args)


def run_xsf(args: Namespace) -> None:
    """Execute `xsf` subcommand."""
    data = []
    for path in args.in_file:
        data.append(parse_file(Path(path)))

    if args.dtags:
        write_data(data, sys.stdout, args.dtags, getattr(args, "extra_data", None))


if __name__ == "__main__":
    run_xsf(parse_args_xsf(sys.argv[1:]))
