"""
Parser for pw.x bands output files.
"""

import re
from pathlib import Path

import numpy as np

from pwproc.util import parse_vector


def group(it, n):
    # type: (Iterable[T], Integral) -> Iterator[Tuple[T, ...]]
    """Split iterable into groups of n"""
    from collections import deque
    it = iter(it)
    q = deque([], n)
    while True:
        for _ in range(n):
            try:
                q.append(next(it))
            except StopIteration:
                if len(q) == 0:
                    return
                else:
                    raise ValueError("Unequal bunching")
        yield tuple(q)
        q.clear()


def parse_kpt(kpt_str, bands_str):
    # type: (Text, Iterable[Text]) -> Tuple[np.ndarray, np.ndarray]
    from itertools import chain

    k_coord = np.array(parse_vector(kpt_str))
    bands = np.array(tuple(chain(*map(parse_vector, bands_str))))

    return (k_coord, bands)


def parse_bands_out(lines):
    # type: (Iterable[Text]) -> Tuple[np.ndarray, np.ndarray]
    """Extract bands from bands.x output.
    :returns:
        kpts: array[nkpt, 3]
        bands: array[nkpt, nband]
    """
    from itertools import chain

    lines = iter(lines)
    header = re.search(r'&plot nbnd=[\s]*([\d]+), nks=[\s]*([\d]+) /', next(lines))
    if header is None:
        raise ValueError("Malformed header for bands.x output")

    nbnd, nk = map(int, header.groups())

    # Parse output into a stream of numbers then group into bunches
    # like (kpt[3], bands[nbnd])
    number_stream = chain.from_iterable((parse_vector(l) for l in lines))
    kpt_bunches = map(lambda a: (a[:3], a[3:]), group(number_stream, nbnd+3))
    kpts, bands = map(np.array, zip(*kpt_bunches))

    return kpts, bands


def parse_pwx_out(lines):
    # type: (Iterable[Text]) -> Tuple[np.ndarray, np.ndarray]
    """Extract bands from pw.x output.
    :returns:
        kpts: array[nkpt, 3]
        bands: array[nkpt, nband]
    """
    header_re = re.compile(r"^[\s]*k =[\s]?((?:[\d.-]+ ){3})\( *[\d]+ PWs\)[\s]+bands \(ev\):$")
    bands_re = re.compile(r"^[\s]+(?:(?:[\d.-]+)[\s]*)+$")

    buffering = False
    found_bands = False
    this_kpt = ''
    parsed_buffer = []
    bands_buffer = []

    for line in lines:
        if buffering:
            if bands_re.match(line):
                found_bands = True
                bands_buffer.append(line)
            elif found_bands:
                parsed_buffer.append(parse_kpt(this_kpt, bands_buffer))
                buffering = False
                found_bands = False
                bands_buffer = []

        else:
            match = header_re.match(line)
            if match:
                this_kpt = match.group(1)
                buffering = True
    
    return tuple(map(np.stack, zip(*parsed_buffer)))


def kpath_coord(kpt_list):
    """Calculate displacement along a continuous path coordinate"""
    deltas = np.linalg.norm(kpt_list[1:] - kpt_list[:-1], axis=1)
    kcoord = np.zeros(deltas.shape[0] + 1)
    kcoord[1:] = deltas
    kcoord = np.cumsum(kcoord)
    kcoord /= kcoord[-1]
    return kcoord


def save_bands(kpath, kpts, bands, csv_path=None, npz_path=None):
    """Save band data in csv or npz format."""

    if npz_path is not None:
        np.savez(npz_path, kpath=kpath, kpoints=kpts, bands=bands)

    if csv_path is not None:
        import csv
        from itertools import chain
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            nbands = bands.shape[1]
            writer.writerow(['kpath', 'kpoint', None, None, 'bands'] + [None]*nbands)
            writer.writerows(map(lambda x: chain(*x), zip(kpath.reshape(-1,1), kpts, bands)))


def parse_args_bands(args):
    from argparse import ArgumentParser

    parser = ArgumentParser(prog='pwproc bands',
                            description="Parse output from bands.x and pw.x")

    parser.add_argument('in_file', action='store', type=Path, help="pw.x or bands.x output file")
    in_grp = parser.add_mutually_exclusive_group()
    in_grp.add_argument(
        "--bands",
        action="store_const",
        const="bands",
        dest="in_type",
        help="Output is from bands.x",
    )
    in_grp.add_argument(
        "--pwx",
        action="store_const",
        const="pwx",
        dest="in_type",
        help="Output is from pw.x (default)",
    )
    out_grp = parser.add_argument_group(
        title="Output",
        description="Output files record kpoint coordinates, band"
        " energies, and progress on a continuous path coordinate",
    )
    out_grp.add_argument(
        "--npz",
        nargs="?",
        type=Path,
        const=Path.cwd(),
        metavar="FILE",
        help="Write data in npz format",
    )
    out_grp.add_argument(
        "--csv",
        nargs="?",
        type=Path,
        const=Path.cwd(),
        metavar="FILE",
        help="Write data to csv file",
    )

    # Apply defaults
    args = parser.parse_args(args)
    if args.in_type is None:
        args.in_type = 'pwx'

    def gen_out_path(out_path, ext):
        if out_path is not None and out_path.is_dir():
            out_path = out_path.joinpath(args.in_file.name)

            # Modify the output extension
            strippable_ext = ('.dat', '.out')
            if out_path.suffix in strippable_ext:
                return out_path.with_suffix(ext)
            else:
                return out_path.with_name(out_path.name + ext)
        else:
            return out_path

    args.npz = gen_out_path(args.npz, '.npz')
    args.csv = gen_out_path(args.csv, '.csv')

    return args


def run_bands(args):
    with open(args.in_file, 'r') as f:
        if args.in_type == 'pwx':
            kpt, bands = parse_pwx_out(f)
        elif args.in_type == 'bands':
            kpt, bands = parse_bands_out(f)
        else:
            raise ValueError

    kpath = kpath_coord(kpt)
    save_bands(kpath, kpt, bands, npz_path=args.npz, csv_path=args.csv)


if __name__ == '__main__':
    import sys
    args = parse_args_bands(sys.argv[1:])
    run_bands(args)

