"""
Parser for pw.x bands output files.
"""

import numpy as np
import re


def parse_vector(s, *_, num_re=re.compile(r"[\d.-]+")):
    # type: (Text) -> Tuple[float]
    return tuple(map(float, num_re.findall(s)))


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


def parse_args(args):
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('in_file', action='store')
    in_grp = parser.add_mutually_exclusive_group()
    in_grp.add_argument('--bands', action='store_const', const='bands', dest='in_type')
    in_grp.add_argument('--pwx', action='store_const', const='pwx', dest='in_type')
    out_grp = parser.add_argument_group('Output type')
    out_grp.add_argument('--npz', action='store')
    out_grp.add_argument('--csv', action='store')

    return parser.parse_args(args)


if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    if args.in_type is None:
        args.in_type = 'pwx'

    with open(args.in_file, 'r') as f:
        if args.in_type == 'pwx':
            kpt, bands = parse_pwx_out(f)
        elif args.in_type == 'bands':
            kpt, bands = parse_bands_out(f)
        else:
            raise ValueError

    kpath = kpath_coord(kpt)

    if args.npz is not None:
        np.savez(args.npz, kpath=kpath, kpoints=kpt, bands=bands)

    if args.csv is not None:
        import csv
        from itertools import chain
        with open(args.csv, 'w') as f:
            writer = csv.writer(f)
            nbands = bands.shape[1]
            writer.writerow(['kpath', 'kpoint', None, None, 'bands'] + [None]*nbands)
            writer.writerows(map(lambda x: chain(*x), zip(kpath.reshape(-1,1), kpt, bands)))
