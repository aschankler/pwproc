"""Parser for band structures produced by pw.x and bands.x."""

import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from pwproc.util import parse_vector

# Todo: add functionality to find the band gap/fermi energy

_T = TypeVar("_T")


def group(items: Iterable[_T], group_len: int) -> Iterator[Tuple[_T, ...]]:
    """Split iterable into groups of length n."""
    # pylint: disable=import-outside-toplevel
    from collections import deque

    iter_items = iter(items)
    batch_queue = deque([], group_len)  # type: deque[_T]
    while True:
        for _ in range(group_len):
            try:
                batch_queue.append(next(iter_items))
            except StopIteration:
                if len(batch_queue) == 0:
                    return
                else:
                    raise ValueError("Unequal bunching") from None
        yield tuple(batch_queue)
        batch_queue.clear()


def parse_bands_out(lines: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract band structure from bands.x output.

    Args:
        lines: Lines from the bands.x output file

    Returns:
        kpts: Array of kpt positions. Shape: [nkpt, 3]
        bands: Array of band energies. Shape: [nkpt, nband]
    """
    # pylint: disable=import-outside-toplevel
    from itertools import chain

    lines = iter(lines)
    header = re.search(
        r"&plot nbnd=[\s]*(?P<nbnd>[\d]+), nks=[\s]*(?P<nk>[\d]+) /", next(lines)
    )
    if header is None:
        raise ValueError("Malformed header for bands.x output")

    n_bands = int(header.group("nbnd"))
    n_kpt = int(header.group("nk"))

    # Parse output into a stream of numbers then group into bunches
    #   Shape: (kpt[3], bands[nbnd])
    number_stream = chain.from_iterable((parse_vector(line) for line in lines))
    kpt_bunches = map(lambda a: (a[:3], a[3:]), group(number_stream, n_bands + 3))
    kpts, bands = map(np.array, zip(*kpt_bunches))

    # Verify dimensions
    if kpts.shape != (n_kpt, 3):
        raise RuntimeError(f"Incorrect k-point array shape {kpts.shape!r}")
    if bands.shape != (n_kpt, n_bands):
        raise RuntimeError(f"Incorrect band array shape {bands.shape!r}")

    return kpts, bands


def parse_pwx_out(lines: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract band structure from pw.x output.

    Args:
        lines: Lines from the pw.x output file

    Returns:
        kpts: Array of kpt positions. Shape: [nkpt, 3]
        bands: Array of band energies. Shape: [nkpt, nband]
    """
    kpt_line_re = re.compile(
        r"^ *k =((?: *-?[\d.]+){3}) *\( *\d+ PWs\) +bands \(ev\):$"
    )
    bands_re = re.compile(r"^[ \t]+(?:-?[.\d]+[ \t]+)+$")

    def _parse_bands(_line: str) -> Iterable[float]:
        return map(float, _line.split())

    kpt_buffer = []  # type: List[Tuple[float, ...]]
    kpt_bands = []  # type: List[float]
    bands_buffer = []  # type: List[List[float]]

    for line in lines:
        # Clear buffer for new kpt
        if match := kpt_line_re.match(line):
            this_kpt = tuple(float(x) for x in match.group(1).split())
            if len(this_kpt) != 3:
                raise RuntimeError(f"Malformed k-point {match.group(1)}")
            kpt_buffer.append(this_kpt)
            if len(kpt_bands) > 0:
                bands_buffer.append(kpt_bands)
                kpt_bands = []

        # Todo: This does not work if filling block is also present
        if bands_re.match(line):
            kpt_bands.extend(_parse_bands(line))

    # Clear remaining buffers
    bands_buffer.append(kpt_bands)

    # Verify dimensions
    if len(kpt_buffer) != len(bands_buffer):
        raise RuntimeError("Unequal number of k-points and band blocks.")
    n_bands = len(bands_buffer[0])
    if any(len(bnd) != n_bands for bnd in bands_buffer):
        raise RuntimeError("Unequal number of bands")

    return np.array(kpt_buffer), np.array(bands_buffer)


def kpath_coord(kpt_list: np.ndarray) -> np.ndarray:
    """Calculate displacement along a continuous path coordinate."""
    deltas = np.linalg.norm(kpt_list[1:] - kpt_list[:-1], axis=1)
    path_coord = np.zeros(deltas.shape[0] + 1)
    path_coord[1:] = np.cumsum(deltas)
    # Normalize path length to one
    path_coord /= path_coord[-1]
    return path_coord


def save_bands(
    k_path: np.ndarray,
    k_points: np.ndarray,
    bands: np.ndarray,
    csv_path: Optional[Path] = None,
    npz_path: Optional[Path] = None,
) -> None:
    """Save band data in csv or npz format.

    Args:
        k_path: Displacement on a path coordinate. Shape: [nkpt]
        k_points: K-point positions. Shape: [nkpt, 3]
        bands: Band energies. Shape: [nkpt, nbnd]
        csv_path: Write csv output here if not None
        npz_path: Write npz output here if not None
    """
    if npz_path is not None:
        np.savez(npz_path, kpath=k_path, kpoints=k_points, bands=bands)

    if csv_path is not None:
        # pylint: disable=import-outside-toplevel
        import csv
        from itertools import chain

        with open(csv_path, "w") as f_csv:
            writer = csv.writer(f_csv)
            n_bands = bands.shape[1]
            writer.writerow(
                ["kpath", "kpoint", None, None, "bands"] + [None] * (n_bands - 1)
            )
            writer.writerows(
                map(lambda x: chain(*x), zip(k_path.reshape(-1, 1), k_points, bands))
            )


def parse_args_bands(cli_args: Sequence[str]) -> Namespace:
    """Parse CLI args for bands submodule."""
    parser = ArgumentParser(
        prog="pwproc bands", description="Parse output from bands.x and pw.x"
    )

    parser.add_argument(
        "in_file", action="store", type=Path, help="pw.x or bands.x output file"
    )
    in_grp = parser.add_mutually_exclusive_group()
    in_grp.add_argument(
        "--bands",
        action="store_const",
        const="bands",
        dest="in_type",
        help="Parse output from bands.x (default)",
    )
    in_grp.add_argument(
        "--pwx",
        action="store_const",
        const="pwx",
        dest="in_type",
        help="Parse output is from pw.x",
    )
    out_grp = parser.add_argument_group(
        title="Output",
        description=(
            "Output files record k-point coordinates, band energies, and progress on a"
            " continuous path coordinate"
        ),
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

    parsed_args = parser.parse_args(cli_args)

    # Apply default arguments
    if parsed_args.in_type is None:
        parsed_args.in_type = "bands"

    def _gen_out_path(out_path: Optional[Path], target_ext: str) -> Optional[Path]:
        if out_path is not None and out_path.is_dir():
            # If out_path is a dir, generate a sensible out file name
            out_file_path = out_path.joinpath(parsed_args.in_file.name)
            # Modify the output extension
            removable_ext = (".dat", ".out")
            if out_file_path.suffix in removable_ext:
                return out_file_path.with_suffix(target_ext)
            else:
                return out_file_path.with_name(out_path.name + target_ext)
        else:
            return out_path

    parsed_args.npz = _gen_out_path(parsed_args.npz, ".npz")
    parsed_args.csv = _gen_out_path(parsed_args.csv, ".csv")

    return parsed_args


def run_bands(cli_args: Namespace) -> None:
    """Main entry point for `bands` submodule."""
    with open(cli_args.in_file, "r") as f_in:
        if cli_args.in_type == "pwx":
            kpt, bands = parse_pwx_out(f_in)
        elif cli_args.in_type == "bands":
            kpt, bands = parse_bands_out(f_in)
        else:
            raise RuntimeError

    k_path = kpath_coord(kpt)
    save_bands(k_path, kpt, bands, npz_path=cli_args.npz, csv_path=cli_args.csv)


if __name__ == "__main__":
    import sys

    args = parse_args_bands(sys.argv[1:])
    run_bands(args)
