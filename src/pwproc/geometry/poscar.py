"""
Read/write for POSCAR files.

This module writes to the ``pwproc.geometry.poscar`` logger
"""

import logging
from collections import Counter
from typing import Iterable, Iterator, List, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np

from pwproc.geometry.cell import Basis, MovableFlags, Species, Tau


class _PoscarHeader(NamedTuple):
    comment: str
    scale: float
    basis: Basis
    species: Mapping[str, int]
    selective_dynamics: bool


def _content_lines(lines):
    # type: (Iterable[str]) -> Iterator[str]
    """Only return non-blank lines of the input. Warn if blank  found."""
    logger = logging.getLogger()
    for line in lines:
        line = line.strip()
        if line != "":
            yield line
        else:
            logger.warning("Ignoring blank line found in POSCAR.")


def _read_poscar_header(poscar_lines: Iterable[str]) -> _PoscarHeader:
    # pylint: disable=import-outside-toplevel
    from pwproc.util import parse_vector

    logger = logging.getLogger()

    # Read data from input
    lines = _content_lines(poscar_lines)
    # Ignore the poscar comment
    comment = next(lines).strip()
    # TODO: when scale is negative, interpret as cell volume
    scale_factor = float(next(lines))

    # Parse the basis
    basis_lines = tuple(parse_vector(next(lines)) for _ in range(3))
    if not all(len(line) == 3 for line in basis_lines):
        raise ValueError("Incorrect basis dimensions")
    basis = Basis(scale_factor * np.array(basis_lines))

    # Parse the species names and counts
    species_names = next(lines).split()
    species_num = next(lines).split()
    if len(species_names) != len(species_num):
        raise ValueError("Lengths of atom type and atom count lines do not match")
    species = {name: int(count) for name, count in zip(species_names, species_num)}

    # Check for selective dynamics
    dynamics_flag = next(lines).strip().lower()
    if dynamics_flag[0] == "s":
        selective_dynamics = True
        # Behavior is defined by the first char, but expect 'selective dynamics'
        if len(dynamics_flag) > 1 and dynamics_flag != "selective dynamics":
            logger.warning(
                "Unexpected selective dynamics flag {!r}. Selective dynamics is enabled.",
                dynamics_flag,
            )
    else:
        selective_dynamics = False

    return _PoscarHeader(comment, scale_factor, basis, species, selective_dynamics)


def read_poscar_comment(poscar_lines: Iterable[str]) -> str:
    """Return the POSCAR comment string."""
    lines = _content_lines(poscar_lines)
    return next(lines).strip()


def read_poscar_scale(poscar_lines: Iterable[str]) -> float:
    """Return the POSCAR scaling factor."""
    lines = _content_lines(poscar_lines)
    # Scale is the second line
    next(lines)
    return float(next(lines))


def read_poscar(
    poscar_lines: Iterable[str], out_type: str = "angstrom"
) -> Tuple[Basis, Species, Tau]:
    """Read geometry data from POSCAR file.

    Args:
        poscar_lines: Lines of the poscar file
        out_type: Desired output coordinate type

    Returns:
        Basis in angstrom
        List of species names
        Position matrix in requested coordinates

    Raises:
        ValueError: when parsing malformed files
    """
    # pylint: disable=import-outside-toplevel
    from pwproc.geometry.cell import convert_positions

    logger = logging.getLogger()

    # Read and store the entire iterable. This is needed for once-through iterables
    # (for example open files)
    poscar_lines = list(poscar_lines)
    header = _read_poscar_header(poscar_lines)

    # Discard the header lines
    lines = _content_lines(poscar_lines)
    for _ in range(7):
        next(lines)
    if header.selective_dynamics:
        next(lines)

    # Format the species labels
    species = Species(tuple(Counter(header.species).elements()))

    # Read atomic position type
    coord_line = next(lines).strip().lower()
    if coord_line[0] == "d":
        in_type = "crystal"
        if len(coord_line) > 1 and coord_line != "direct":
            logger.warning(
                "Unexpected coordinate flag {!r}. Coordinates set to crystal.",
                coord_line,
            )
    elif coord_line[0] == "c" or coord_line[0] == "k":
        in_type = "angstrom"
        if len(coord_line) > 1 and coord_line != "cartesian":
            logger.warning(
                "Unexpected coordinate flag {!r}. Coordinates set to cartesian.",
                coord_line,
            )
    else:
        raise ValueError(f"Bad coordinate flag {coord_line!r}")

    # Parse atomic positions
    n_atoms = sum(header.species.values())
    pos_lines = []
    for i, line in enumerate(lines):
        if i > n_atoms:
            # Stop here if the file has a forces section
            break
        line_fields = line.split()
        if len(line_fields) != 3 and len(line_fields) != 6:
            raise ValueError(f"Incorrect position format {line!r}")
        # This ignores any selective dynamics flags
        pos_lines.append(tuple(float(x) for x in line_fields[:3]))
    if len(pos_lines) != n_atoms:
        raise ValueError("Incorrect number of position lines")

    if in_type == "crystal":
        positions = Tau(np.array(pos_lines))
    else:
        positions = Tau(header.scale * np.array(pos_lines))

    # Convert the input coordinates
    positions = convert_positions(positions, header.basis, in_type, out_type)

    return header.basis, species, positions


def read_poscar_selective_dynamics(poscar_lines: Iterable[str]) -> MovableFlags:
    """Read the selective dynamics flags."""
    # Read and store the entire iterable
    poscar_lines = list(poscar_lines)
    header = _read_poscar_header(poscar_lines)
    n_atoms = sum(header.species.values())

    if not header.selective_dynamics:
        return tuple(None for _ in range(n_atoms))

    # Discard up to the atomic positions
    lines = _content_lines(poscar_lines)
    for _ in range(9):
        next(lines)

    def parse_sd_flag(flag: str) -> bool:
        if flag == "T":
            return True
        elif flag == "F":
            return False
        else:
            raise ValueError(f"Bad selective dynamics flag {flag!r}")

    sd_flags: List[Union[None, Tuple[bool, bool, bool]]] = []

    # Read position lines and extract selective dynamics flags
    for i, line in enumerate(lines):
        if i > n_atoms:
            # Stop here if the file has a forces section
            break
        line_fields = line.split()
        if len(line_fields) != 3 and len(line_fields) != 6:
            raise ValueError(f"Incorrect position format {line!r}")
        if len(line_fields) == 3:
            # No SD flags on this line
            sd_flags.append(None)
        else:
            sd_flags.append(
                (
                    parse_sd_flag(line_fields[3]),
                    parse_sd_flag(line_fields[4]),
                    parse_sd_flag(line_fields[5]),
                )
            )

    if len(sd_flags) != n_atoms:
        print(sd_flags)
        raise ValueError("Incorrect number of position lines")

    return tuple(sd_flags)


def gen_poscar(
    basis: Basis,
    species: Species,
    positions: Tau,
    *,
    comment: Optional[str] = None,
    scale: float = 1.0,
    in_type: str = "angstrom",
    coordinate_type: str = "cartesian",
    selective_dynamics: MovableFlags = None,
) -> Iterator[str]:
    """Format the crystal structure in the POSCAR format.

    Args:
        basis: Lattice vectors in Angstrom
        species: Vector of atomic species, in the same order as `positions`
        positions: Vector of atomic positions
        comment: Single line comment written as the first line in the output
        scale: Uniform scaling factor applied to basis and positions
        in_type: Coordinate type used to supply the atomic positions
        coordinate_type: Coordinates used to write positions in output. One of
            "cartesian" or "direct"
        selective_dynamics: Mark atoms as fixed during relaxation

    Yields:
        Lines of the formatted POSCAR file
    """
    # pylint: disable=import-outside-toplevel
    from operator import itemgetter

    from pwproc.geometry.cell import convert_positions, format_basis
    from pwproc.geometry.format_util import (
        POSITION_PRECISION,
        as_fixed_precision,
        columns,
    )

    if len(species) != len(positions):
        raise ValueError("Lengths for species and positions do not match")
    if selective_dynamics is not None and len(species) != len(selective_dynamics):
        raise ValueError("Incorrect length for selective dynamics flags")
    comment = "POSCAR" if comment is None else comment.strip()
    if "\n" in comment:
        raise ValueError("Comment may not be multiline")

    # Write the header information
    yield comment + "\n"
    yield "{}\n".format(scale)
    yield from format_basis(basis / scale)

    # Count each unique species
    species_counts = Counter(species)
    s_names = sorted(species_counts)
    s_counts = tuple(species_counts[name] for name in s_names)

    # Write atomic species
    yield from columns((s_names, s_counts), min_space=1, left_pad=0)

    # Write selective dynamics flag
    if selective_dynamics is not None:
        yield "Selective dynamics\n"

    # Group the positions by species
    idx = tuple(i for i, _ in sorted(enumerate(species), key=itemgetter(1)))
    positions = positions[(idx,)]

    # Write atomic positions
    if coordinate_type == "cartesian":
        yield "Cartesian\n"
        positions = convert_positions(
            positions, basis, in_type=in_type, out_type="angstrom"
        )
        positions /= scale
    elif coordinate_type == "direct":
        yield "Direct\n"
        positions = convert_positions(
            positions, basis, in_type=in_type, out_type="crystal"
        )
    else:
        raise ValueError(f"Bad coordinate type {coordinate_type}")

    pos_lines = columns(
        positions,
        min_space=3,
        left_pad=0,
        convert_fn=lambda x: as_fixed_precision(x, POSITION_PRECISION),
    )

    if selective_dynamics is None:
        yield from pos_lines
    else:
        for pos, sd_flags in zip(pos_lines, selective_dynamics):
            if sd_flags is None:
                yield pos[:-1] + "\n"
            else:
                fmt_sd = " ".join("T" if x else "F" for x in sd_flags)
                yield pos[:-1] + " " + fmt_sd + "\n"
