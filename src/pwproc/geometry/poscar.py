"""
Read/write for POSCAR files.

This module writes to the ``pwproc.geometry.poscar`` logger
"""

import logging
from collections import Counter
from typing import Iterable, Iterator, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from pwproc.geometry.cell import Basis, Species, Tau


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
    species = tuple(Counter(header.species).elements())
    species = Species(species)

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
    for line in lines:
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


def read_poscar_selective_dynamics():
    # TODO: implement
    ...


def gen_poscar(
    basis: Basis,
    species: Species,
    positions: Tau,
    *,
    comment: Optional[str] = None,
    scale: float = 1.0,
    in_type: str = "angstrom",
    coordinate_type: str = "cartesian",
    # TODO: write selective dynamics
    # pylint disable-next=unused-argument
    selective_dynamics=None,
) -> Iterator[str]:
    # pylint: disable=import-outside-toplevel
    from pwproc.geometry.cell import convert_positions, format_basis
    from pwproc.geometry.format_util import (
        POSITION_PRECISION,
        as_fixed_precision,
        columns,
    )

    # Write the header information
    comment = "POSCAR" if comment is None else comment
    yield comment + "\n"
    yield "{}\n".format(scale)
    yield from format_basis(basis / scale)

    # Group the positions by species
    idx = tuple(i[0] for i in sorted(enumerate(species), key=lambda x: x[1]))
    positions = positions[(idx,)]

    # Count each unique species
    species_counts = Counter(species)

    # Write atomic species
    yield from columns(
        (species_counts, species_counts.values()), min_space=1, left_pad=0
    )

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

    yield from columns(
        positions,
        min_space=3,
        left_pad=0,
        convert_fn=lambda x: as_fixed_precision(x, POSITION_PRECISION),
    )
