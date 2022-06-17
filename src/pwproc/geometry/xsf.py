"""Read/Write for XSF file format."""

from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from pwproc.geometry import Basis, Species, Tau


def _get_next_line(lines: Iterator[str]) -> str:
    """Consume items from `lines` until a non-comment, non-blank line is reached."""
    while True:
        line = next(lines).strip()
        # Blank line
        if line == "":
            continue
        # Comment line
        if line[0] == "#":
            continue
        return line.strip()


def _read_primvec_header(header_line: str) -> Optional[int]:
    if not header_line.startswith("PRIMVEC"):
        raise ValueError("PRIMVEC not found")
    step = None
    if len(header_line.split()) > 1:
        step = int(header_line.split()[1])
    return step


def _read_primvec(lines: Iterator[str]) -> Tuple[Basis, Optional[int]]:
    # Parse the header
    step = _read_primvec_header(_get_next_line(lines))
    # Parse the content
    basis_lines = []
    for _ in range(3):
        lat_vec = tuple(float(x) for x in _get_next_line(lines).split())
        assert len(lat_vec) == 3
        basis_lines.append(lat_vec)
    basis = Basis(np.array(basis_lines))
    return basis, step


def _read_primcoord_header(header_line: str) -> Optional[int]:
    if not header_line.startswith("PRIMCOORD"):
        raise ValueError("PRIMCOORD not found")
    step = None
    if len(header_line.split()) > 1:
        step = int(header_line.split()[1])
    return step


def _read_primcoord(lines: Iterator[str]) -> Tuple[Species, Tau, Optional[int]]:
    # Parse the header
    step = _read_primcoord_header(_get_next_line(lines))
    _nat, _flag = _get_next_line(lines).split()
    assert _flag == "1"
    nat = int(_nat)

    # Parse coordinates and species
    species = []
    positions = []
    for _ in range(nat):
        line = _get_next_line(lines).split()
        species.append(line[0])
        positions.append(tuple(float(x) for x in line[1:4]))

    return Species(tuple(species)), Tau(np.array(positions)), step


def _read_xsf_single(lines: Iterable[str]) -> Tuple[Basis, Species, Tau]:
    """Read a single-structure XSF file."""
    lines = iter(lines)

    basis, step = _read_primvec(lines)
    assert step is None
    species, tau, step = _read_primcoord(lines)
    assert step is None

    return basis, species, tau


def _read_xsf_animate(
    lines: Iterator[str], n_steps: int
) -> Tuple[Union[Basis, Sequence[Basis]], Species, Sequence[Tau]]:
    # Read the first step
    step_basis, step = _read_primvec(lines)
    animate_cell = step is not None
    species, step_pos, step = _read_primcoord(lines)
    assert step == 1

    # Initialize accumulators
    if animate_cell:
        basis = [step_basis]
    else:
        basis = step_basis
    positions = [step_pos]

    # Read the remaining steps
    for i in range(2, n_steps + 1):
        if animate_cell:
            step_basis, step = _read_primvec(lines)
            assert step == i
            basis.append(step_basis)
        step_spec, step_pos, step = _read_primcoord(lines)
        assert step == i
        assert species == step_spec
        positions.append(step_pos)

    return basis, species, positions


def read_xsf(
    lines: Iterable[str], *, axsf_allowed: bool = True, axsf_expected: bool = False
) -> Tuple[Union[Basis, Sequence[Basis]], Species, Union[Tau, Sequence[Tau]]]:
    lines = iter(lines)

    # Read the first two header lines
    header_line = _get_next_line(lines)
    if header_line.startswith("ANIMSTEPS"):
        if not axsf_allowed:
            raise RuntimeError(".axsf files not allowed")
        # Reading animated file
        n_steps = int(header_line.split()[1])
        header_line = _get_next_line(lines)
        assert header_line.startswith("CRYSTAL") or header_line.startswith("POLYMER")
        return _read_xsf_animate(lines, n_steps)
    else:
        if axsf_expected:
            raise RuntimeError(".axsf file expected")
        # Reading single-structure xsf
        assert header_line.startswith("CRYSTAL") or header_line.startswith("POLYMER")
        return _read_xsf_single(lines)


def read_axsf(
    lines: Iterable[str],
) -> Tuple[Union[Basis, Sequence[Basis]], Species, Sequence[Tau]]:
    """Read a *.axsf file with a sequence of structures."""
    return read_xsf(lines, axsf_expected=True)


def gen_xsf(basis, species, tau, write_header=True, step=None):
    # type: (Basis, Species, Tau, bool, Optional[int]) -> Iterator[str]
    # pylint: disable=import-outside-toplevel
    from pwproc.geometry.cell import format_basis, format_positions

    nat = len(species)

    if write_header:
        yield "CRYSTAL\n"

    step_str = f" {step:d}" if step is not None else ""

    yield f"PRIMVEC{step_str}\n"
    yield from format_basis(basis)
    yield f"PRIMCOORD{step_str}\n"
    yield "{} 1\n".format(nat)
    yield from format_positions(species, tau)
    yield "\n"


def gen_xsf_animate(basis, species, tau):
    # type: (Sequence[Basis], Species, Sequence[Tau]) -> Iterator[str]
    from itertools import chain

    nsteps = len(basis)
    yield "ANIMSTEPS {}\n".format(nsteps)
    yield "CRYSTAL\n"

    yield from chain(
        *(
            gen_xsf(basis[i], species, tau[i], write_header=False, step=(i + 1))
            for i in range(nsteps)
        )
    )
