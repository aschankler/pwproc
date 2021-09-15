"""Read/write for Quantum ESPRESSO input files."""

import re
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import scipy.constants  # type: ignore[import]

from pwproc.geometry import Basis, Species, Tau
from pwproc.util import LookaheadIter

# CellDM as defined by `pw.x`
CellDM = NewType("CellDM", Tuple[float, float, float, float, float, float])


class NamelistData:
    """Parse lines in a pw.x namelist."""
    _kind_re: ClassVar = re.compile(r"^&([a-zA-Z]*)[\s]*$")
    _field_re: ClassVar = re.compile(r"^[ \t]*([()\w]*)[ \t]*=[ \t]*(.*)[\w]*$")
    _namelist_kinds: ClassVar = frozenset(("CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL"))

    def __init__(self, lines):
        # type: (Iterable[str]) -> None
        self.lines = tuple(lines)
        self.kind = self._get_kind(self.lines)
        self.fields = self._get_fields(self.lines)

    @classmethod
    def _get_kind(cls, lines):
        # type: (Sequence[str]) -> str
        m = cls._kind_re.match(lines[0])
        if m is not None:
            kind = m.group(1).upper()
            if kind in cls._namelist_kinds:
                return kind
        raise ValueError("Namelist kind " + lines[0])

    @classmethod
    def _get_fields(cls, lines):
        # type: (Sequence[str]) -> Dict[str, str]
        fields = {}
        for line in lines[1:-1]:
            # Skip comments and blanks
            if line.strip() == '' or line.strip()[0] == '!':
                continue
            m = cls._field_re.match(line)
            if m is not None:
                fields[m.group(1)] = m.group(2)
            else:
                raise ValueError("Error in namelist:\n'{}'".format(line.strip()))
        return fields


class CardData:
    """Parse lines in a pw.x input card."""
    _kind_re: ClassVar = re.compile(r"^([\w]+)[ \t]*([\w]+)?[ \t]*$")
    _card_kinds: ClassVar = frozenset({"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS",
                                       "CELL_PARAMETERS", "OCCUPATIONS", "CONSTRAINTS", "ATOMIC_FORCES"})
    _card_units: ClassVar = {'CELL_PARAMETERS': frozenset({'alat', 'bohr', 'angstrom'}),
                             'ATOMIC_POSITIONS': frozenset({'alat', 'bohr', 'crystal', 'angstrom'}),
                             'K_POINTS': frozenset({'automatic', 'gamma', 'crystal', 'tpiba',
                                                    'crystal_b', 'tpiba_b'})}

    def __init__(self, lines):
        # type: (Iterable[str]) -> None
        self.lines = tuple(lines)
        self.kind, self.unit = self._get_kind(self.lines[0])
        self.body = self.lines[1:]

    @classmethod
    def _get_kind(cls, line):
        # type: (str) -> Tuple[str, str]
        m = cls._kind_re.match(line)
        if m is None:
            raise ValueError("Card: " + line)

        kind = m.group(1).strip().upper()
        unit = m.group(2)
        if kind not in cls._card_kinds:
            raise ValueError("Card: " + kind)

        if kind in cls._card_units:
            unit = unit.strip().lower()
            if unit not in cls._card_units[kind]:
                raise ValueError("Card: " + line)
        elif unit is not None:
            raise ValueError("Card: " + line)

        return kind, unit


def _match_namelist(lines):
    # type: (LookaheadIter[str]) -> Union[None, NamelistData]
    """Consume and parse a single namelist from input."""
    namelist_re = r"^&[a-zA-Z]*[\s]*$"
    nl_lines = []
    buff = []

    # Ignore blanks and comments
    line = next(lines)
    while line.strip() == '' or line.strip()[0] == '!':
        buff.append(line)
        line = next(lines)

    # Match start of namelist or abort
    m = re.match(namelist_re, line)
    if m is None:
        buff.append(line)
        for line in reversed(buff):
            lines.push(line)
        return

    while line.strip() != '/':
        nl_lines.append(line)
        line = next(lines)

    nl_lines.append(line)
    return NamelistData(nl_lines)


def _match_card(lines):
    # type: (LookaheadIter[str]) -> Union[None, CardData]
    def gather_to_header():
        # type: () -> Tuple[str, List[str]]
        header_re = r"(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS" \
                    r"|CONSTRAINTS|OCCUPATIONS|ATOMIC_FORCES)"

        line_buffer = []
        while True:
            try:
                line = next(lines)
            except StopIteration:
                header = None
                break

            if re.match(header_re, line):
                header = line
                break
            else:
                line_buffer.append(line)

        return header, line_buffer

    # Consume lines until a card header is reached
    h_this, _ = gather_to_header()

    if h_this is None:
        return

    # Gather the rest of the header until the next header or EOF
    h_next, card_lines = gather_to_header()
    card_lines = [h_this] + card_lines

    if h_next:
        lines.push(h_next)

    return CardData(card_lines)


def read_pwi(lines):
    # type: (Iterable[str]) -> Tuple[List[NamelistData], List[CardData]]

    lines = LookaheadIter(lines)
    namelists = []
    cards = []

    nl = _match_namelist(lines)
    while nl is not None:
        namelists.append(nl)
        nl = _match_namelist(lines)

    ca = _match_card(lines)
    while ca is not None:
        cards.append(ca)
        ca = _match_card(lines)

    return namelists, cards


def parse_pwi_cell(cell_card):
    # type: (CardData) -> Basis
    """Parse the cell basis."""
    from pwproc.util import parse_vector
    assert(cell_card.kind == 'CELL_PARAMETERS' and cell_card.unit == 'angstrom')
    basis = [parse_vector(el) for el in cell_card.body if el.strip() != '']
    return Basis(np.array(basis))


IfPos = Sequence[Union[None, Tuple[bool, bool, bool]]]


def parse_pwi_atoms(atom_card):
    # type: (CardData) -> Tuple[Species, Tau, Union[IfPos, None]]
    """Parse atomic positions."""
    assert(atom_card.kind == 'ATOMIC_POSITIONS')
    species = []
    tau = []
    if_pos = []

    for line in atom_card.body:
        if line.strip() == '':
            continue
        s, *rest = line.split()
        species.append(s)
        assert(len(rest) == 3 or len(rest) == 6)
        tau.append(tuple(map(float, rest[:3])))
        if len(rest) == 6:
            if_pos.append(tuple(map(lambda b: True if b == '1' else False, rest[3:])))

    species = tuple(species)
    tau = np.array(tau)

    if all(ip is None for ip in if_pos):
        if_pos = None

    return Species(species), Tau(tau), if_pos


def gen_pwi_cell(basis):
    # type: (Basis) -> Iterator[str]
    # pylint: disable=import-outside-toplevel
    from pwproc.geometry.cell import format_basis
    yield "CELL_PARAMETERS {}\n".format("angstrom")
    yield from format_basis(basis)
    yield "\n\n"


def gen_pwi_atoms(species, pos, coord_type, if_pos=None):
    # type: (Species, Tau, str, Optional[IfPos]) -> Iterator[str]
    from itertools import starmap

    # pylint: disable=import-outside-toplevel
    from pwproc.geometry.cell import format_positions
    yield "ATOMIC_POSITIONS {}\n".format(coord_type)
    if if_pos is None:
        yield from format_positions(species, pos)
    else:
        # Optionally add the position freeze
        assert(len(if_pos) == len(species))

        def if_string(ifp):
            # type: (Optional[Tuple[bool, bool, bool]]) -> str
            if ifp:
                return "   {}  {}  {}".format(*map(lambda b: int(b), ifp))
            else:
                return ""

        pos_lines = format_positions(species, pos)
        yield from starmap(
            lambda p, ifp: p.rstrip() + if_string(ifp) + "\n", zip(pos_lines, if_pos)
        )


def gen_pwi(basis: Basis, species: Species, pos: Tau, coord_type: str,
            write_cell: bool = True, write_pos: bool = True,
            if_pos: Optional[IfPos] = None
            ) -> Iterator[str]:
    """Generate the `CELL_PARAMETERS` and `ATOMIC_POSITIONS` cards.

    Basis is assumed to be in angstroms and tau should agree with `coord_type`
    """

    # Yield basis
    if write_cell:
        yield from gen_pwi_cell(basis)

    # Yield atomic positions
    if write_pos:
        yield from gen_pwi_atoms(species, pos, coord_type, if_pos)


def cell_dimensions(basis):
    # type: (Basis) -> CellDM
    """Return celldm as used by QuantumEspresso."""
    vec1 = basis[0]
    vec2 = basis[1]
    vec3 = basis[2]

    len_1 = np.linalg.norm(vec1)
    len_2 = np.linalg.norm(vec2)
    len_3 = np.linalg.norm(vec3)

    cdm = [0.0 for _ in range(6)]

    # Convert angstrom -> bohr
    cdm[0] = len_1 * scipy.constants.angstrom / scipy.constants.value("Bohr radius")

    cdm[1] = len_2 / len_1
    cdm[2] = len_3 / len_1

    cdm[3] = np.abs(np.dot(vec2, vec3)) / (len_2 * len_3)
    cdm[4] = np.abs(np.dot(vec1, vec3)) / (len_1 * len_3)
    cdm[5] = np.abs(np.dot(vec1, vec2)) / (len_1 * len_2)

    return CellDM(tuple(cdm))


def get_ibrav(celldm):
    # type: (CellDM) -> int
    """Returns the Bravis lattice type as defined by pw.x"""
    # pylint: disable=too-many-return-statements
    def close(_a, _b):
        # type: (float, float) -> bool
        return bool(np.isclose(_a, _b))

    if any(not close(c, 0.0) for c in celldm[3:]):
        orthogonal = [close(c, 0.0) for c in celldm[3:]]
        if orthogonal.count(True) < 2:
            # Triclinic
            return 14
        if orthogonal.count(True) != 2:
            raise ValueError

        if close(celldm[5], 0.5) and close(celldm[1], 1.0):
            # Hexagonal
            return 4
        else:
            if not orthogonal[2]:
                # Monoclinic, c unique, ab not orthogonal
                return 12
            elif not orthogonal[1]:
                # Monoclinic, b unique, ac not orthogonal
                return -12
            else:
                raise ValueError("Invalid monoclinic cell; bc not orthogonal.")
    elif close(celldm[1], 1.0) and close(celldm[2], 1.0):
        # Cubic
        return 1
    elif close(celldm[1], 1.0):
        # Tetragonal
        return 6
    else:
        # Orthorhombic
        return 8


# Function type to define approximate equality
_CloseFn = Callable[[float, float], bool]


def _is_perp(vec1, vec2, close):
    # type: (np.ndarray, np.ndarray, _CloseFn) -> bool
    cos_theta = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    return close(cos_theta, 1.0)


def _is_orthogonal(vec1, vec2, close):
    # type: (np.ndarray, np.ndarray, _CloseFn) -> bool
    cos_theta = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    return close(cos_theta, 0.0)


def _check_orthorhombic(basis, close):
    # type: (Basis, _CloseFn) -> bool
    check = True
    check &= _is_perp(basis[0], np.array([1, 0, 0]), close)
    check &= _is_perp(basis[1], np.array([0, 1, 0]), close)
    check &= _is_perp(basis[2], np.array([0, 0, 1]), close)
    return check


def _check_tetragonal(basis, close):
    # type: (Basis, _CloseFn) -> bool
    check = True
    check &= _check_orthorhombic(basis, close)
    b_over_a = np.linalg.norm(basis[1]) / np.linalg.norm(basis[0])
    check &= close(b_over_a, 1.0)
    return check


def _check_cubic(basis, close):
    # type: (Basis, _CloseFn) -> bool
    b_over_a = np.linalg.norm(basis[1]) / np.linalg.norm(basis[0])
    c_over_a = np.linalg.norm(basis[2]) / np.linalg.norm(basis[0])
    check = True
    check &= _check_orthorhombic(basis, close)
    check &= close(b_over_a, 1.0)
    check &= close(c_over_a, 1.0)
    return check


def _check_hexagonal(basis, close):
    # type: (Basis, _CloseFn) -> bool
    check = True
    check &= _is_perp(basis[0], np.array([1, 0, 0]), close)
    check &= _is_perp(basis[1], np.array([-0.5, np.sqrt(3) / 2, 0]), close)
    check &= _is_perp(basis[2], np.array([0, 0, 1]), close)
    return check


def _check_monoclinic(basis, b_unique, close):
    # type: (Basis, bool, _CloseFn) -> bool
    check = True
    check &= _is_perp(basis[0], np.array([1, 0, 0]), close)
    if b_unique:
        check &= _is_perp(basis[1], np.array([0, 1, 0]), close)
        check &= _is_orthogonal(basis[2], np.array([0, 1, 0]), close)
    else:
        check &= _is_orthogonal(basis[1], np.array([0, 0, 1]), close)
        check &= _is_perp(basis[2], np.array([0, 0, 1]), close)
    return check


def _check_triclinic(basis, close):
    # type: (Basis, _CloseFn) -> bool
    check = True
    check &= _is_perp(basis[0], np.array([1, 0, 0]), close)
    check &= _is_orthogonal(basis[1], np.array([0, 0, 1]), close)
    return check


def check_canonical(basis, ibrav, close=None):
    # type: (Basis, int, Optional[_CloseFn]) -> bool
    """Verify that the basis aligns with the coordinate vectors."""

    def _close(_a, _b):
        # type: (float, float) -> bool
        return bool(np.isclose(_a, _b))

    if close is None:
        close = _close

    if ibrav == 1:
        return _check_cubic(basis, close)
    elif ibrav == 4:
        return _check_hexagonal(basis, close)
    elif ibrav == 6:
        return _check_tetragonal(basis, close)
    elif ibrav == 8:
        return _check_orthorhombic(basis, close)
    elif ibrav in (12, -12):
        b_unique = ibrav == -12
        return _check_monoclinic(basis, b_unique, close)
    elif ibrav == 14:
        return _check_triclinic(basis, close)
    elif ibrav in (2, 3, -3, 5, -5, 7, 9, -9, 91, 10, 11, 13, -13):
        raise NotImplementedError(f"ibrav = {ibrav}")
    else:
        raise ValueError(f"Bad ibrav = {ibrav}")
