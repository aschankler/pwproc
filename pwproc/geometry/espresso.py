"""Read/write for Quantum ESPRESSO input files."""

import re
import numpy as np
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from pwproc.geometry import Basis, Species, Tau
from pwproc.util import LookaheadIter


class NamelistData:
    """Parse lines in a pw.x namelist."""
    _kind_re: ClassVar = re.compile(r"^&([a-zA-Z]*)[\s]*$")
    _field_re: ClassVar = re.compile(r"^[ \t]*([\w]*)[ \t]*=[ \t]*(.*)[\w]*$")
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
                raise ValueError("Error in namelist:\n'{}'".format(line))
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
        unit = m.group(2).strip().lower()
        if kind not in cls._card_kinds:
            raise ValueError("Card: " + kind)

        if kind in cls._card_units:
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
    from pwproc.geometry.format_util import format_basis
    yield "CELL_PARAMETERS {}\n".format("angstrom")
    yield format_basis(basis)
    yield "\n\n"


def gen_pwi_atoms(species, pos, coord_type, if_pos=None):
    # type: (Species, Tau, str, Optional[IfPos]) -> Iterator[str]
    from itertools import starmap
    from pwproc.geometry.format_util import format_tau
    yield "ATOMIC_POSITIONS {}\n".format(coord_type)
    if if_pos is None:
        yield format_tau(species, pos)
    else:
        # Optionally add the position freeze
        assert(len(if_pos) == len(species))

        def if_string(ifp):
            # type: (Optional[Tuple[bool, bool, bool]]) -> str
            if ifp:
                return "   {}  {}  {}".format(*map(lambda b: int(b), ifp))
            else:
                return ""

        pos_lines = format_tau(species, pos).split('\n')
        yield from starmap(lambda p, ifp: p + if_string(ifp) + '\n', zip(pos_lines, if_pos))


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
