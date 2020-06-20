"""Read/write for Quantum ESPRESSO input files."""

import re
import numpy as np
from typing import Iterator, Optional, Sequence, Tuple, Union

Species = Tuple[str, ...]
Basis = np.ndarray      # Shape: (3, 3)
Tau = np.ndarray        # Shape: (natoms, 3)


class _InfileStack:
    """Wrapper for an iterator that supports push operations."""
    def __init__(self, lines):
        self.lines = iter(lines)
        self.top = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.top) > 0:
            return self.top.pop()
        else:
            return next(self.lines)

    def push(self, item):
        self.top.append(item)


class NamelistData:
    """Parse lines in a pw.x namelist."""
    def __init__(self, lines):
        self.lines = tuple(lines)
        self.kind = self._get_kind(lines)
        self.fields = self._get_fields(lines)

    @classmethod
    def _get_kind(cls, lines):
        kind_re = r"^&([a-zA-Z]*)[\s]*$"
        allowed_kinds = {"CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL"}
        m = re.match(kind_re, lines[0])
        if m is not None:
            kind = m.group(1).upper()
            if kind in allowed_kinds:
                return kind
        raise ValueError("Namelist kind " + lines[0])

    @classmethod
    def _get_fields(cls, lines):
        field_re = r"^[ \t]*([\w]*)[ \t]*=[ \t]*(.*)[\w]*$"
        fields = {}
        for l in lines[1:-1]:
            if l.strip() == '' or l.strip()[0] == '!':
                continue
            m = re.match(field_re, l)
            if m is not None:
                fields[m.group(1)] = m.group(2)
        return fields


class CardData:
    """Parse lines in a pw.x input card."""
    def __init__(self, lines):
        self.lines = tuple(lines)
        self.kind, self.unit = self._get_kind(lines[0])
        self.body = lines[1:]

    @classmethod
    def _get_kind(cls, line):
        kind_re = r"^([\w]+)[ \t]*([\w]+)?[ \t]*$"
        allowed_kinds = {"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS",
                         "CELL_PARAMETERS", "OCCUPATIONS", "CONSTRAINTS", "ATOMIC_FORCES"}
        allowed_units = {'CELL_PARAMETERS': {'alat', 'bohr', 'angstrom'},
                         'ATOMIC_POSITIONS': {'alat', 'bohr', 'crystal', 'angstrom'},
                         'K_POINTS': {'automatic'}}

        m = re.match(kind_re, line)
        if m is None:
            raise ValueError("Card: " + line)

        kind = m.group(1).strip().upper()
        unit = m.group(2)
        if kind not in allowed_kinds:
            raise ValueError("Card: " + kind)

        if kind in allowed_units:
            if unit not in allowed_units[kind]:
                raise ValueError("Card: " + line)
            kind = kind.strip()
        elif unit is not None:
            raise ValueError("Card: " + line)

        return kind, unit


def _match_namelist(lines):
    """Consume and parse a single namelist from input."""
    namelist_re = r"^&[a-zA-Z]*[\s]*$"
    nl_lines = []
    buff = []

    # Ignore blanks and comments
    l = next(lines)
    while l.strip() == '' or l.strip()[0] == '!':
        buff.append(l)
        l = next(lines)

    # Match start of namelist or abort
    m = re.match(namelist_re, l)
    if m is None:
        buff.append(l)
        for l in reversed(buff):
            lines.push(l)
        return

    while l.strip() != '/':
        nl_lines.append(l)
        l = next(lines)

    nl_lines.append(l)
    return NamelistData(nl_lines)


def _match_card(lines):
    def gather_to_header(lines):
        header_re = r"(ATOMIC_SPECIES|ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS" \
                    r"|CONSTRAINTS|OCCUPATIONS|ATOMIC_FORCES)"

        line_buffer = []
        while True:
            try:
                l = next(lines)
            except StopIteration:
                header = None
                break

            if re.match(header_re, l):
                header = l
                break
            else:
                line_buffer.append(l)

        return header, line_buffer

    # Consume lines until a card header is reached
    header, _ = gather_to_header(lines)

    if header is None:
        return

    # Gather the rest of the header until the next header or EOF
    h_next, card_lines = gather_to_header(lines)
    card_lines = [header] + card_lines

    if h_next:
        lines.push(h_next)

    return CardData(card_lines)


def read_pwi(lines):
    lines = _InfileStack(lines)
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
    """Parse the cell basis."""
    from pwproc.util import parse_vector
    assert(cell_card.kind == 'CELL_PARAMETERS' and cell_card.unit == 'angstrom')
    basis = [parse_vector(l) for l in cell_card.body if l.strip() != '']
    return np.array(basis)


def parse_pwi_atoms(atom_card):
    """Parse atomic positions."""
    from pwproc.util import convert_coords
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

    if all(ip is None for ip in if_pos):
        if_pos = None

    return species, np.array(tau), if_pos

def gen_pwi_cell(basis):
    # type: (Basis) -> Iterator[str]
    from pwproc.geometry.format_util import format_basis
    yield "CELL_PARAMETERS {}\n".format("angstrom")
    yield format_basis(basis)
    yield "\n\n"


def gen_pwi_atoms(species, pos, coord_type, if_pos=None):
    # type(Species, Tau, str, Optional[Sequence[Union[None, Sequence[bool]]]]) -> Iterator[str]
    from itertools import starmap
    from pwproc.geometry.format_util import format_tau
    yield "ATOMIC_POSITIONS {}\n".format(coord_type)
    if if_pos is None:
        yield format_tau(species, pos)
    else:
        # Optionally add the position freeze
        assert(len(if_pos) == len(species))

        def if_string(ifp):
            if ifp:
                return "   {}  {}  {}".format(*map(lambda b: int(b), ifp))
            else:
                return ""

        pos_lines = format_tau(species, pos).split('\n')
        yield from starmap(lambda p, ifp: p + if_string(ifp) + '\n', zip(pos_lines, if_pos))


def gen_pwi(basis: Basis, species: Species, pos: Tau, coord_type: str,
            write_cell: bool = True, write_pos: bool = True,
            if_pos: Optional[Sequence[Union[None, Sequence[bool]]]] = None
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

