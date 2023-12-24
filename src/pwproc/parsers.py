"""Parsers for pw.x output."""

import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Match,
    MutableMapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from pwproc.geometry import Basis, GeometryData, RelaxData, Species, Tau
from pwproc.util import LookaheadIter, parse_vector


class ParserError(RuntimeError): pass


T = TypeVar('T')
# pos_type, basis, species, tau
RawGeometry = Tuple[str, Sequence[Basis], Species, Sequence[Tau]]


def get_save_file(path):
    # type: (Path) -> str
    """Extract the prefix from pw.x output."""
    from pwproc.util import parser_one_line

    save_re = re.compile(
        r"^[ \t]+(?:Writing all to output data dir|Writing output data file"
        "|Writing config-only to output data dir) "
        r"(?:\./)?([-.\w]+)\.save/?"
    )
    save_parser = parser_one_line(save_re, lambda m: m.group(1))

    with open(path) as f:
        prefix = save_parser(f)

    if prefix is None:
        raise ParserError("Could not find calculation prefix")

    return prefix


def get_init_basis(path):
    # type: (Path) -> Tuple[float, Basis]
    """Extracts the initial basis in angstrom from pw.x output."""
    from scipy import constants

    from pwproc.util import parser_one_line, parser_with_header

    bohr_to_ang = constants.value('Bohr radius') / constants.angstrom
    alat_re = re.compile(r"[ \t]+lattice parameter \(alat\)[ \t]+=[ \t]+([\d.]+)[ \t]+a\.u\.")
    basis_head_re = re.compile(r"[ \t]+crystal axes: \(cart. coord. in units of alat\)")
    basis_line_re = re.compile(r"[ \t]+a\([\d]\) = \(((?:[ \t]+[-.\d]+){3}[ \t]+)\)")

    alat_parser = parser_one_line(alat_re, lambda m: float(m.group(1)))
    basis_parser = parser_with_header(basis_head_re, basis_line_re, lambda m: parse_vector(m.group(1)))

    with open(path) as f:
        alat: float = alat_parser(f)
        # TODO: Remove seek and run parsers in correct order
        f.seek(0)
        basis = basis_parser(f)

    # Convert basis from alat to angstrom
    assert len(basis) == 3
    basis = Basis(np.array(basis))
    basis *= alat * bohr_to_ang

    return alat, basis


def get_init_coord(path):
    # type: (Path) -> Tuple[str, Species, Tau]
    """Extracts starting atomic positions."""
    from pwproc.util import parser_with_header

    header_re = re.compile(r"[ \t]+site n\.[ \t]+atom[ \t]+positions \((cryst\. coord\.|alat units)\)")
    line_re = re.compile(r"[ \t]+[\d]+[ \t]+([\w]{1,3})[ \t]+tau\([ \d\t]+\) = \(((?:[ \t]+[-.\d]+){3}[ \t]+)\)")

    # Translate the tags in the output header to coordinate types
    coord_types = {"cryst. coord.": 'crystal', "alat units": 'alat'}
    # Precedence for coord types when multiple are present
    ctype_order = ('crystal', 'alat')

    coord_parser = parser_with_header(header_re, line_re, lambda m: m.groups(),
                                      header_proc=lambda m: m.group(1), find_multiple=True)

    with open(path) as f:
        init_coords = {coord_types[c_tag]: coords for c_tag, coords in coord_parser(f)}

    for ct in ctype_order:
        try:
            atom_coords = init_coords[ct]
        except KeyError:
            pass
        else:
            coord_type = ct
            break
    else:
        raise ParserError("Initial coordinates not found.")

    spec, pos = zip(*atom_coords)
    pos = np.array(tuple(map(parse_vector, pos)))

    return coord_type, Species(spec), Tau(pos)


def _count_relax_steps(path):
    # type: (Path) -> Tuple[int, int, int, bool]
    """Count the number of completed steps."""
    scf_re = re.compile(r"^ +number of scf cycles += +(?P<scf>[\d]+)$")
    bfgs_re = re.compile(r"^ +number of bfgs steps += +(?P<bfgs>[\d]+)$")
    last_step_re = re.compile(r"^ +bfgs converged in +(?P<scf>[\d]+) scf cycles and +(?P<bfgs>[\d]+) bfgs steps$")
    zero_mag_re = re.compile(r"^ +lsda relaxation :  a final configuration with zero *$")

    steps = []
    last_step = None
    zero_mag_relax = False

    with open(path) as f:
        lines = iter(f)
        for line in lines:
            m1 = scf_re.match(line)
            if m1 is not None:
                n_scf = int(m1.group('scf'))
                m2 = bfgs_re.match(next(lines))
                if m2 is None:
                    raise ParserError("Malformed step count")
                n_bfgs = int(m2.group('bfgs'))
                steps.append((n_scf, n_bfgs))

            m3 = last_step_re.match(line)
            if m3 is not None:
                last_step = (int(m3.group('scf')), int(m3.group('bfgs')))
                break

            m4 = zero_mag_re.match(line)
            if m4 is not None:
                if zero_mag_relax:
                    raise ParserError("Two zero-magnetization relaxations")
                zero_mag_relax = True

    if last_step is not None:
        steps.append(last_step)

    return len(steps), steps[0][0], steps[-1][0], zero_mag_relax


class ParserBase(Generic[T]):
    """Base class for local parsers."""
    header_re: Pattern

    def __init__(self):
        # type: () -> None
        self._buffer: List[T] = []

    def __call__(self, lines):
        # type: (LookaheadIter[str]) -> bool
        line = lines.top()
        match = self.header_re.match(line)
        if match:
            # Consume the matched line
            next(lines)
            self.complete_match(match, lines)
            return True
        return False

    @property
    def buffer(self):
        # type: () -> List[T]
        return self._buffer

    def complete_match(self, match, lines):
        # type: (Match, LookaheadIter[str]) -> None
        raise NotImplementedError


class EnergyParser(ParserBase[float]):
    header_re = re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry")

    def complete_match(self, match, _):
        # type: (Match, LookaheadIter[str]) -> None
        self.buffer.append(float(match.group(1)))


class FEnergyParser(ParserBase[Tuple[str, float]]):
    header_re = re.compile(r"[ \t]+Final (energy|enthalpy)[ \t]+=[ \t]+(-[.\d]+) Ry")

    def complete_match(self, match, _):
        # type: (Match, LookaheadIter[str]) -> None
        self.buffer.append((match.group(1), float(match.group(2))))


class GeometryParser(ParserBase[Tuple[str, Species, Tau]]):
    header_re = re.compile(r"ATOMIC_POSITIONS \((angstrom|crystal|alat|bohr)\)")
    atom_re = re.compile(r"([a-zA-Z]{1,2})((?:[\s]+[-\d.]+){3})")

    def complete_match(self, match, lines):
        # type: (Match, LookaheadIter[str]) -> None
        pos_type = match.group(1)
        species = []
        tau = []
        m = self.atom_re.match(lines.top())
        while m:
            lines.pop()
            s, pos = m.groups()
            species.append(s)
            tau.append(parse_vector(pos))
            m = self.atom_re.match(lines.top())
        species = tuple(species)
        tau = np.array(tau)
        self.buffer.append((pos_type, Species(species), Tau(tau)))


class BasisParser(ParserBase[Basis]):
    """Capture basis converted to angstrom."""

    header_re = re.compile(
        r"CELL_PARAMETERS (?:\()?(?P<coord>angstrom|bohr|alat= +[.\d]+)(?(1)\))"
    )
    basis_row_re = re.compile(r"(?:[\s]+-?[\d.]+){3}")

    def complete_match(self, match, lines):
        # type: (Match, LookaheadIter[str]) -> None
        coord_type = match.group("coord")
        basis_tmp = []
        while self.basis_row_re.match(lines.top()):
            line = lines.pop()
            basis_tmp.append(parse_vector(line))
        assert len(basis_tmp) == 3
        basis = Basis(np.array(basis_tmp))
        if coord_type != "angstrom":
            # pylint: disable=import-outside-toplevel
            from pwproc.geometry import convert_basis

            basis = convert_basis(basis, coord_type.strip(), "angstrom")
        self.buffer.append(basis)


class ForceParser(ParserBase[Tuple[float, float]]):
    header_re = re.compile(r"^ *Total force = +([.\d]+) +Total SCF correction = +([.\d]+) *$")

    def complete_match(self, match, _):
        # type: (Match, LookaheadIter[str]) -> None
        self.buffer.append((float(match.group(1)), float(match.group(2))))


class PressParser(ParserBase[Tuple[float, np.array, np.array]]):
    header_re = re.compile(r"^ *total {3}stress .* \(kbar\) +P= +(-?[.\d]+) *$")
    press_row_re = re.compile(r"^(?: +-?[.\d]+){6} *$")

    def complete_match(self, match, lines):
        # type: (Match, LookaheadIter[str]) -> None
        tot_p = float(match.group(1))
        press_tmp = []

        line = next(lines)
        while self.press_row_re.match(line):
            press_tmp.append(parse_vector(line))
            line = next(lines)
        assert(len(press_tmp) == 3)

        press_tmp = np.array(press_tmp)
        press_au = press_tmp[:, :3]
        press_bar = press_tmp[:, 3:]
        self.buffer.append((tot_p, press_au, press_bar))


class MagParser(ParserBase[Tuple[float, float]]):
    header_re = re.compile(r"^ *total magnetization += +(-?[.\d]+) +Bohr mag/cell *$")
    abs_mag_re = re.compile(r"^ * absolute magnetization += +(-?[.\d]+) +Bohr mag/cell *$")
    conv_re = re.compile(r"^ +convergence has been achieved in +[\d]+ iterations *$")

    def complete_match(self, match, lines):
        # type: (Match, LookaheadIter[str]) -> None
        m_tot = float(match.group(1))
        line = next(lines)
        m = self.abs_mag_re.match(line)
        assert(m is not None)
        m_abs = float(m.group(1))
        assert(next(lines).strip() == '')
        line = next(lines)
        m_conv = self.conv_re.match(line)
        if m_conv:
            self.buffer.append((m_tot, m_abs))


class FermiParser(ParserBase[float]):
    header_re = re.compile(r"[ \t]+the Fermi energy is[ \t]+(-?[.\d]+) ev")

    def complete_match(self, match, _):
        # type: (Match, LookaheadIter[str]) -> None
        self.buffer.append(float(match.group(1)))


def _run_relax_parsers(path, parsers):
    # type: (Path, Mapping[str, ParserBase]) -> Mapping[str, List[Any]]
    """Run arbitrary parsers on the pw.x output."""

    # Iterate through file
    with open(path) as f:
        lines = LookaheadIter(f)
        while True:
            try:
                parser_matched = False
                for parser in parsers.values():
                    parser_matched &= parser(lines)
                if not parser_matched:
                    next(lines)
            except StopIteration:
                break

    return {tag: parser.buffer for tag, parser in parsers.items()}


_base_tags = frozenset(('energy', 'fenergy', 'geom', 'basis'))
_all_tags = frozenset(['energy', 'fenergy', 'geom', 'basis', 'force', 'press', 'mag', 'fermi'])
_parser_map = {'energy': EnergyParser, 'fenergy': FEnergyParser, 'geom': GeometryParser,
               'basis': BasisParser, 'force': ForceParser, 'press': PressParser,
               'mag': MagParser, 'fermi': FermiParser}


# energy, final_en, relax_kind, geometry, data_buffers, zmag_relax
_RawParsed = Tuple[Sequence[float], Optional[float], Optional[str], RawGeometry, Dict[str, Sequence], bool]
# n_steps, _relax_kind, _relax_done, _zmag_relax
_RelaxDims = Tuple[int, Optional[str], bool, bool]


def _proc_relax_data(buffers, n_steps, zmag_relax):
    # type: (Mapping[str, Sequence[Any]], int, bool) -> _RawParsed
    tags = set(buffers)
    assert(_base_tags <= tags)
    assert(tags <= _all_tags)

    # Deal with energies first
    energy: Sequence[float] = buffers['energy']

    final_en: Optional[float] = None
    relax_kind: Optional[str] = None
    assert(len(buffers['fenergy']) < 2)
    if len(buffers['fenergy']) == 1:
        final_type, final_en = buffers['fenergy'][0]

        # If vc-relax, the true final energy is run in a new scf calculation,
        # which is captured in the `energy` buffer. The `fenergy` buffer has
        # a duplicate of last relaxation SCF step, which is discarded.

        # A zero-magnetization check occurs before the BFGS completes, so it
        # is possible to have an extra entry even if the final energy parsers
        # are not triggered. Thus, we strip this entry after the final energy
        # routine but before the buffer length check.
        if final_type == 'enthalpy':
            relax_kind = 'vcrelax'
            expected_len = n_steps + 1 if not zmag_relax else n_steps + 2
            if len(energy) == expected_len:
                final_en = energy[-1]
                energy = energy[:-1]
            else:
                # In this case, the final SCF step was interrupted
                final_en = None
        elif final_type == 'energy':
            relax_kind = 'relax'
        else:
            raise ParserError("Unknown final energy type")

    if zmag_relax:
        if len(energy) == n_steps + 1:
            energy = energy[:-1]
        else:
            # The magnetization check was interrupted
            zmag_relax = False

    if len(energy) != n_steps:
        raise ParserError("Incorrect length in energy buffer")

    def all_equal(seq):
        # type: (List) -> bool
        return seq.count(seq[0]) == len(seq)

    # Re-package geometry
    pos_type, species, pos = zip(*buffers['geom'])
    assert(all_equal(pos_type) and all_equal(species))
    pos_type, species = pos_type[0], species[0]
    bases = buffers['basis']
    if len(bases) > 0:
        assert(len(bases) == len(pos))
    geometry: RawGeometry = (pos_type, bases, species, pos)

    # Save the other buffers
    data_buffers = {}
    for t in tags - _base_tags:
        data_buffers[t] = buffers[t]

    return energy, final_en, relax_kind, geometry, data_buffers, zmag_relax


def _get_relax_data(path, tags, n_steps, zmag_relax):
    # type: (Path, Union[Iterable[str], None], int, bool) -> _RawParsed
    if tags is None:
        tags = set()
    else:
        tags = set(tags)
    tags |= _base_tags
    parsers = {tag: _parser_map[tag]() for tag in tags}
    buffers = _run_relax_parsers(path, parsers)
    return _proc_relax_data(buffers, n_steps, zmag_relax)


def _proc_geom_buffs(geom_buff: Tuple[str, Sequence[Basis], Species, Sequence[Tau]],
                     geom_init: Tuple[str, float, Basis, Species, Tau],
                     target_coord: str,
                     relax_dims: _RelaxDims
                     ) -> Tuple[Sequence[Basis], Species, Sequence[Tau]]:
    from itertools import starmap

    from pwproc.geometry import convert_positions

    # Unpack geometry
    ctype_i, alat, basis_i, species_i, pos_i = geom_init
    ctype, basis_steps, species, pos = geom_buff

    assert(species_i == species)

    # Unpack relax dimensions
    n_steps, relax_kind, relax_done, zmag_relax = relax_dims

    # Check relative length of basis buffer
    if relax_kind == "vcrelax":
        assert len(basis_steps) == len(pos)

    # Convert coordinates if needed
    pos_i = convert_positions(pos_i, basis_i, ctype_i, target_coord, alat=alat)
    basis_steps = (basis_i,) * len(pos) if len(basis_steps) == 0 else tuple(basis_steps)
    pos = tuple(
        starmap(
            lambda basis, tau: convert_positions(
                tau, basis, ctype, target_coord, alat=alat
            ),
            zip(basis_steps, pos),
        )
    )

    # The geometry is unchanged in a magnetization check, just eliminate the last
    if zmag_relax:
        pos = pos[:-1]
        basis_steps = basis_steps[:-1]

    # Check length of buffer
    if relax_done:
        # The final duplicate SCF in vc-relax does not register on the step count
        # The converged geometry is printed again after SCF at the end of relax
        # However the first geometry is captured in the init_geom buffer
        if len(pos) != n_steps:
            raise ParserError("Unexpected length for geometry")

        # Remove final duplicate geometry
        if relax_kind == 'relax':
            pos = pos[:-1]
            basis_steps = basis_steps[:-1]
    else:
        # First geometry is not counted here
        if len(pos) == n_steps - 1:
            pass
        elif len(pos) == n_steps:
            # Geometry written for a step that did not finish
            pos = pos[:-1]
            basis_steps = basis_steps[:-1]
        else:
            raise ParserError("Unexpected length for geometry")

    return (basis_i,) + basis_steps, species, (pos_i,) + pos


def _trim_data_buffs(buffers, relax_dims):
    # type: (MutableMapping[str, Sequence[Any]], _RelaxDims) -> MutableMapping[str, Sequence[Any]]

    # Unpack relax dims
    n_steps, relax_kind, relax_done, zmag_relax = relax_dims

    if relax_done:
        # The final duplicate SCF in vc-relax does not register in step count
        expected_len = n_steps if relax_kind == 'relax' else n_steps + 1
    else:
        expected_len = n_steps

    for tag in buffers:
        if zmag_relax:
            if relax_kind == 'vcrelax' and relax_done:
                # We want to keep the result from the final scf in a vc-relax
                del buffers[tag][-2]
            else:
                buffers[tag] = buffers[tag][:-1]

        if len(buffers[tag]) != expected_len:
            if not relax_done and len(buffers[tag]) == expected_len + 1:
                # This qty was written for a step that did not finish
                buffers[tag] = buffers[tag][:-1]
            else:
                raise ParserError("Unexpected length for {!r} buffer".format(tag))

    return buffers


def parse_relax(path, tags=None, coord_type='crystal'):
    # type: (Path, Optional[Iterable[str]], str) -> Tuple[Optional[GeometryData], RelaxData]
    """Gather data from pw.x relax run.

    :param path: path to pw.x output
    :param tags: Tags to specify which parsers to run on output
    :param coord_type: coordinate type of output

    :returns:
        final_data: None if relaxation did not finish, else a data object
        relax_data: Object with data from each step
    """
    # TODO: Replace typed tuples in returns/params with more explicit datastructure

    # Run parsers on output
    prefix = get_save_file(path)
    n_steps, i_start, i_end, _zmag_relax = _count_relax_steps(path)

    alat, basis_i = get_init_basis(path)
    ctype_i, species_i, pos_i = get_init_coord(path)
    geom_init = (ctype_i, alat, basis_i, species_i, pos_i)

    _relax_data = _get_relax_data(path, tags, n_steps, _zmag_relax)
    energies, final_e, _relax_kind, geom, data_buffs, _zmag_relax = _relax_data
    _relax_done = final_e is not None
    # TODO: Turn relax dims into an explicit flags object
    _relax_dims: _RelaxDims = (n_steps, _relax_kind, _relax_done, _zmag_relax)

    # Trim data buffers
    geom_buffs = _proc_geom_buffs(geom, geom_init, coord_type, _relax_dims)
    basis, species, tau = geom_buffs
    data_buffs = _trim_data_buffs(data_buffs, _relax_dims)

    # Decide if relaxation finished
    if _relax_done:
        # Gather final data
        final_dat = {k: v[-1] for k, v in data_buffs.items()}
        final_data = GeometryData(prefix, basis[-1], species, tau[-1],
                                  energy=final_e, **final_dat,
                                  coord_type=coord_type)

        # Trim final data (only for vc-relax)
        if _relax_kind == 'vcrelax':
            basis = basis[:-1]
            tau = tau[:-1]
            data_buffs = {k: v[:-1] for k, v in data_buffs.items()}
    else:
        final_data = None

    relax_data = RelaxData(prefix, basis, species, tau, energy=energies,
                           coord_type=coord_type, **data_buffs)

    return final_data, relax_data
