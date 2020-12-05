"""Parsers for pw.x output."""

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Text, Tuple, List
from typing import Callable, Iterator, Optional, Pattern, Match, Union
import numpy as np

from pwproc.util import parse_vector

# Vector of atomic species
Species = Tuple[str, ...]
# Crystal basis [3x3]
Basis = np.ndarray
# Position matrix [n_atoms x 3]
Tau = np.ndarray


def get_save_file(path):
    # type: (Path) -> str
    """Extract the prefix from pw.x output."""
    from pwproc.util import parser_one_line

    save_re = re.compile(r"^[ \t]+Writing output data file (?:\./)?([-.\w]+).save/?$")
    save_parser = parser_one_line(save_re, lambda m: m.group(1))

    with open(path) as f:
        prefix = save_parser(f)

    assert(prefix is not None)
    return prefix


def get_init_basis(path):
    # type: (Path) -> Tuple[float, np.ndarray]
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
        alat = alat_parser(f)
        f.seek(0)
        basis = basis_parser(f)

    # Convert basis from alat to angstrom
    assert len(basis) == 3
    basis = np.array(basis)
    basis *= alat * bohr_to_ang

    return alat, basis


def get_init_coord(path):
    # type: (Path) -> Tuple[str, Species, Tau]
    """Extracts starting atomic positions."""
    from pwproc.util import parser_with_header

    header_re = re.compile(r"[ \t]+site n\.[ \t]+atom[ \t]+positions \((cryst\. coord\.|alat units)\)")
    line_re = re.compile(r"[ \t]+[\d]+[ \t]+([\w]{1,2})[ \t]+tau\([ \d\t]+\) = \(((?:[ \t]+[-.\d]+){3}[ \t]+)\)")

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
        raise ValueError("Initial coordinates not found.")

    spec, pos = zip(*atom_coords)
    pos = np.array(tuple(map(parse_vector, pos)))

    return coord_type, spec, pos


def _count_relax_steps(path):
    # type: (Path) -> Tuple[int, int, int]
    """Count the number of completed steps."""
    scf_re = re.compile(r"^ +number of scf cycles += +(?P<scf>[\d]+)$")
    bfgs_re = re.compile(r"^ +number of bfgs steps += +(?P<bfgs>[\d]+)$")
    last_step_re = re.compile(r"^ +bfgs converged in +(?P<scf>[\d]+) scf cycles and +(?P<bfgs>[\d]+) bfgs steps$")

    steps = []
    last_step = None

    with open(path) as f:
        lines = iter(f)
        for line in lines:
            m1 = scf_re.match(line)
            if m1 is not None:
                n_scf = int(m1.group('scf'))
                m2 = bfgs_re.match(next(lines))
                if m2 is None:
                    raise ValueError("Malformed step count")
                n_bfgs = int(m2.group('bfgs'))
                steps.append((n_scf, n_bfgs))

            m3 = last_step_re.match(line)
            if m3 is not None:
                last_step = (int(m3.group('scf')), int(m3.group('bfgs')))
                break

    if last_step is not None:
        steps.append(last_step)

    return len(steps), steps[0][0], steps[-1][0]


# Standard datastructures for parsing output
ParseProcFn = Callable[[Match, Iterator[Text], List[Any]], None]
ParseData = Tuple[Pattern, ParseProcFn]


def _run_relax_parsers(path, parsers):
    # type: (Path, Mapping[str, ParseData]) -> Mapping[str, List[Any]]
    """Run arbitrary parsers on the pw.x output."""
    # Set up buffers
    buffers = {tag: [] for tag in parsers}

    # Iterate through file
    with open(path) as f:
        lines = iter(f)
        line = next(lines)
        while True:
            for tag, parser in parsers.items():
                header_re, proc_fn = parser

                # Try matching headers
                match = header_re.match(line)

                # If header match, apply the corresponding parser
                if match:
                    proc_fn(match, lines, buffers[tag])
                    break

            try:
                line = next(lines)
            except StopIteration:
                break

    return buffers


_base_tags = frozenset(('energy', 'fenergy', 'geom', 'basis'))
_all_tags = frozenset(['energy', 'fenergy', 'geom', 'basis', 'force', 'press', 'mag'])


def _get_relax_parsers(tags):
    # type: (Iterable[str]) -> Mapping[str, ParseData]
    """Defines parsers for requested data types."""
    def _proc_geometry(match, lines, buff):
        # type: (Match, Iterator[Text], List[Any]) -> None
        atom_re = re.compile(r"([a-zA-Z]{1,2})((?:[\s]+[-\d.]+){3})")
        geom_tmp = []
        pos_type = match.group(1)
        line = next(lines)
        m = atom_re.match(line)
        while m:
            s, pos = m.groups()
            geom_tmp.append((s, parse_vector(pos)))
            line = next(lines)
            m = atom_re.match(line)
        species, tau = zip(*geom_tmp)
        buff.append((pos_type, species, np.array(tau)))

    def _proc_basis(_, lines, buff):
        # type: (Match, Iterator[Text], List[Any]) -> None
        basis_row_re = re.compile(r"(?:[\s]+-?[\d.]+){3}")
        basis_tmp = []
        line = next(lines)
        while basis_row_re.match(line):
            basis_tmp.append(parse_vector(line))
            line = next(lines)
        assert(len(basis_tmp) == 3)
        buff.append(np.array(basis_tmp))

    def _proc_press(match, lines, buff):
        # type: (Match, Iterator[Text], List[Any]) -> None
        press_row_re = re.compile(r"^(?: +-?[.\d]+){6} *$")
        press_tmp = []
        line = next(lines)
        while press_row_re.match(line):
            press_tmp.append(parse_vector(line))
            line = next(lines)
        assert(len(press_tmp) == 3)
        tot_p = float(match.group(1))
        press_tmp = np.array(press_tmp)
        press_au = press_tmp[:, :3]
        press_bar = press_tmp[:, 3:]
        buff.append((tot_p, press_au, press_bar))

    def _proc_mag(match, lines, buff):
        # type: (Match, Iterator[Text], List[Any]) -> None
        m_tot = float(match.group(1))
        line = next(lines)
        m = re.match(r"^ * absolute magnetization += +(-?[.\d]+) +Bohr mag/cell *$", line)
        assert(m is not None)
        m_abs = float(m.group(1))
        assert(next(lines).strip() == '')
        line = next(lines)
        m_conv = re.match(r"^ +convergence has been achieved in +[\d]+ iterations *$", line)
        if m_conv:
            buff.append((m_tot, m_abs))

    parsers = {'energy': (re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry"),
                          lambda m, _, b: b.append(float(m.group(1)))),
               'fenergy': (re.compile(r"[ \t]+Final (energy|enthalpy)[ \t]+=[ \t]+(-[.\d]+) Ry"),
                           lambda m, _, b: b.append((m.group(1), float(m.group(2))))),
               'basis': (re.compile(r"CELL_PARAMETERS \((angstrom)\)"), _proc_basis),
               'geom': (re.compile(r"ATOMIC_POSITIONS \((angstrom|crystal|alat|bohr)\)"),
                        _proc_geometry),
               'force': (re.compile(r"^ *Total force = +([.\d]+) +Total SCF correction = +([.\d]+) *$"),
                         lambda m, _, b: b.append((float(m.group(1)), float(m.group(2))))),
               'press': (re.compile(r"^ *total {3}stress .* \(kbar\) +P= +(-?[.\d]+) *$"),
                         _proc_press),
               'mag': (re.compile(r"^ *total magnetization += +(-?[.\d]+) +Bohr mag/cell *$"),
                       _proc_mag)
               }

    return {tag: parsers[tag] for tag in tags}


def _proc_relax_data(buffers, n_steps):
    # type: (Mapping[str, Sequence[Any]], int) -> Any
    tags = set(buffers)
    assert(_base_tags <= tags)
    assert(tags <= _all_tags)

    # Deal with energies first
    energy = buffers['energy']

    final_en = None
    relax_kind = None
    assert(len(buffers['fenergy']) < 2)
    if len(buffers['fenergy']) == 1:
        final_type, final_en = buffers['fenergy'][0]

        # If vc-relax, the true final energy is run in a new scf calculation,
        # which is captured in the `energy` buffer. The `fenergy` buffer has
        # a duplicate of last relaxation SCF step, which is discarded.
        if final_type == 'enthalpy':
            relax_kind = 'vcrelax'
            if len(energy) == n_steps + 1:
                final_en = energy[-1]
                energy = energy[:-1]
            elif len(energy) == n_steps:
                # In this case, the final SCF step was interrupted
                final_en = None
            else:
                raise ValueError("Unexpected length in energy buffer")

        else:
            assert(final_type == 'energy')
            assert(len(energy) == n_steps)
            relax_kind = 'relax'

    def all_equal(l):
        return l.count(l[0]) == len(l)

    # Re-package geometry
    pos_type, species, pos = zip(*buffers['geom'])
    assert(all_equal(pos_type) and all_equal(species))
    pos_type, species = pos_type[0], species[0]
    bases = buffers['basis']
    if len(bases) > 0:
        assert(len(bases) == len(pos))
    geometry = (pos_type, bases, species, pos)

    # Save the other buffers
    data_buffers = {}
    for t in tags - _base_tags:
        data_buffers[t] = buffers[t]

    return energy, final_en, relax_kind, geometry, data_buffers


def _get_relax_data(path, tags, n_steps):
    # type: (Path, Union[Iterable[str], None], int) -> Any
    if tags is None:
        tags = set()
    else:
        tags = set(tags)
    tags |= _base_tags
    parsers = _get_relax_parsers(tags)
    buffers = _run_relax_parsers(path, parsers)
    return _proc_relax_data(buffers, n_steps)


def _proc_geom_buffs(geom_buff: Tuple[str, Sequence[Basis], Species, Sequence[Tau]],
                     geom_init: Tuple[str, float, Basis, Species, Tau],
                     target_coord: str,
                     n_steps: int,
                     relax_kind: str,
                     relax_done: bool
                     ) -> Tuple[Sequence[Basis], Species, Sequence[Tau]]:
    from itertools import starmap
    from pwproc.util import convert_coords

    # Unpack geometry
    ctype_i, alat, basis_i, species_i, pos_i = geom_init
    ctype, basis_steps, species, pos = geom_buff

    assert(species_i == species)

    # Convert coordinates if needed
    pos_i = convert_coords(alat, basis_i, pos_i, ctype_i, target_coord)
    basis_steps = (basis_i,) * len(pos) if len(basis_steps) == 0 else tuple(basis_steps)
    pos = tuple(starmap(lambda basis, tau: convert_coords(alat, basis, tau, ctype, target_coord),
                        zip(basis_steps, pos)))

    # Check length of buffer
    if relax_done:
        # The final duplicate SCF in vc-relax does not register on the step count
        # However the first geometry is captured in the init_geom buffer
        expected_len = n_steps - 1 if relax_kind == 'relax' else n_steps
        if len(pos) != expected_len:
            raise ValueError("Unexpected length for geometry")
    else:
        # First geometry is not counted here
        if len(pos) == n_steps - 1:
            pass
        elif len(pos) == n_steps:
            # Geometry written for a step that did ont finish
            pos = pos[:-1]
            basis_steps = basis_steps[:-1]
        else:
            raise ValueError("Unexpected length for geometry")

    return (basis_i,) + basis_steps, species, (pos_i,) + pos


def _trim_data_buffs(buffers, n_steps, relax_kind, relax_done):
    # type: (Mapping[str, Sequence[Any]], int, str, bool) -> Mapping[str, Sequence[Any]]

    if relax_done:
        # The final duplicate SCF in vc-relax does not register in step count
        expected_len = n_steps if relax_kind == 'relax' else n_steps + 1
    else:
        expected_len = n_steps

    for tag in buffers:
        if len(buffers[tag]) != expected_len:
            if not relax_done and len(buffers[tag]) == expected_len + 1:
                # This qty was written for a step that did not finish
                buffers[tag] = buffers[tag][:-1]
            else:
                raise ValueError("Unexpected length for {!r} buffer".format(tag))

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
    from pwproc.geometry import GeometryData, RelaxData

    # Run parsers on output
    prefix = get_save_file(path)
    n_steps, i_start, i_end = _count_relax_steps(path)

    alat, basis_i = get_init_basis(path)
    ctype_i, species_i, pos_i = get_init_coord(path)
    geom_init = (ctype_i, alat, basis_i, species_i, pos_i)

    energies, final_e, _relax_kind, geom, data_buffs = _get_relax_data(path, tags, n_steps)
    _relax_done = final_e is not None

    # Trim data buffers
    geom_buffs = _proc_geom_buffs(geom, geom_init, coord_type, n_steps, _relax_kind, _relax_done)
    basis, species, tau = geom_buffs
    data_buffs = _trim_data_buffs(data_buffs, n_steps, _relax_kind, _relax_done)

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
