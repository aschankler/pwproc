"""
Parsers for pw.x output.
"""

import re
import numpy as np

from pwproc.util import parse_vector

# Vector of atomic species
# Species = Tuple[str, ...]
# Position matrix
# Tau = np.ndarray[natoms, 3]


def get_save_file(path):
    # (path) -> str
    """Extract the prefix from pw.x output."""
    from pwproc.util import parser_one_line

    save_re = re.compile(r"^[ \t]+Writing output data file (?:\./)?([-.\w]+).save/?$")
    save_parser = parser_one_line(save_re, lambda m: m.group(1))

    with open(path) as f:
        prefix = save_parser(f)

    assert(prefix is not None)
    return prefix


def get_init_basis(path):
    # type: (path) -> Tuple[float, np.ndarray]
    """Extracts the initial basis in angstrom from pw.x output."""
    from pwproc.util import parser_one_line, parser_with_header

    bohr_to_ang = 0.529177
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
    # type: (path) -> Tuple[str, Species, Tau]
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

    spec, pos = zip(*atom_coords)
    pos = np.array(tuple(map(parse_vector, pos)))

    return coord_type, spec, pos


# ParseProcFn = Callable[[Match, Iterator[Text], List[Any]], None]
# ParseData = Tuple[Pattern, ParseProcFn]

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
    # (Iterable[str]) -> Mapping[str, ParseData]
    """Defines parsers for requested data types."""
    def _proc_geometry(match, lines, buff):
        atom_re = re.compile(r"([a-zA-Z]{1,2})((?:[\s]+[-\d.]+){3})")
        geom_tmp = []
        pos_type = match.group(1)
        l = next(lines)
        m = atom_re.match(l)
        while m:
            s, pos = m.groups()
            geom_tmp.append((s, parse_vector(pos)))
            l = next(lines)
            m = atom_re.match(l)
        species, tau = zip(*geom_tmp)
        buff.append((pos_type, species, np.array(tau)))

    def _proc_basis(_, lines, buff):
        basis_row_re = re.compile(r"(?:[\s]+-?[\d.]+){3}")
        basis_tmp = []
        l = next(lines)
        while basis_row_re.match(l):
            basis_tmp.append(parse_vector(l))
            l = next(lines)
        assert(len(basis_tmp) == 3)
        buff.append(np.array(basis_tmp))

    def _proc_press(match, lines, buff):
        press_row_re = re.compile(r"^(?: +-?[.\d]+){6} *$")
        press_tmp = []
        l = next(lines)
        while press_row_re.match(l):
            press_tmp.append(parse_vector(l))
            l = next(lines)
        assert(len(press_tmp) == 3)
        tot_p = float(match.group(1))
        press_tmp = np.array(press_tmp)
        press_au = press_tmp[:, :3]
        press_bar = press_tmp[:, 3:]
        buff.append((tot_p, press_au, press_bar))

    def _proc_mag(match, lines, buff):
        m_tot = float(match.group(1))
        l = next(lines)
        m = re.match(r"^ * absolute magnetization += +(-?[.\d]+) +Bohr mag/cell *$", l)
        assert(m is not None)
        m_abs = float(m.group(1))
        assert(next(lines).strip() == '')
        l = next(lines)
        m_conv = re.match(r"^ +convergence has been achieved in +[\d]+ iterations *$", l)
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
               'press': (re.compile(r"^ *total   stress .* \(kbar\) +P= +(-?[.\d]+) *$"),
                         _proc_press),
               'mag': (re.compile(r"^ *total magnetization += +(-?[.\d]+) +Bohr mag/cell *$"),
                       _proc_mag)
               }

    return {tag: parsers[tag] for tag in tags}


def _proc_relax_data(buffers):
    # type: (Mapping[str, Sequence[Any]]) -> Any
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

        # If vc-relax, the true final energy is run in a new scf calculation
        # TODO: unclear if we should discard the energy or duplicate the geometry
        if final_type == 'enthalpy':
            relax_kind = 'vcrelax'
            energy = energy[:-1]
        else:
            assert(final_type == 'energy')
            relax_kind = 'relax'

    n_steps = len(energy)

    def all_equal(l):
        return l.count(l[0]) == len(l)

    # Process geometry
    pos_type, species, pos = zip(*buffers['geom'])
    assert(all_equal(pos_type) and all_equal(species))
    pos_type, species = pos_type[0], species[0]
    bases = buffers['basis']
    if len(bases) > 0:
        assert(len(bases) == len(pos))
    geometry = (pos_type, bases, species, pos)

    # Verify size of other buffers
    data_buffers = {}
    for t in tags - _base_tags:
        data_buffers[t] = buffers[t]
        if relax_kind == 'vcrelax' and final_en is not None:
            # TODO: Again a semi-duplicate entry from the final SCF calculation
            assert(len(data_buffers[t]) == n_steps + 1)
            data_buffers[t] = data_buffers[t][:-1]
        else:
            assert(len(data_buffers[t]) == n_steps)

    # TODO: process other buffers if present

    return energy, final_en, relax_kind, geometry, data_buffers


def _get_relax_data(path, tags):
    # type: (Path, Optional[Iterable[str]]) -> Any
    if tags is None:
        tags = set()
    else:
        tags = set(tags)
    tags = tags | _base_tags
    parsers = _get_relax_parsers(tags)
    buffers = _run_relax_parsers(path, parsers)
    return _proc_relax_data(buffers)


def parse_relax(path, tags=None, coord_type='crystal'):
    # type: (Path, Optional[Iterable[str]], str) -> Tuple[Optional[GeometryData], RelaxData]
    """Gather data from pw.x relax run.
    
    :param path: path to pw.x output
    :param coord_type: coordinate type of output

    :returns:
        final_data: None if relaxation did not finish, else a data object
        relax_data: Object with data from each step
    """
    from itertools import starmap
    from pwproc.util import convert_coords
    from pwproc.geometry import GeometryData, RelaxData

    # Run parsers on output
    prefix = get_save_file(path)
    alat, basis_i = get_init_basis(path)
    ctype_i, species_i, pos_i = get_init_coord(path)

    energies, final_e, _relax_kind, geom, data_bufs =  _get_relax_data(path, tags)
    pos_type, bases, species, pos = geom

    assert species_i == species

    # Convert coordinates if needed
    pos_i = convert_coords(alat, basis_i, pos_i, ctype_i, coord_type)
    basis_steps = (basis_i,) * len(pos) if len(bases) == 0 else tuple(bases)
    pos = tuple(starmap(lambda basis, tau: convert_coords(alat, basis, tau, pos_type, coord_type),
                        zip(basis_steps, pos)))

    # Decide if relaxation finished
    if final_e is not None:
        final_dat = {k: v[-1] for k, v in data_bufs.items()}
        final_data = GeometryData(prefix, basis_steps[-1], species, pos[-1],
                                  energy=final_e, **final_dat,
                                  coord_type=coord_type)
    else:
        final_data = None

    # If relaxation finished, pos and bases contain duplicate final data
    if final_data is not None:
        pos = pos[:-1]
        basis_steps = basis_steps[:-1]
    else:
        # If the relaxation was interupted, there may be an extra position entry
        if len(energies) == len(pos):
            # Extra entry after adding the initial coords
            pos = pos[:-1]
            basis_steps = basis_steps[:-1]
        else:
            assert(len(energies) == len(pos) + 1)

    relax_data = RelaxData(prefix, (basis_i,) + basis_steps, species,
                           (pos_i,) + pos, energy=energies, **data_bufs,
                           coord_type=coord_type)

    return final_data, relax_data
