"""
Parsers for pw.x output.
"""

import re
import numpy as np

from util import parse_vector

# Vector of atomic species
# Species = Tuple[str, ...]
# Position matrix
# Tau = np.ndarray[natoms, 3]


def get_save_file(path):
    # (path) -> str
    """Extract the prefix from pw.x output."""
    from util import parser_one_line

    save_re = re.compile(r"[ \t]+Writing output data file ([-\w]+).save")
    save_parser = parser_one_line(save_re, lambda m: m.group(1))

    with open(path) as f:
        return save_parser(f)


def get_init_basis(path):
    # type: (path) -> Tuple[float, np.ndarray]
    """Extracts the initial basis in angstrom from pw.x output."""
    from util import parser_one_line, parser_with_header

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
    from util import parser_with_header

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


def get_relax_data(path):
    # type: (path) -> Tuple[str, List[float,...], Union[float, None], List[np.ndarray], Species, Tuple[Tau,...]]
    """Get geometry and energy data from relaxation.

    :param path: Path to pw.x output

    :returns:
        pos_type: Describes the coordinate type of `positions`
        energies: Vector of energies. Includes energy for initial state
        final_energy: Energy of final structure. None if relaxation did not finsh
        species: Vector of atom types
        positions: Vector of atomic positions for each step. Does not include
            initial position but includes final position if relaxation finished
    """
    basis_header_re = re.compile(r"CELL_PARAMETERS \((angstrom)\)")
    geom_header_re = re.compile(r"ATOMIC_POSITIONS \((angstrom|crystal|alat|bohr)\)")
    energy_re = re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry")
    final_energy_re = re.compile(r"[ \t]+Final (energy|enthalpy)[ \t]+=[ \t]+(-[.\d]+) Ry")
    atom_re = re.compile(r"([a-zA-Z]{1,2})((?:[\s]+[-\d.]+){3})")
    basis_row_re = re.compile(r"(?:[\s]+-?[\d.]+){3}")

    buffer_geom = False
    buffer_basis = False
    energies = []
    final_energy = None
    geoms = []
    bases = []
    geom_buf = []
    basis_buf = []

    # Parse the output file
    with open(path) as f:
        for line in f:
            if buffer_geom:
                if atom_re.match(line):
                    spec, pos = atom_re.match(line).groups()
                    pos = parse_vector(pos)
                    geom_buf.append((spec, pos))
                else:
                    buffer_geom = False
                    geoms.append(geom_buf)
                    geom_buf = []
            elif buffer_basis:
                if basis_row_re.match(line):
                    basis_buf.append(parse_vector(line))
                else:
                    buffer_basis = False
                    bases.append(np.array(basis_buf))
                    basis_buf = []
            else:
                if energy_re.match(line):
                    energies.append(float(energy_re.match(line).group(1)))
                elif basis_header_re.match(line):
                    buffer_basis = True
                elif geom_header_re.match(line):
                    buffer_geom = True
                    # Capture the coordinate type
                    geom_buf.append(geom_header_re.match(line).group(1))
                elif final_energy_re.match(line):
                    assert final_energy is None
                    m = final_energy_re.match(line)
                    final_type = m.group(1)
                    final_energy = float(m.group(2))
                else:
                    pass

    # If vc-relax, the true final energy is run in a new scf calculation
    if final_energy is not None:
        if final_type == 'enthalpy':
            final_energy = energies[-1]
            energies = energies[:-1]
        else:
            assert final_type == 'energy'

    # Process geometry buffer to gather species and coordinate types
    def parse_geom_buf(buf):
        pos_type = buf[0]
        spec, pos = zip(*buf[1:])
        return pos_type, spec, np.array(pos)

    pos_type, spec, pos = zip(*map(parse_geom_buf, geoms))

    def all_equal(l):
        return l.count(l[0]) == len(l)

    assert all_equal(pos_type) and all_equal(spec)
    pos_type, spec = pos_type[0], spec[0]

    # Process basis buffer
    if len(bases) > 0:
        assert len(bases) == len(pos)

    return pos_type, energies, final_energy, bases, spec, pos


def parse_relax(path, coord_type='crystal'):
    # type: (path, str) -> Tuple[Union[None, Tuple[float, np.ndarray, Species, Tau]],
    #                            Tuple[List[float], List[np.ndarray], Species, Tuple[Tau, ...]]]
    """Gather data from pw.x relax run.
    
    :param path: path to pw.x output
    :param coord_type: coordinate type of output

    :returns:
        final_data: None if relaxation did not finish, else Tuple[energy, species, positions]
            energy: in Rydberg
            basis: in Angstrom
            species: vector of atomic species
            posititons: shape (natoms, 3), same order as species
        relax_data: Like final_data except energies, bases, and positions are a vector
            with entries for each step
    """
    from itertools import starmap
    from util import convert_coords

    # Run parsers on output
    alat, basis_i = get_init_basis(path)
    ctype_i, species_i, pos_i = get_init_coord(path)
    pos_type, energies, final_e, bases, species, pos = get_relax_data(path)

    assert species_i == species

    # Convert coordinates if needed
    pos_i = convert_coords(alat, basis_i, pos_i, ctype_i, coord_type)
    basis_steps = (basis_i,) * len(pos) if len(bases) == 0 else tuple(bases)
    pos = tuple(starmap(lambda basis, tau: convert_coords(alat, basis, tau, pos_type, coord_type),
                        zip(basis_steps, pos)))

    # Decide if relaxation finished
    final_data = (final_e, basis_steps[-1], species, pos[-1]) if final_e is not None else None

    # If relaxation finished, pos and bases contain duplicate final data
    if final_data is not None:
        pos = pos[:-1]
        basis_steps = basis_steps[:-1]

    relax_data = (energies, (basis_i,) + basis_steps, species, (pos_i,) + pos)

    return final_data, relax_data
