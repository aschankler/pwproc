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

def get_init_basis(path):
    # type: (path) -> Tuple[float, np.ndarray]
    """Extracts the initial basis in angstrom from pw.x output."""
    bohr_to_ang = 0.529177
    alat_re = r"[ \t]+lattice parameter \(alat\)[ \t]+=[ \t]+([\d.]+)[ \t]+a\.u\."
    basis_head_re = r"[ \t]+crystal axes: \(cart. coord. in units of alat\)"
    basis_line_re = r"[ \t]+a\([\d]\) = \(((?:[ \t]+[-.\d]+){3}[ \t]+)\)"

    found_basis = False
    i_basis = None
    alat = None
    basis = None

    with open(path) as f:
        for line in f:
            if found_basis:
                b_vec = parse_vector(re.match(basis_line_re, line).group(1))
                basis[i_basis] = b_vec
                i_basis = i_basis + 1 if i_basis < 2 else None
                if i_basis is None:
                    # Assume that alat line precedes basis
                    break
            elif re.match(alat_re, line):
                assert alat is None
                alat = float(re.match(alat_re, line).group(1))
            elif re.match(basis_head_re, line):
                assert basis is None
                found_basis = True
                i_basis = 0
                basis = [None] *3
            else:
                pass

    # Convert basis from alat to angstrom
    basis = np.array(basis)
    basis *= alat * bohr_to_ang

    return alat, basis


def get_init_coord(path):
    # type: (path) -> Tuple[Species, Tau]
    """Extracts starting atomic positions in crystal coords."""
    header_re = re.compile(r"[ \t]+site n\.[ \t]+atom[ \t]+positions \(cryst\. coord\.\)")
    line_re = re.compile(r"[ \t]+[\d]+[ \t]+([\w]{1,2})[ \t]+tau\([ \d\t]+\) = \(((?:[ \t]+[-.\d]+){3}[ \t]+)\)")

    capturing = False
    atoms = []

    with open(path) as f:
        for line in f:
            if capturing:
                m = line_re.match(line)
                if m is not None:
                    atoms.append(m.groups())
                else:
                    break
            elif header_re.match(line):
                capturing = True
            else:
                pass

    spec, pos = zip(*atoms)
    pos = np.array(tuple(map(parse_vector, pos)))

    return spec, pos


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
    final_energy_re = re.compile(r"[ \t]+Final energy[ \t]+=[ \t]+(-[.\d]+) Ry")
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
                    final_energy = float(final_energy_re.match(line).group(1))
                else:
                    pass

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
    species_i, pos_i = get_init_coord(path)
    pos_type, energies, final_e, bases, species, pos = get_relax_data(path)

    assert species_i == species

    # Convert coordinates if needed
    pos_i = convert_coords(alat, basis_i, pos_i, 'crystal', coord_type)
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


def get_save_file(path):
    # (path) -> str
    """Extract the prefix from pw.x output."""
    save_re = re.compile(r"[ \t]+Writing output data file ([-\w]+).save/")
    with open(path) as f:
        for line in f:
            if save_re.match(line):
                return save_re.match(line).group(1)
            else:
                pass
