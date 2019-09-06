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

def get_basis(path):
    # type: (path) -> Tuple[float, np.ndarray]
    """Extracts basis in angstrom from pw.x output."""
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
    # type: (path) -> Tuple[str, List[float,...], Union[float, None], Species, Tuple[Tau,...]]
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
    geom_header_re = re.compile(r"ATOMIC_POSITIONS \((angstrom|crystal|alat|bohr)\)")
    energy_re = re.compile(r"![\s]+total energy[\s]+=[\s]+(-[\d.]+) Ry")
    final_energy_re = re.compile(r"[ \t]+Final energy[ \t]+=[ \t]+(-[.\d]+) Ry")
    atom_re = re.compile(r"([a-zA-Z]{1,2})((?:[\s]+[-\d.]+){3})")

    buffering = False
    energies = []
    final_energy = None
    geoms = []
    geom_buf = []

    # Parse the output file
    with open(path) as f:
        for line in f:
            if buffering:
                if atom_re.match(line):
                    spec, pos = atom_re.match(line).groups()
                    pos = parse_vector(pos)
                    geom_buf.append((spec, pos))
                else:
                    buffering = False
                    geoms.append(geom_buf)
                    geom_buf = []
            else:
                if energy_re.match(line):
                    energies.append(float(energy_re.match(line).group(1)))
                elif geom_header_re.match(line):
                    buffering = True
                    # Capture the coordinate type
                    geom_buf.append(geom_header_re.match(line).group(1))
                elif final_energy_re.match(line):
                    assert final_energy is None
                    final_energy = float(final_energy_re.match(line).group(1))
                else:
                    pass

    # Process geometry buffer to gather species and coordinate types
    def parse_entry(buf):
        pos_type = buf[0]
        spec, pos = zip(*buf[1:])
        return pos_type, spec, np.array(pos)

    pos_type, spec, pos = zip(*map(parse_entry, geoms))

    def all_equal(l):
        return l.count(l[0]) == len(l)

    assert all_equal(pos_type) and all_equal(spec)
    pos_type, spec = pos_type[0], spec[0]

    return pos_type, energies, final_energy, spec, pos


def parse_relax(path, coord_type='crystal'):
    # type: (path, str) -> Tuple[Union[None, Tuple[float, Species, Tau]],
    #                            Tuple[List[float], Species, Tuple[Tau, ...]]]
    """Gather data from pw.x relax run.
    
    :param path: path to pw.x output
    :param coord_type: coordinate type of output

    :returns:
        final_data: None if relaxation did not finish, else Tuple[energy, species, positions]
            energy: in Rydberg
            species: vector of atomic species
            posititons: shape (natoms, 3), same order as species
        relax_data: Like final_data except energies and positions are a vector
            with entries for each step
    """
    from util import convert_coords

    # Run parsers on output
    alat, basis = get_basis(path)
    species_i, pos_i = get_init_coord(path)
    pos_type, energies, final_e, species, pos = get_relax_data(path)

    assert species_i == species

    # Convert coordinates if needed
    pos_i = convert_coords(alat, basis, pos_i, 'crystal', coord_type)
    pos = tuple(map(lambda t: convert_coords(alat, basis, t, pos_type, coord_type), pos))

    # Decide if relaxation finished
    final_data = (final_e, species, pos[-1]) if final_e is not None else None

    # If relaxation finished, pos contains duplicate final position
    if final_data is not None:
        pos = pos[:-1]

    relax_data = (energies, species, (pos_i,) + pos)

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
