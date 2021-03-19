"""Add jitter to geometry."""

from pathlib import Path
import numpy as np


def do_rattle_atoms(species, tau, which_atoms=None, if_pos=None, abs_scale=None):
    assert(if_pos is None)
    # TODO: consider if_pos
    if which_atoms is None:
        which_atoms = set(species)
    else:
        which_atoms = set(which_atoms) if len(which_atoms) > 0 else set(species)
    if abs_scale is None:
        abs_scale = 0.1

    mask = [at in which_atoms for at in species]
    noise = np.random.uniform(-1, 1, tau.shape)
    noise = np.where(mask, noise.T, 0.0).T

    return tau + abs_scale * noise


def do_rattle_cell(basis, rel_scale=None, abs_scale=None):
    # TODO: this disrupts the atomic positions if they are not in crystal coord.
    if abs_scale is None and rel_scale is None:
        abs_scale = 0.1
    if abs_scale:
        noise = abs_scale * np.random.uniform(-1, 1, (3, 3))
    else:
        noise = rel_scale * basis * np.random.uniform(-1, 1, (3, 3))

    return basis + noise


def do_rattle_pwi(namelists, cards, rattle_cell=False, which_atoms=None, scale=None):
    from pwproc.geometry import parse_pwi_cell, parse_pwi_atoms
    from pwproc.geometry import gen_pwi_atoms, gen_pwi_cell

    # Yield namelists unchanged
    for nl in namelists:
        for l in nl.lines:
            yield l

    # Yield cards, modifying cell and positions
    for ca in cards:
        yield '\n'
        if ca.kind == 'ATOMIC_POSITIONS':
            species, tau, if_pos = parse_pwi_atoms(ca)
            tau = do_rattle_atoms(species, tau, abs_scale=scale,
                                  which_atoms=which_atoms, if_pos=if_pos)
            yield from gen_pwi_atoms(species, tau, ca.unit, if_pos=if_pos)
        elif ca.kind == 'CELL_PARAMETERS':
            basis = parse_pwi_cell(ca)
            if rattle_cell:
                basis = do_rattle_cell(basis)
            yield from gen_pwi_cell(basis)
        else:
            for l in ca.lines:
                if l.strip() != '':
                    yield l
    yield '\n'


def do_rattle_xsf(basis, species, tau, which_atoms=None, rattle_cell=False, scale=None):
    from pwproc.geometry import gen_xsf

    tau = do_rattle_atoms(species, tau, which_atoms=which_atoms, abs_scale=scale)
    if rattle_cell:
        basis = do_rattle_cell(basis)

    return gen_xsf(basis, species, tau)


def parse_args_rattle(args):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('infile', type=Path)
    parser.add_argument('-n', type=int, action='store', default=1)
    parser.add_argument('--out', action='store', type=Path)

    type_grp = parser.add_mutually_exclusive_group()
    type_grp.add_argument('--pwi', action='store_true')
    type_grp.add_argument('--xsf', action='store_true')

    parser.add_argument('-e', '--element', action='append', dest='which_atoms')
    parser.add_argument('--cell', action='store_true', dest='rattle_cell')
    parser.add_argument('--scale', type=float)

    return parser.parse_args(args)


def rattle(args):
    from pwproc.geometry import read_pwi, read_xsf

    if args.out:
        out_base = args.out.parent
        out_stem = args.out.stem
        out_ext = args.out.suffix
    else:
        out_base = Path.cwd()
        out_stem = args.infile.stem
        out_ext = args.infile.suffix

    if args.n > 1:
        def out_path(i):
            return out_base.joinpath(out_stem + '.{:02d}'.format(i) + out_ext)
    else:
        def out_path(_):
            return out_base.joinpath(out_stem + '.rattle' + out_ext)

    if args.pwi:
        with open(args.infile) as f:
            namelists, cards = read_pwi(f)

        for i in range(args.n):
            out_data = do_rattle_pwi(namelists, cards,
                                      rattle_cell=args.rattle_cell,
                                      which_atoms=args.which_atoms,
                                      scale=args.scale)
            with open(out_path(i), 'w') as f:
                f.writelines(out_data)

    elif args.xsf:
        with open(args.infile) as f:
            geom = read_xsf(f)

        for i in range(args.n):
            out_data = do_rattle_xsf(*geom, rattle_cell=args.rattle_cell,
                                     which_atoms=args.which_atoms,
                                     scale=args.scale)

            with open(out_path(i), 'w') as f:
                f.writelines(out_data)

    else:
        raise ValueError


if __name__ == '__main__':
    import sys
    args = parse_args_rattle(sys.argv[1:])
    rattle(args)

