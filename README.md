# pwproc
> Parsers for Quantum Espresso output files

## Usage
`pwproc` has a number of submodules:
- `scf` - Extract geometry and energy from energy calculations using `pw.x`
- `relax` - Extract geometries and energies from structural relaxation using `pw.x`
- `xsf` - Process geometries from `.xsf` files
- `fermi` - Extract the fermi energy from `pw.x` output
- `bands` - Band structures using `pw.x` or `bands.x`
- `template` - Template processor for generating input files

### scf
Usage: `pwproc scf [--xsf FILE] [--energy [FILE]] in_file ...`
### relax
Usage: `pwproc relax [--xsf FILE] [--energy FILE] [--initial|--final] in_file ...`

### fermi
Usage: `pwproc fermi in_file [in_file ...]`

### bands
Usage: `pwproc bands [--bands|--pwx] [--npz FILE] [--csv FILE] in_file`

### template
Usage: `pwproc template [[-v KEY=VAL] ...] [[-f FILE] ...] [--use_env] in_file [out_file]`
