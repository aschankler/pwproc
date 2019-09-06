# pwproc
> Parsers for Quantum Espresso output files

## Usage
`pwproc` has a number of submodules:
- `relax` - Extract geometries and energies from structural relaxation using `pw.x`
- `bands` - Band structures using `pw.x` or `bands.x`
- `template` - Template processor for generating input files

### relax
Usage: `pwproc relax [--xsf FILE] [--energy FILE] [--initial|--final] in_file ...`

### bands
Usage: `pwproc bands [--bands|--pwx] [--npz FILE] [--csv FILE] in_file`

### template
Usage: `pwproc template [[-v KEY=VAL] ...] [[-f FILE] ...] [--use_env] in_file [out_file]`
