"""CLI for the main app."""

import sys

def show_usage():
    usage = """Usage: pwproc [subcommand] [args]

    Subcommands:
        scf
        relax
        fermi
        bands
        berry
        rattle
        template
        xsf
        help
    """
    print(usage)


def run_cli(cli_args):

    if len(cli_args) < 2:
        show_usage()
        return

    sub_cmd = cli_args[1]
    cli_args = cli_args[2:]

    if sub_cmd == "scf":
        from pwproc.scf import parse_args_scf, run_scf
        args = parse_args_scf(cli_args)
        run_scf(args)
    elif sub_cmd == "relax":
        from pwproc.relax import parse_args_relax, run_relax
        args = parse_args_relax(cli_args)
        run_relax(args)
    elif sub_cmd == "xsf":
        from pwproc.xsfmod import parse_args_xsf, run_xsf
        args = parse_args_xsf(cli_args)
        run_xsf(args)
    elif sub_cmd == "fermi":
        from pwproc.fermi import parse_args_fermi, run_fermi
        args = parse_args_fermi(cli_args)
        run_fermi(args)
    elif sub_cmd == "bands":
        from pwproc.bands import parse_args_bands, run_bands
        args = parse_args_bands(cli_args)
        run_bands(args)
    elif sub_cmd == "berry":
        from pwproc.berry import parse_args_berry, run_berry
        args = parse_args_berry(cli_args)
        run_berry(args)
    elif sub_cmd == "rattle":
        from pwproc.rattle import parse_args_rattle, run_rattle
        args = parse_args_rattle(cli_args)
        run_rattle(args)
    elif sub_cmd == "template":
        from pwproc.template import template_cli
        template_cli(cli_args)
    elif sub_cmd in ("help", "--help", "-h"):
        show_usage()
    else:
        show_usage()
        sys.exit(1)


def cli():
    run_cli(sys.argv)


if __name__ == '__main__':
    cli()
