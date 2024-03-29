"""Basic template processor."""

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

Substitutions = Mapping[str, str]


def parse_args_template(args):
    # type: (Sequence[str]) -> Namespace
    from argparse import Action, ArgumentParser, FileType

    parser = ArgumentParser(prog="pwproc template")

    # Get in/out file handles
    parser.add_argument('in_file', nargs='?', type=FileType('r'), default=sys.stdin)
    parser.add_argument('out_file', nargs='?', type=FileType('w'), default=sys.stdout)

    # Whether to substitute keys strictly
    parser.add_argument('--permissive', action='store_true')

    # Set locations to load variables
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--use_file', '-f', action='append', type=Path)

    # Collect command line vars
    class ParseVar(Action):
        def __call__(self, parser, namespace, values, option_string=None):

            if not values.count("=") == 1:
                raise parser.error(
                    "Variable declaration must be in the form '{}'".format(self.metavar)
                )

            values = values.split('=')
            var_list = getattr(namespace, self.dest)
            try:
                var_list.append(values)
            except AttributeError:
                var_list = [values]
                setattr(namespace, self.dest, var_list)

    parser.add_argument('--var', '-v', action=ParseVar, dest='vars', metavar='VAR=VALUE')
    parser.add_argument('--paste', '-p', action=ParseVar, dest='paste_files', metavar='VAR=FILE')

    parsed = parser.parse_args(args)
    try:
        parsed.vars = dict(parsed.vars)
    except TypeError:
        parsed.vars = dict()

    return parsed


def substitute(tmplt_str, subs, permissive=False):
    # type: (str, Substitutions, bool) -> str
    from string import Template

    template = Template(tmplt_str)
    if not permissive:
        return template.substitute(subs)
    else:
        return template.safe_substitute(subs)


def load_file(path):
    # type: (Path) -> Substitutions
    """Evaluate file and capture variables defined on the main namespace."""

    module = path.parent.name
    ctx = dict(__file__=str(path), __name__=module)

    with open(path) as f:
        source = f.read()

    code = compile(source, str(path), 'exec')
    exec(code, ctx)

    return {
        k: v for k, v in ctx.items() if (not k.startswith("_")) and isinstance(v, str)
    }


def update_subs(sub_dict, files=None, paste_files=None, use_env=False):
    # type: (Substitutions, Optional[Iterable[Path]], Optional[Mapping[str, Path]], bool) -> Substitutions
    """Updates substitution dict from files and env.
     Priority is cmdline args > vars from file > vars from env.
    """

    # Capture vars from environment
    if use_env:
        env_subs = {k: v for k, v in os.environ.items() if not k.startswith('_')}
    else:
        env_subs = {}

    # Load vars from files
    file_subs = {}
    if files is not None:
        for f in files:
            file_subs.update(load_file(f))

    # Load the paste files
    if paste_files:
        paste_subs = {}
        for k, p in paste_files:
            with open(p) as f:
                paste_subs[k] = f.read()
    else:
        paste_subs = {}

    # Update the provided substitutions
    new_subs = env_subs
    new_subs.update(file_subs)
    new_subs.update(paste_subs)
    new_subs.update(sub_dict)

    return new_subs


def run_template(args):
    # type: (Namespace) -> None

    # Get substitutions from all sources
    substitutions = update_subs(args.vars, args.use_file, args.paste_files, args.use_env)

    # Read the template from input
    tmpl = args.in_file.read()

    # Do the substitution and write result
    out_str = substitute(tmpl, substitutions, permissive=args.permissive)
    args.out_file.write(out_str)

    # Close in/out files
    args.in_file.close()
    args.out_file.close()


def template_cli(args):
    # type: (Sequence[str]) -> None
    parsed_args = parse_args_template(args)
    run_template(parsed_args)


if __name__ == "__main__":
    template_cli(sys.argv[1:])
