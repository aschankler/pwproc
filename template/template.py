
import sys
import os

#from argparse import Namespace
#from typing import Iterable, Mapping, Any


def parse_args(args):
    # type: (Iterable[str]) -> Namespace
    from argparse import ArgumentParser, Action

    parser = ArgumentParser()

    # Set in/out file names
    parser.add_argument('in_file')
    parser.add_argument('out_file', nargs='?')

    # Set optional args
    parser.add_argument('--use_env', action='store_true')
    parser.add_argument('--use_file', '-f', action='append')

    # Collect command line vars
    class ParseVar(Action):
        def __call__(self, parser, namespace, values, option_string=None):

            if not values.count('=') == 1:
                raise parser.error("Variable declaration must be in the form '{}'".format(self.metavar))

            values = values.split('=')
            var_list = getattr(namespace, self.dest)
            try:
                var_list.append(values)
            except AttributeError:
                var_list = [values]
                setattr(namespace, self.dest, var_list)

    parser.add_argument('--var', '-v', action=ParseVar, dest='vars', metavar='VAR=VALUE')

    parsed = parser.parse_args(args)
    try:
        parsed.vars = dict(parsed.vars)
    except TypeError:
        parsed.vars = dict()

    return parsed


def substitute(tmplt_str, subs):
    # type: (str, Mapping[str, str]) -> str
    from string import Template
    
    template = Template(tmplt_str)
    return template.substitute(subs)


def load_file(filename):
    # type: (str) -> Mapping[str, Any]
    module = os.path.splitext(os.path.basename(filename))[0]
    ctx = dict(__file__=filename, __name__=module)

    with open(filename, 'r') as f:
        source = f.read()

    code = compile(source, filename, 'exec')
    exec(code, ctx)

    return {k: v for k, v in ctx.items() if not k.startswith('_')}


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    if args.use_env:
        args.vars.update(os.environ)

    if args.use_file is not None:
        for f in args.use_file:
            args.vars.update(load_file(f))

    with open(args.in_file, 'r') as f:
        tmpl = f.read()

    out_str = substitute(tmpl, args.vars)

    if args.out_file is None:
        sys.stdout.write(out_str)
    else:
        with open(args.out_file, 'w') as f:
            f.write(out_str)
