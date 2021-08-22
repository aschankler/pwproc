"""Parser for pw.x relax output."""

from argparse import Namespace
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

from pwproc.geometry.data import GeometryData, RelaxData


def parse_file(path, tags):
    from pwproc.parsers import parse_relax

    final_data, relax_data = parse_relax(path, tags=tags, coord_type='angstrom')
    prefix = relax_data.prefix

    return (prefix, final_data, relax_data)


def parse_files(paths, tags):
    data = {}

    for p in paths:
        prefix, final, relax = parse_file(p, tags)

        if prefix not in data:
            data[prefix] = (final, relax)
        else:
            # We must merge several files
            # TODO: this merge is broken if a relax is split over >2 files
            old_final, old_relax = data[prefix]
            if final is not None:
                # `final` and `relax` are the more recent steps
                data[prefix] = (final, old_relax | relax)
            elif old_final is not None:
                # `old_final` and `old_relax` are more recent
                data[prefix] = (old_final, relax | old_relax)
            else:
                # Join arbitrarily
                data[prefix] = (final, old_relax | relax)

    return data


_DataFormatFn = Callable[[GeometryData], str]


def _format_data_output(
    prefix: str,
    tag: str,
    data_record: Union[GeometryData, RelaxData],
    extra_tags: Optional[Mapping[str, Any]],
) -> Generator[str, None, None]:
    def _fmt_force(_geom_data: GeometryData) -> str:
        _precision = 3
        force, scf_corr = _geom_data.force
        if abs(scf_corr) > 10 ** -_precision:
            corr_fstr = "{{:0{:d}.{:d}f}}".format(_precision + 2, _precision)
        else:
            corr_fstr = "{{:0{:d}.{:d}e}}".format(_precision + 5, _precision - 1)
        corr_fmt = corr_fstr.format(scf_corr)
        return "{:05.3f}  {}".format(force, corr_fmt)

    def _fmt_pressure(_geom_data: GeometryData) -> str:
        tot_press, press_tensor = _geom_data.press
        max_press = abs(press_tensor).max()
        return f"{tot_press: .2f}  {max_press: .2f}"

    def _fmt_volume(_geom_data: GeometryData) -> str:
        # pylint: disable=import-outside-toplevel
        from pwproc.geometry.util import cell_volume

        volume = cell_volume(_geom_data.basis)
        return f"{volume:.2f}"

    def _fmt_lat(_geom_data: GeometryData) -> str:
        # pylint: disable=import-outside-toplevel
        from pwproc.geometry.util import cell_parameters

        if extra_tags is None or "lat" not in extra_tags:
            raise RuntimeError("Extra data for 'lat' formatter not present.")
        fields = sorted(x[1] for x in extra_tags["lat"])
        cell_params = cell_parameters(_geom_data.basis)
        formatted_fields = [f"{cell_params[i]:f}" for i in fields]
        return "  ".join(formatted_fields)

    formatters: Mapping[str, _DataFormatFn] = {
        "energy": lambda _dat: f"{_dat.energy:.5f}",
        "force": _fmt_force,
        "press": _fmt_pressure,
        "mag": lambda _dat: f"{_dat.mag[0]}  {_dat.mag[1]}",
        "vol": _fmt_volume,
        "lat": _fmt_lat,
    }

    # Select the formatting function
    _fmt = formatters[tag]

    # Extract and format data from the record
    if isinstance(data_record, GeometryData):
        yield "{}: {}".format(prefix, _fmt(data_record))
        yield "\n"
    else:
        yield "{} {}\n".format(prefix, len(data_record))
        for _step in data_record:
            yield _fmt(_step)
            yield "\n"
        yield "\n"


def write_data(
    relax_data: Mapping[str, Union[GeometryData, RelaxData]],
    data_file: TextIO,
    data_tags: Iterable[str],
    extra_tags: Optional[Mapping[str, Any]] = None,
):
    """Write additional data to file."""

    def _lat_header() -> str:
        _units = {
            "a": "A",
            "b": "A",
            "c": "A",
            "alpha": "deg",
            "beta": "deg",
            "gamma": "deg",
        }
        if extra_tags is None or "lat" not in extra_tags:
            raise ValueError
        fields = sorted(extra_tags["lat"], key=lambda x: x[1])
        header_parts = ["{} ({})".format(name, _units[name]) for name, _ in fields]
        return "  ".join(header_parts)

    headers = {
        "energy": "Energy (Ry)",
        "force": "Total force   SCF correction  (Ry/au)",
        "press": "Total Press.  Max Press.  (kbar)",
        "mag": "Total mag.  Abs. mag.  (Bohr mag/cell)",
        "vol": "Unit cell volume (A^3)",
        "lat": _lat_header(),
    }

    for tag in data_tags:
        # Write header for this data type
        data_file.write(headers[tag] + "\n")
        # Write data for each file
        for prefix, record in relax_data.items():
            data_file.writelines(_format_data_output(prefix, tag, record, extra_tags))


def write_xsf(xsf, data):
    # type: (str, Mapping[str, GeometryData]) -> None
    """Write structure data to xsf files."""
    if '{PREFIX}' not in xsf:
        if len(data) != 1:
            raise ValueError('Saving multiple structures to same file')
        else:
            # Grab the first (and only) entry
            save_data = zip((xsf,), data.values())
    else:
        save_data = ((xsf.format(PREFIX=pref), geom_data)
                     for pref, geom_data in data.items())

    for path, geom_data in save_data:
        with open(path, 'w') as xsf_f:
            xsf_f.writelines(geom_data.to_xsf())


def parse_args_relax(args):
    # type: (Sequence[str]) -> Namespace
    """Argument parser for `relax` subcommand."""
    import sys
    from argparse import Action, ArgumentParser, FileType
    from pathlib import Path

    parser = ArgumentParser(
        prog="pwproc relax", description="Parser for relax and vc-relax output."
    )

    parser.add_argument(
        "in_file",
        action="store",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="List of pw.x output files",
    )
    parser.add_argument(
        "--xsf",
        action="store",
        metavar="FILE",
        help=(
            "Write xsf structures to file. The key `{PREFIX}` in FILE is replaced by"
            " the calculation prefix"
        ),
    )
    parser.add_argument(
        "--data",
        action="store",
        type=FileType("w"),
        default=sys.stdout,
        metavar="FILE",
        help="Output file for structure data",
    )
    data_grp = parser.add_argument_group(
        "Data fields", "Specify additional data to gather from output files"
    )
    data_grp.add_argument(
        "--energy",
        "-e",
        action="append_const",
        dest="dtags",
        const="energy",
        help="Write energy (in Ry)",
    )
    data_grp.add_argument(
        "--force",
        "-f",
        action="append_const",
        dest="dtags",
        const="force",
        help="Output force data",
    )
    data_grp.add_argument(
        "--press",
        "-p",
        action="append_const",
        dest="dtags",
        const="press",
        help="Output pressure data",
    )
    data_grp.add_argument(
        "--mag",
        "-m",
        action="append_const",
        dest="dtags",
        const="mag",
        help="Output magnetization data",
    )
    data_grp.add_argument(
        "--volume",
        "-v",
        action="append_const",
        dest="dtags",
        const="vol",
        help="Output unit cell volume",
    )

    def lat_param_type(value):
        # type: (str) -> Tuple[str, int]
        _param_names = ("a", "b", "c", "alpha", "beta", "gamma")
        value = value.strip().lower()
        if value not in _param_names:
            raise ValueError
        return value, _param_names.index(value)

    class ExtraDataAction(Action):
        # pylint: disable=too-many-arguments,too-few-public-methods
        _extra_data_dest = "extra_data"

        # noinspection PyShadowingBuiltins
        def __init__(
            # pylint: disable=redefined-builtin
            self,
            option_strings: Sequence[str],
            dest: str,
            nargs: Optional[Union[int, str]] = None,
            const: Any = None,
            default: Any = None,
            type: Union[Callable[[str], Any], FileType, None] = None,
            choices: Optional[Iterable[Any]] = None,
            required: bool = False,
            help: Optional[str] = None,
            metavar: Optional[Union[str, Tuple[str, ...]]] = None,
        ):
            if nargs is None:
                raise ValueError("nargs may not be None")
            if nargs == 0:
                raise ValueError("nargs must be nonzero")
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=nargs,
                const=const,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: Union[str, Sequence[Any], None],
            option_string: Optional[str] = None,
        ) -> None:
            if values is None or isinstance(values, str):
                raise TypeError(
                    f"Incorrect argument type to {self.__class__}. Check nargs?"
                )
            # Add flag to main group
            items = getattr(namespace, self.dest, None)
            items = [] if items is None else items
            if self.const not in items:
                items.append(self.const)
            setattr(namespace, self.dest, items)

            # Add extra data info
            extra_data = getattr(namespace, self._extra_data_dest, None)
            extra_data = {} if extra_data is None else extra_data
            if self.const in extra_data:
                for val in values:
                    if val not in extra_data[self.const]:
                        extra_data[self.const].append(val)
            else:
                extra_data[self.const] = values
            setattr(namespace, self._extra_data_dest, extra_data)

    data_grp.add_argument(
        "--lat",
        "-L",
        action=ExtraDataAction,
        type=lat_param_type,
        dest="dtags",
        const="lat",
        nargs=1,
        help=(
            "Output unit cell parameter; possible values are [a, b, c, alpha, beta,"
            " gamma]"
        ),
        metavar="PARAM",
    )

    endpt = parser.add_mutually_exclusive_group()
    endpt.add_argument(
        "--final",
        dest="endpoint",
        action="store_const",
        const="final",
        help=(
            "Save data only for the final structure. Warn if relaxation did not finish"
        ),
    )
    endpt.add_argument(
        "--last",
        dest="endpoint",
        action="store_const",
        const="last",
        help="Save data for the last structure, even if the relaxation did not finish",
    )
    endpt.add_argument(
        "--initial",
        dest="endpoint",
        action="store_const",
        const="initial",
        help="Save data only for the initial structure",
    )

    return parser.parse_args(args)


def _get_parser_tags(dtags):
    # type: (Iterable[str]) -> Set[str]
    _parser_tags = ("energy", "force", "press", "mag", "fermi")
    # Only pass on some data tags to the parser
    return {tag for tag in dtags if tag in _parser_tags}


def run_relax(args):
    """Main function for `relax` subcommand."""
    # Parse the output files
    parser_tags = _get_parser_tags(args.dtags)
    relax_data = parse_files(args.in_file, parser_tags)

    # Take the desired step
    out_data = {}
    for prefix, data in relax_data.items():
        final, relax = data
        if args.endpoint == 'final':
            if final is None:
                print('Relaxation did not finish for {}'.format(prefix))
            else:
                out_data[prefix] = final
        elif args.endpoint == 'initial':
            out_data[prefix] = relax.get_init()
        elif args.endpoint == 'last':
            if final is None:
                out_data[prefix] = relax[-1]
            else:
                out_data[prefix] = final
        else:
            out_data[prefix] = relax

    # Write XSF file
    if args.xsf:
        write_xsf(args.xsf, out_data)

    # Write additional data
    if args.dtags:
        # extra_data field is not initialized to None
        extra_data = getattr(args, "extra_data", None)
        write_data(out_data, args.data, args.dtags, extra_data)
    args.data.close()


if __name__ == '__main__':
    import sys
    args = parse_args_relax(sys.argv[1:])
    run_relax(args)
