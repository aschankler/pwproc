"""Common interface for writing data about structures."""

from argparse import Action, ArgumentParser, FileType, Namespace
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, TextIO, Union

from pwproc.geometry import GeometryData, RelaxData


def write_xsf(xsf_tag: str, data: Iterable[Union[GeometryData, RelaxData]]) -> None:
    """Write structure data to xsf files."""
    written = False
    if xsf_tag == "":

        def _out_path(_prefix: str, _ext: str) -> Path:
            return Path(_prefix + _ext)

    elif "{PREFIX}" in xsf_tag:

        def _out_path(_prefix: str, _ext: str) -> Path:
            stem = xsf_tag.format(PREFIX=_prefix)
            if not stem.endswith(_ext):
                stem += _ext
            return Path(stem)

    else:

        def _out_path(_prefix: str, _ext: str) -> Path:
            nonlocal written
            if written:
                raise ValueError("Saving multiple structures to same file")
            written = True
            return Path(xsf_tag) if xsf_tag.endswith(_ext) else Path(xsf_tag + _ext)

    for geom in data:
        ext = ".xsf" if isinstance(geom, GeometryData) else ".axsf"
        path = _out_path(geom.prefix, ext)
        with open(path, "w", encoding="utf8") as xsf_f:
            xsf_f.writelines(geom.to_xsf())


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
        if abs(scf_corr) > 10**-_precision:
            corr_fstr = f"{{:0{_precision + 2:d}.{_precision:d}f}}"
        else:
            corr_fstr = f"{{:0{_precision + 5:d}.{_precision - 1:d}e}}"
        corr_fmt = corr_fstr.format(scf_corr)
        return f"{force:05.3f}  {corr_fmt}"

    def _fmt_pressure(_geom_data: GeometryData) -> str:
        tot_press, _, press_tensor_kbar = _geom_data.press
        max_press = abs(press_tensor_kbar).max()
        return f"{tot_press: .2f}  {max_press: .2f}"

    def _fmt_volume(_geom_data: GeometryData) -> str:
        # pylint: disable=import-outside-toplevel
        from pwproc.geometry.cell import cell_volume

        volume = cell_volume(_geom_data.basis)
        return f"{volume:.2f}"

    def _fmt_lat(_geom_data: GeometryData) -> str:
        # pylint: disable=import-outside-toplevel
        from pwproc.geometry.cell import cell_parameters

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
        yield f"{prefix}: {_fmt(data_record)}"
        yield "\n"
    else:
        yield f"{prefix} {len(data_record)}\n"
        for _step in data_record:
            yield _fmt(_step)
            yield "\n"


def write_data(
    relax_data: Iterable[Union[GeometryData, RelaxData]],
    data_file: TextIO,
    data_tags: Iterable[str],
    extra_tags: Optional[Mapping[str, Any]] = None,
) -> None:
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
            # FixMe: cannot raise error, since this will be called even if the
            # header is never printed. None should raise error when it is joined
            # with newline, but this is not a good solution
            return None
        fields = sorted(extra_tags["lat"], key=lambda x: x[1])
        header_parts = [f"{name} ({_units[name]})" for name, _ in fields]
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
        for record in relax_data:
            data_file.writelines(
                _format_data_output(record.prefix, tag, record, extra_tags)
            )
        data_file.write("\n")


def add_xsf_arg(parser: ArgumentParser) -> None:
    """Add an option to write .xsf files.

    If the tag is specified,
    """
    parser.add_argument(
        "--xsf",
        action="store",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help=(
            "Write xsf structures to file. The key `{PREFIX}` in FILE is replaced by"
            " the calculation prefix"
        ),
    )


def add_data_args(parser: ArgumentParser, which_tags: Iterable[str]) -> None:
    """Add options to print various data values.

    Modifies `dtags` and `extra_data` fields in the namespace object.
    """
    data_grp = parser.add_argument_group(
        "Data fields", "Specify additional data to gather from output files"
    )

    class ExtraDataAction(Action):
        """Action to save additional fields associated with data tags."""

        # pylint: disable=too-many-arguments,too-few-public-methods

        _extra_data_dest = "extra_data"

        # noinspection PyShadowingBuiltins
        def __init__(
            # pylint: disable=redefined-builtin
            self,
            option_strings: Sequence[str],
            dest: str,
            nargs: Union[int, str, None] = None,
            const: Any = None,
            default: Any = None,
            type: Union[Callable[[str], Any], FileType, None] = None,
            choices: Optional[Iterable[Any]] = None,
            required: bool = False,
            help: Optional[str] = None,
            metavar: Optional[Union[str, tuple[str, ...]]] = None,
        ) -> None:
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

    def _lat_param_convert(value: str) -> tuple[str, int]:
        _param_names = ("a", "b", "c", "alpha", "beta", "gamma")
        value = value.strip().lower()
        if value not in _param_names:
            raise ValueError
        return value, _param_names.index(value)

    _option_params = {
        "energy": ("e", "energy", "Write energy (in Ry)"),
        "force": ("f", "force", "Write force data"),
        "press": ("p", "press", "Write pressure data"),
        "mag": ("m", "mag", "Output magnetization data"),
        "vol": ("v", "volume", "Output unit cell volume"),
        "lat": (
            "L",
            "lat",
            "Output unit cell parameter; possible values are [a, b, c, alpha, beta, gamma]",
        ),
    }
    _extra_data = {
        "lat": (1, "PARAM", _lat_param_convert),
    }

    for tag in which_tags:
        if tag not in _option_params:
            raise ValueError(f"Unrecognized data field {tag}")

        short, long, usage = _option_params[tag]
        if tag not in _extra_data:
            data_grp.add_argument(
                f"--{long}",
                f"-{short}",
                action="append_const",
                dest="dtags",
                const=tag,
                help=usage,
            )
        else:
            nargs, metavar, convert_fn = _extra_data[tag]
            data_grp.add_argument(
                f"--{long}",
                f"-{short}",
                action=ExtraDataAction,
                dest="dtags",
                const=tag,
                nargs=nargs,
                type=convert_fn,
                help=usage,
                metavar=metavar,
            )
