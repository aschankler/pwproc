"""todo"""

from collections.abc import Callable, Generator, Iterable, Mapping
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
        if abs(scf_corr) > 10 ** -_precision:
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
            data_file.writelines(_format_data_output(record.prefix, tag, record, extra_tags))
        data_file.write("\n")
