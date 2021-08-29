"""Formatting functions for geometry data."""

import re
from decimal import Decimal
from typing import Callable, List, Optional, Sequence, TypeVar

import numpy as np

_T = TypeVar("_T")

LATTICE_PRECISION = 9
POSITION_PRECISION = 9


def _float_decimal_exp(number: float) -> int:
    """Compute the decimal part of the float in base 10.

    Returns:
        Exponent *e* such that::
            number = m * (10 ** e)

    Refs:
        https://stackoverflow.com/a/45359185
    """
    (_, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def _float_decimal_man(number: float) -> Decimal:
    """Compute the mantissa of the float in base 10.

    Returns:
        Mantissa *m* such that::
            number = m * (10 ** e)
    """
    return Decimal(number).scaleb(-_float_decimal_exp(number)).normalize()


def as_fixed_float(value: float, precision: int):
    """Format ``value`` as a float with ``precision`` digits after the decimal."""
    return f"{value:.{precision}f}"


def as_fortran_exp(value: float, precision: int = 5) -> str:
    """Print as a fortran literal double.

    Examples:
        >>> as_fortran_exp(1.0, 2)
        "1.00d0"
        >>> as_fortran_exp(256.)
        "2.56000d2"
        >>> as_fortran_exp(-0.035, 2)
        "-3.50d-2"

    Args:
        value: Number to format
        precision: Number of digits after the decimal in the mantissa

    Returns:
        Number formatted as a fortran double
    """
    mantissa = _float_decimal_man(value)
    exp = _float_decimal_exp(value)
    return f"{mantissa:.{precision}f}d{exp}"


def from_fortran_exp(str_value: str) -> float:
    """Parse a float written as a fortran literal double.

    Examples:
        >>> from_fortran_exp("1.0d0")
        1.0
        >>> from_fortran_exp("2.3d-2")
        0.023
        >>> from_fortran_exp("-2")
        -2.

    Args:
        str_value: Number formatted as a fortran double

    Returns:
        Formatted value converted to float

    Raises:
        ValueError: If the conversion is not possible
    """
    value_match = re.match(
        r"^(?P<man>-?\d+(?:\.\d*)?)(?:[dD](?P<exp>-?\d+))?$", str_value
    )
    if not value_match:
        raise ValueError(f"Could not convert string as fortran double: {str_value!r}")
    mantissa = float(value_match.group("man"))
    exp = int(value_match.group("exp")) if value_match.group("exp") is not None else 0
    return mantissa * 10 ** exp


def FORMAT_LAT(lat):
    # type: (float) -> str
    """Format component of lattice vector."""
    return as_fixed_float(lat, LATTICE_PRECISION)


def FORMAT_POS(pos):
    # type: (float) -> str
    """Format component of atom position."""
    return as_fixed_float(pos, POSITION_PRECISION)


def columns(
    matrix: Sequence[Sequence[_T]],
    *,
    min_space: int = 3,
    left_pad: Optional[int] = None,
    convert_fn: Callable[[_T], str] = str,
    sep: str = " ",
    align: str = "left",
) -> List[str]:
    """Format `matrix` into a string with aligned columns.

    Args:
        matrix: 2d array of values to print; may not be ragged
        min_space: Minimum number of separators between columns
        left_pad: Minimum number of leading separators (default `min_space`)
        convert_fn: Conversion function applied to each element of `matrix` before
            aligning (default :py:func:`str`)
        sep: String used as a separator between columns (default space)
        align: Direction to align elements if elements in a column are not the same
            width. Either 'left' or 'right' (default 'left')

    Returns:
        Formatted matrix as a list of formatted lines
    """
    if align not in ("left", "right"):
        raise ValueError(f"Incorrect alignment {align!r}")
    if any(len(line) != len(matrix[0]) for line in matrix):
        raise ValueError("Matrix may not be ragged")
    if left_pad is None:
        left_pad = min_space

    field_strings = [list(map(convert_fn, line)) for line in matrix]
    col_widths = [max(map(len, col)) for col in zip(*matrix)]

    complete_lines = []
    for row in field_strings:
        line = []
        for width, field in zip(col_widths, row):
            if align == "left":
                line.append(field.ljust(width, sep))
            else:
                line.append(field.rjust(width, sep))
        complete_lines.append(
            left_pad * sep + (min_space * sep).join(line).rstrip() + "\n"
        )
    return complete_lines


def format_basis(basis, lspace=None):
    # type: (np.ndarray, int) -> str
    """Format a basis (3x3 array)."""
    formatter = FORMAT_LAT
    return "".join(columns(basis, left_pad=lspace, convert_fn=formatter))


def format_tau(species, tau):
    # type: (Sequence[str], np.ndarray) -> str
    """Format a list of atomic positions preceded by their species."""

    def _row_formatter(s):
        return FORMAT_POS(s) if isinstance(s, float) else str(s)

    mat = [[name] + list(pos) for name, pos in zip(species, tau)]
    return "".join(columns(mat, min_space=3, left_pad=0, convert_fn=_row_formatter))
