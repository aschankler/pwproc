"""Formatting functions for geometry data."""

import re
from decimal import Decimal
from typing import Callable, List, Optional, Sequence, TypeVar

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


def as_fixed_precision(value: float, precision: int) -> str:
    """Format ``value`` as a float with ``precision`` digits after the decimal."""
    if precision < 0:
        raise ValueError("Negative precision")
    return f"{value:#.{precision}f}"


def as_fixed_width(
    value: float, width: int, precision: Optional[int] = None, signed: bool = False
) -> str:
    """Format float to fill exactly `width` characters.

    Examples:
        >>> as_fixed_width(1.0, 5)
        "1.000"
        >>> as_fixed_width(1.0, 5, precision=2)
        "01.00"
        >>> as_fixed_width(12020.2, 5)
        "1.2e4"
        >>> as_fixed_width(0.356, 5)
        "0.356"
        >>> as_fixed_width(0.00356, 5)
        "4.e-3"
        >>> as_fixed_width(0.356, 5, signed=True)
        " 0.36"

    Args:
        value: Number to format
        width: Exact length of the formatted string
        precision: Minimum number of digits after the decimal point. The formatted value
            will always have at least `precision + 1` significant digits.

    Returns:
        Formatted value with width `width`

    Raises:
        ValueError: If the value cannot be accurately represented in fewer than
            `width` characters
    """
    if precision is not None and precision < 0:
        raise ValueError("Negative precision")
    if width < 1:
        raise ValueError("Width too small")
    mantissa = float(_float_decimal_man(value))
    exponent = _float_decimal_exp(value)

    # Width for the trailing 'eNN' string
    exp_suffix_width = 2 + _float_decimal_exp(exponent)
    if exponent < 0:
        exp_suffix_width += 1

    # Minimum width for a float number to give the correct "scale"
    # E.g. 12 -> 2, 1 -> 1, 0.001 -> 5
    if exponent >= 0:
        min_float_width = 1 + exponent
    else:
        min_float_width = 2 + abs(exponent)

    def _format_float(_value: float, _width: int, _units_width: int) -> str:
        sign_char = " " if signed else "-"
        if signed or _value < 0:
            _units_width += 1
        if _width == _units_width:
            # No space for decimals
            if precision is not None and precision != 0:
                raise ValueError("Width too small for desired precision")
            return f"{_value:{sign_char}.0f}"
        max_precision = _width - _units_width - 1
        tgt_precision = precision if precision is not None else max_precision
        if max_precision < tgt_precision:
            raise ValueError("Width too small for desired precision")
        return f"{_value:{sign_char}#0{_width}.{tgt_precision}f}"

    # Choose between float or exp form by whichever allows more significant digits
    if 1 + exp_suffix_width < min_float_width:
        # Use exp form
        if exp_suffix_width + 1 > width:
            # Width too small for exponential form
            if exponent < 0:
                # Print as rounded zero
                return _format_float(0.0, width, 1)
            raise ValueError("Width to small to accurately represent value")
        fmt_man = _format_float(mantissa, width - exp_suffix_width, 1)
        return f"{fmt_man:s}e{exponent:d}"
    else:
        # Format as float
        if exponent < 0:
            # Value is < 1; therefore it can always be formatted (by rounding to 0/1)
            return _format_float(value, width, 1)
        if width < min_float_width:
            raise ValueError("Width too small to accurately represent value")
        return _format_float(value, width, min_float_width)


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
    if precision < 0:
        raise ValueError("Negative precision")
    mantissa = float(_float_decimal_man(value))
    exp = _float_decimal_exp(value)
    return f"{mantissa:#.{precision}f}d{exp}"


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
    col_widths = [max(map(len, col)) for col in zip(*field_strings)]

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
