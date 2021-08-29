"""Formatting functions for geometry data."""

import re
from decimal import Decimal
from typing import Callable, Iterable, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar('T')

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


def columns(matrix: Iterable[Iterable[T]],
            min_space: int, lspace: Optional[int] = None,
            s_func: Callable[[T], str] = str, sep: str = ' ',
            align: str = 'front') -> str:
    """Arrange `matrix` into a string with aligned columns.

    Parameters
    ----------
    matrix :
        May not be ragged
    min_space : int
        Minimum number of separators between columns
    lspace : int, optional
        Minimum number of leading separators (default `minspace`)
    s_func : Callable, optional
        Applied to elements of `matrix` before aligning (default str)
    sep : str, optional
        String used as a separator between columns (default ' ')
    align : str, optional
        Either 'front' or 'back'. Where to align elements if all elements
        in a column are not the same width. (default 'front')
    """
    if align != 'front' and align != 'back':
        raise ValueError("Incorrect pad specification")
    if lspace is None:
        lspace = min_space

    matrix = [list(map(s_func, line)) for line in matrix]
    widths = [max(map(len, col)) for col in zip(*matrix)]

    acc = []
    for row in matrix:
        line = []
        for width, item in zip(widths, row):
            extra = width - len(item)
            if align == 'back':
                line.append(extra*sep + item)
            else:
                line.append(item + extra*sep)
        acc.append(lspace * sep + (min_space * sep).join(line).rstrip())
    return "\n".join(acc)


def format_basis(basis, lspace=None):
    # type: (np.ndarray, int) -> str
    """Format a basis (3x3 array)."""
    formatter = FORMAT_LAT
    return columns(basis, min_space=3, s_func=formatter, lspace=lspace)


def format_tau(species, tau):
    # type: (Sequence[str], np.ndarray) -> str
    """Format a list of atomic positions preceded by their species."""
    from itertools import chain, starmap
    formatter = lambda s: FORMAT_POS(s) if isinstance(s, float) else str(s)
    mat = starmap(chain, zip(map(lambda x: [x], species), tau))
    return columns(mat, min_space=3, lspace=0, s_func=formatter)
