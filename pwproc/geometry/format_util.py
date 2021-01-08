"""Formatting functions for geometry data."""

import numpy as np
from typing import Callable, Iterable, Sequence, Optional, TypeVar

T = TypeVar('T')

LAT_WIDTH = 9
POS_WIDTH = 9


def _set_precision(x, n, signed=True):
    # type: (float, int, bool) -> str
    """Format x as a float with `n` digits after the decimal."""
    sflag = ' ' if signed else ''
    x = abs(x) if not signed else x
    nunits = 1 if round(x) == 0 else int(np.log10(abs(x))) + 1
    p = n + nunits
    return "{{v:0<{sf}{w}.{p}f}}".format(p=p, w=(p+2), sf=sflag) \
           .format(v=round(x, n))


def FORMAT_LAT(lat):
    # type: (float) -> str
    """Format component of lattice vector."""
    return _set_precision(lat, LAT_WIDTH)


def FORMAT_POS(pos):
    # type: (float) -> str
    """Format component of atom position."""
    return _set_precision(pos, POS_WIDTH)


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
    # type: (np.ndarray) -> str
    """Format a basis (3x3 array)."""
    formatter = FORMAT_LAT
    return columns(basis, minspace=3, s_func=formatter, lspace=lspace)


def format_tau(species, tau):
    # type: (Sequence[str], np.ndarray) -> str
    """Format a list of atomic positions preceded by their species."""
    from itertools import chain, starmap
    formatter = lambda s: FORMAT_POS(s) if isinstance(s, float) else str(s)
    mat = starmap(chain, zip(map(lambda x: [x], species), tau))
    return columns(mat, min_space=3, lspace=0, s_func=formatter)
