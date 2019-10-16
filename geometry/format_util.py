
LAT_WIDTH = 9
POS_WIDTH = 9


def _set_precision(x, n, signed=True):
    "display x as a float with n digits after the decimal"
    from numpy import log10, floor
    sflag = ' ' if signed else ''
    x = abs(x) if not signed else x
    nunits = 1 if round(x) == 0 else int(log10(abs(x))) + 1
    p = n + nunits
    return "{{v:0<{sf}{w}.{p}f}}".format(p=p, w=(p+2), sf=sflag) \
           .format(v=round(x, n))


def FORMAT_LAT(lat):
    return _set_precision(lat, LAT_WIDTH)


def FORMAT_POS(pos):
    return _set_precision(pos, POS_WIDTH)


def columns(matrix, minspace, lspace=None, s_func=str, sep=' ', align='front'):
    """
    columns(matrix, minspace[, lspace, s_func, sep, pad]) -> string

    Arranges `matrix` into a string with aligned columns

    Parameters
    ----------
    matrix : Iterable[Iterable]
        May not be ragged
    minspace : int
        Minimum number of separators between columns
    lspace : int, optional
        Minimum number of leading separators (default `minspace`)
    s_func : Callable, optional
        Applied to elements of `matrix` before aligning (default str)
    sep : str, optional
        String used as a separator between columns (default ' ')
    align : str, optional
        Either 'front' or 'rear'. Where to align elements if all elements
        in a column are not the same width. (default 'front')
    """
    if align != 'front' and align != 'rear':
        raise ValueError("Incorrect pad specification")
    if lspace is None:
        lspace = minspace

    matrix = [list(map(s_func, line)) for line in matrix]
    widths = [max(map(len, col)) for col in zip(*matrix)]

    acc = []
    for row in matrix:
        line = []
        for width, item in zip(widths, row):
            extra = width - len(item)
            if align == 'rear':
                line.append(extra*sep + item)
            else:
                line.append(item + extra*sep)
        acc.append(lspace*sep + (minspace*sep).join(line).rstrip())
    return "\n".join(acc)


def format_basis(basis):
    formater = FORMAT_LAT
    return columns(basis, minspace=3, s_func=formater)


def format_tau(species, tau):
    from itertools import chain, starmap
    formater = lambda s: FORMAT_POS(s) if isinstance(s, float) else str(s)
    mat = starmap(chain, zip(map(lambda x: [x], species), tau))
    return columns(mat, minspace=3, lspace=0, s_func=formater)
    