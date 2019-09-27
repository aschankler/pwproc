
import re
import numpy as np


# Function type to extract data from match objects
# matchProcF[T] = Callable[[Match], T]
# Function to parse file
# Parser[T] = Callable[[Iterable[Text]], Union[T, List[T]]]

def parser_one_line(line_re, proc_fn, find_multiple=False):
    # type: (Pattern, matchProcF[T], bool) -> Parser[T]
    """Generates a parser to look for isolated lines."""

    def parser(lines):
        # type: (Iterable[Text]) -> Any
        lines = iter(lines)
        results = []

        for line in lines:
            match = line_re.match(line)
            if match:
                processed = proc_fn(match)
                if not find_multiple:
                    return processed
                else:
                    results.append(processed)
            else:
                pass
        return results

    return parser


def parser_with_header(header_re, line_re, line_proc, header_proc=None, find_multiple=False):
    # type: (Pattern, Pattern, matchProcF[T], Optional[matchProcF[S]], bool) -> Parser[Union[List[T], Tuple[S, List[T]]]]
    """Match a group of lines preceded by a header"""

    def parser(lines):
        # type: (Iterable[Text]) -> Any
        lines = iter(lines)
        capturing = False
        results = []
        result = None
        header = None
        buffer = []

        for line in lines:
            if capturing:
                match = line_re.match(line)
                if match is not None:
                    buffer.append(line_proc(match))
                else:
                    result = (header, buffer) if header_proc else buffer
                    buffer = []
                    capturing = False
                    if find_multiple:
                        results.append(result)
                    else:
                        return result
            elif header_re.match(line):
                capturing = True
                if header_proc is not None:
                    header = header_proc(header_re.match(line))
            else:
                pass
        return results

    return parser


def parse_vector(s, *_, num_re=re.compile(r"[\d.-]+")):
    # type: (Text) -> Tuple[float]
    """Convert a vector string into a tuple."""
    return tuple(map(float, num_re.findall(s)))


def convert_coords(alat, basis, tau, in_type, out_type):
    # type: (float, np.ndarray, np.ndarray, str, str) -> np.ndarray
    """Converts coordinate type.

    :param alat: lattice parameter in bohr
    :param basis: basis in angstrom
    :param tau: vector of posititons shape (natoms, 3)
    :param in_type: coordinate type of `tau`
    :param out_type: coordinate type to return
    """
    if in_type == out_type:
        return tau

    # Otherwise convert first to crystal
    tau = to_crystal(alat, basis, tau, in_type)

    # Then to desired type
    tau = from_crystal(alat, basis, tau, out_type)

    return tau


def to_crystal(alat, basis, tau, in_type):
    # type: (float, np.ndarray, np.ndarray, str) -> np.ndarray
    from scipy import constants
    bohr_to_ang = constants.value('Bohr radius') / constants.angstrom

    if in_type == 'crystal':
        return tau
    elif in_type == 'alat':
        return alat * bohr_to_ang * tau @ np.linalg.inv(basis)
    elif in_type == 'angstrom':
        return tau @ np.linalg.inv(basis)
    else:
        raise ValueError("Coord. type {}".format(in_type))


def from_crystal(alat, basis, tau, out_type):
    # type: (float, np.ndarray, np.ndarray, str) -> np.ndarray
    # lattice vectors are rows of the basis
    if out_type == 'crystal':
        return tau
    elif out_type == 'angstrom':
        return tau @ basis
    else:
        raise ValueError("Coord. type {}".format(in_type))
