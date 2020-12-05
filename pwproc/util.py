"""Utilities for use throughout the program.

Defines parser generators and coordinate transformations.
"""

import re

from typing import Callable, Iterable, Match, Optional, Pattern, Sequence, \
                   Tuple, TypeVar, Union

S = TypeVar('S')
T = TypeVar('T')
# Function type to extract data from match objects
matchProcF = Callable[[Match], T]
# Function to parse file
Parser = Callable[[Iterable[str]], Union[T, Sequence[T]]]


def parser_one_line(line_re, proc_fn, find_multiple=False):
    # type: (Pattern, matchProcF[T], bool) -> Parser[T]
    """Generate a parser to look for isolated lines."""

    def parser(lines):
        # type: (Iterable[str]) -> T
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
        return results if find_multiple else None

    return parser


def parser_with_header(header_re: Pattern, line_re: Pattern,
                       line_proc: matchProcF[T],
                       header_proc: Optional[matchProcF[S]] = None,
                       find_multiple: bool = False
                       ) -> Parser[Union[Sequence[T], Tuple[S, Sequence[T]]]]:
    """Generate a parser to match a group of lines preceded by a header.

    :param header_re: Regex to match the header
    :param line_re: Regex to match lines following the header
    :param line_proc: Function to extract results for each matched line
    :param header_proc: Function to extract results from the header
    :param find_multiple: If true, searches for a new header after the end of
        the first group is found.

    :returns: Parser to call on lines of a file
    """
    def parser(lines):
        # type: (Iterable[str]) -> Union[Sequence[T], Tuple[S, Sequence[T]]]
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


def parse_vector(s, *, num_re=re.compile(r"[\d.-]+")):
    # type: (str) -> Tuple[float]
    """Convert a vector string into a tuple."""
    return tuple(map(float, num_re.findall(s)))
