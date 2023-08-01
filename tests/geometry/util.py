from contextlib import contextmanager
from typing import Any, Generator, Sequence, Tuple, Union

import numpy as np

from pwproc.geometry import Basis, Species, Tau

Geometry = Tuple[Basis, Species, Tau]
Trajectory = Tuple[Union[Basis, Sequence[Basis]], Species, Sequence[Tau]]


@contextmanager
def no_exception() -> Generator[None, None, None]:
    yield


def check_basis(basis: Basis, basis_ref: Basis, **kw: Any) -> None:
    assert np.isclose(basis, basis_ref, **kw).all()


def canonical_order(tau: Tau) -> Sequence[int]:
    """Return an argsort based on a canonical order for the positions."""
    from functools import total_ordering

    @total_ordering
    class _VectorApproxOrder:
        def __init__(self, obj: np.ndarray) -> None:
            if obj.shape != (3,):
                raise ValueError
            self.obj = obj

        def __eq__(self, other: object) -> bool:
            if isinstance(other, np.ndarray):
                return np.isclose(self.obj, other).all()
            if isinstance(other, _VectorApproxOrder):
                return np.isclose(self.obj, other.obj).all()
            return NotImplemented

        @staticmethod
        def _vec_approx_lt(x: np.ndarray, y: np.ndarray) -> bool:
            for xi, yi in zip(x, y):
                if np.isclose(xi, yi):
                    continue
                return xi < yi
            # All components equal
            return False

        def __lt__(self, other: object) -> bool:
            if isinstance(other, np.ndarray):
                if other.shape != (3,):
                    return NotImplemented
                return self._vec_approx_lt(self.obj, other)
            if isinstance(other, _VectorApproxOrder):
                return self._vec_approx_lt(self.obj, other.obj)
            return NotImplemented

    decorated = [(_VectorApproxOrder(x), i) for i, x in enumerate(tau)]
    decorated.sort()
    return [i for _, i in decorated]


def check_geometry(geom: Geometry, geom_ref: Geometry) -> None:
    basis, species, tau = geom
    basis_ref, species_ref, tau_ref = geom_ref
    check_basis(basis, basis_ref)

    # Put both in canonical order
    def _order_geom(_species: Species, _tau: Tau) -> Tuple[Species, Tau]:
        idx = canonical_order(_tau)
        return Species(tuple(_species[i] for i in idx)), Tau(_tau[idx])

    species, tau = _order_geom(species, tau)
    species_ref, tau_ref = _order_geom(species_ref, tau_ref)

    # Now geometry should match exactly
    assert species == species_ref
    assert np.isclose(tau, tau_ref).all()
