"""Data structures for holding geometry data."""

from typing import Any, Dict, Final, Iterable, Iterator, Optional, \
    Sequence, Sized, Union

from pwproc.geometry.util import Basis, Species, Tau


class GeometryDataBase:
    """Base class for crystal structure data."""
    _prefix: str = None
    _coord_type: str = None
    _allowed_data_types: Final = ('energy', 'force', 'press', 'mag', 'fermi')
    _basis: Union[Basis, Sequence[Basis]] = None
    _species: Species = None
    _tau: Union[Tau, Sequence[Tau]] = None
    _data: Dict[str, Any] = None

    def __init__(self, prefix, basis, species, tau, coord_type=None, **data):
        # type: (str, Union[Basis, Sequence[Basis]], Species, Union[Tau, Sequence[Tau]], Optional[str], Any) -> None
        self.prefix = prefix
        self.basis = basis
        self.species = species
        self.tau = tau
        self._coord_type = coord_type if coord_type else 'angstrom'
        self._data = {}
        for k, v in data.items():
            setattr(self, k, v)

    @property
    def prefix(self):
        # type: () -> str
        return self._prefix

    @prefix.setter
    def prefix(self, pref):
        # type: (str) -> None
        self._prefix = pref

    @property
    def basis(self):
        # type: () -> Union[Basis, Sequence[Basis]]
        return self._basis

    @basis.setter
    def basis(self, b):
        # type: (Union[Basis, Sequence[Basis]]) -> None
        raise NotImplementedError

    @property
    def species(self):
        # type: () -> Species
        return self._species

    @species.setter
    def species(self, s):
        # type: (Species) -> None
        raise NotImplementedError

    @property
    def tau(self):
        # type: () -> Union[Tau, Sequence[Tau]]
        return self._tau

    @tau.setter
    def tau(self, t):
        # type: (Union[Tau, Sequence[Tau]]) -> None
        raise NotImplementedError

    @property
    def data(self):
        # type: () -> Dict[str, Any]
        return self._data

    @property
    def coord_type(self):
        # type: () -> str
        return self._coord_type

    def convert_coords(self, new_coords):
        # type: (str) -> None
        raise NotImplementedError

    def with_coords(self, new_coords):
        # type: (str) -> GeometryDataBase
        from copy import deepcopy
        new_obj = deepcopy(self)
        new_obj.convert_coords(new_coords)
        return new_obj

    def __getattr__(self, name):
        # type: (str) -> Any
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError


class GeometryData(GeometryDataBase):
    """Crystal structure data.

    prefix: str
        Calculation prefix
    basis: Basis
        Crystal basis in angstrom
    species: Species
        Atomic species present
    tau: Tau
        Atomic positions in crystal basis
    data:
        Optional additional data about the structure
    """

    def __init__(self, prefix, basis, species, tau, coord_type=None, **data):
        # type: (str, Basis, Species, Tau, Optional[str], Any) -> None
        # Updates type signature for class
        super().__init__(prefix, basis, species, tau, coord_type, **data)

    @GeometryDataBase.basis.setter
    def basis(self, b):
        # type: (Basis) -> None
        assert(b.shape == (3, 3))
        self._basis = b

    @GeometryDataBase.species.setter
    def species(self, s):
        # type: (Species) -> None
        if self.tau:
            assert(len(s) == len(self.tau))
        self._species = s

    @GeometryDataBase.tau.setter
    def tau(self, t):
        # type: (Tau) -> None
        assert(t.shape[1] == 3)
        if self.species:
            assert(t.shape[0] == len(self.species))
        self._tau = t

    @classmethod
    def from_poscar(cls, lines):
        # type: (Iterable[str]) -> GeometryData
        """Initialize structure from poscar file"""
        from pwproc.geometry import read_poscar
        pref, _, basis, species, tau = read_poscar(lines)

        return cls(pref, basis, species, tau)

    def convert_coords(self, new_coord):
        # type: (str) -> None
        from pwproc.geometry import convert_coords
        self.tau = convert_coords(1.0, self.basis, self.tau, self.coord_type, new_coord)
        self._coord_type = new_coord

    def to_xsf(self):
        # type: () -> Iterator[str]
        from pwproc.geometry import gen_xsf
        return gen_xsf(self.basis, self.species, self.tau)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        if name in self._allowed_data_types:
            self.data[name] = value
        else:
            object.__setattr__(self, name, value)


class RelaxData(GeometryDataBase, Sized):
    """Data about a relaxation run.

    prefix: str
        Calculation prefix
    basis: Sequence[Basis]
        Crystal basis in angstrom
    species: Species
        Atomic species present
    tau: Sequence[Tau]
        Atomic positions in crystal basis
    data:
        Optional additional data about the structure
    """

    def __init__(self, prefix, basis, species, tau, coord_type=None, **data):
        # type: (str, Sequence[Basis], Species, Sequence[Tau], Optional[str], Any) -> None
        # Updates type signature for class
        super().__init__(prefix, basis, species, tau, coord_type, **data)

    @property
    def nsteps(self):
        # type: () -> int
        if self.basis:
            return len(self.basis)
        elif self.tau:
            return len(self.tau)
        else:
            return 0

    def convert_coords(self, new_coords):
        # type: (str) -> None
        from pwproc.geometry import convert_coords

        self.tau = tuple(convert_coords(1.0, b, t, self.coord_type, new_coords)
                         for b, t in zip(self.basis, self.tau))
        self._coord_type = new_coords

    @GeometryData.basis.setter
    def basis(self, b):
        # type: (Sequence[Basis]) -> None
        assert(b[0].shape == (3, 3))
        if self.nsteps:
            assert(len(b) == self.nsteps)
        self._basis = b

    @GeometryData.species.setter
    def species(self, s):
        # type: (Species) -> None
        if self.tau:
            assert(len(s) == self.tau[0].shape)
        self._species = s

    @GeometryData.tau.setter
    def tau(self, t):
        # type: (Sequence[Tau]) -> None
        assert(t[0].shape[1] == 3)
        if self.species:
            assert(t[0].shape[0] == len(self.species))
        if self.nsteps:
            assert(len(t) == self.nsteps)
        self._tau = t

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        if name in self._allowed_data_types:
            if self.nsteps and len(value) != self.nsteps:
                raise ValueError("Incorrect length for {!r}".format(name))
            self.data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __or__(self, other):
        # type: (RelaxData) -> RelaxData
        """Overloaded to join datasets."""
        assert(isinstance(other, self.__class__))
        assert(self.prefix == other.prefix)
        assert(self.species == other.species)
        assert(self.coord_type == other.coord_type)
        new_basis = self.basis + other.basis
        new_tau = self.tau + other.tau
        new_data = {k: getattr(self, k) + getattr(other, k) for k in self.data}

        return self.__class__(self.prefix, new_basis, self.species, new_tau,
                              coord_type=self.coord_type, **new_data)

    def __len__(self):
        # type: () -> int
        return self.nsteps

    def __getitem__(self, item):
        # type: (int) -> GeometryData
        slice_data = {k: v[item] for k, v in self.data.items()}
        return GeometryData(self.prefix, self.basis[item], self.species, self.tau[item],
                            coord_type=self.coord_type, **slice_data)

    def get_init(self):
        # type: () -> GeometryData
        return self[0]

    def to_xsf(self):
        # type: () -> Iterator[str]
        from pwproc.geometry import gen_xsf_animate
        return gen_xsf_animate(self.basis, self.species, self.tau)
