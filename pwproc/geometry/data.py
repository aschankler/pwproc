 
class GeometryData:
    """Crystal structure data.

    prefix: str
        Calculation prefix
    basis: np.ndarray
        Crystal basis in angstrom
    species: Tuple[str, ...]
        Atomic species present
    tau: np.ndarray
        Atomic positions in crystal basis
    data:
        Optional additional data about the structure
    """
    _prefix = None
    _basis = None
    _species = None
    _tau = None
    _coord_type = None
    _data = None
    _allowed_data_types = ('energy', 'force', 'press', 'mag', 'fermi')

    def __init__(self, prefix, basis, species, tau, coord_type=None, **data):
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
        return self._prefix

    @prefix.setter
    def prefix(self, pref):
        self._prefix = pref

    @property
    def coord_type(self):
        return self._coord_type

    def convert_coords(self, new_coord):
        from pwproc.geometry import convert_coords
        self.tau = convert_coords(1.0, self.basis, self.tau, self.coord_type, new_coord)
        self._coord_type = new_coord

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, b):
        assert(b.shape == (3,3))
        self._basis = b

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, s):
        if self.tau:
            assert(len(s) == len(self.tau))
        self._species = s

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, t):
        assert(t.shape[1] == 3)
        if self.species:
            assert(t.shape[0] == len(self.species))
        self._tau = t

    @classmethod
    def from_poscar(cls, lines):
        """Initialize structure from poscar file"""
        from pwproc.geometry import read_poscar
        pref, _, basis, species, tau = read_poscar(lines)

        return cls(pref, basis, species, tau)

    def with_coords(self, new_coords):
        from copy import deepcopy
        new_obj = deepcopy(self)
        new_obj.convert_coords(new_coords)
        return new_obj

    def to_xsf(self):
        from pwproc.geometry import gen_xsf
        return gen_xsf(self.basis, self.species, self.tau)

    @property
    def data(self):
        return self._data

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name in self._allowed_data_types:
            self.data[name] = value
        else:
            object.__setattr__(self, name, value)


class RelaxData(GeometryData):
    """Data about a relaxation run.

    prefix: str
        Calculation prefix
    basis: Sequence[np.ndarray]
        Crystal basis in angstrom
    species: Tuple[str, ...]
        Atomic species present
    tau: Sequence[np.ndarray]
        Atomic positions in crystal basis
    data:
        Optional additional data about the structure
    """
    @property
    def nsteps(self):
        if self.basis:
            return len(self.basis)
        elif self.tau:
            return len(self.tau)
        else:
            return None

    def convert_coords(self, new_coords):
        from pwproc.geometry import convert_coords

        self.tau = tuple(convert_coords(1.0, b, t, self.coord_type, new_coords)
                         for b, t in zip(self.basis, self.tau))
        self._coord_type = new_coords

    @GeometryData.basis.setter
    def basis(self, b):
        assert(b[0].shape == (3,3))
        if self.nsteps: assert(len(b) == self.nsteps)
        self._basis = b

    @GeometryData.species.setter
    def species(self, s):
        if self.tau: assert(len(s) == self.tau[0].shape)
        self._species = s

    @GeometryData.tau.setter
    def tau(self, t):
        assert(t[0].shape[1] == 3)
        if self.species: assert(t[0].shape[0] == len(self.species))
        if self.nsteps: assert(len(t) == self.nsteps)
        self._tau = t

    @classmethod
    def from_poscar(cls, poscar):
        raise NotImplementedError

    def __setattr__(self, name, value):
        if name in self._allowed_data_types:
            if self.nsteps and len(value) != self.nsteps:
                raise ValueError("Incorrect length for {!r}".format(name))
            self.data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __or__(self, other):
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
        return self.nsteps

    def get_init(self):
        init_data = {k: v[0] for k, v in self.data.items()}
        return GeometryData(self.prefix, self.basis[0], self.species, self.tau[0],
                            coord_type=self.coord_type, **init_data)

    def to_xsf(self):
        from pwproc.geometry import gen_xsf_animate
        return gen_xsf_animate(self.basis, self.species, self.tau)
