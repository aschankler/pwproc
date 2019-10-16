 
class GeometryData:
    """Crystal structure data.

    prefix: str
        Calculation prefix
    basis: np.ndarray
        Crystal basis in angstrom
    energy: float
        Energy in Ry
    species: Tuple[str, ...]
        Atomic species present
    tau: np.ndarray
        Atomic positions in crystal basis
    """
    _prefix = None
    _basis = None
    _species = None
    _tau = None
    _energy = None

    def __init__(self, prefix, basis, species, tau, energy=None):
        self.prefix = prefix
        self.basis = basis
        self.species = species
        self.tau = tau
        if energy:
            self.energy = energy

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, pref):
        self._prefix = pref

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

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, e):
        self._energy = e

    @classmethod
    def from_poscar(cls, poscar):
        """"""
        from geometry.poscar import read_poscar
        with open(poscar) as f:
            pref, _, basis, species, tau = read_poscar(f.readlines())

        return cls(pref, basis, species, tau)

    def to_xsf(self):
        from geometry.xsf import gen_xsf
        return gen_xsf(self.basis, self.species, self.tau)


class RelaxData(GeometryData):
    """Data about a relaxation run.

    prefix: str
        Calculation prefix
    basis: Sequence[np.ndarray]
        Crystal basis in angstrom
    energy: List[float]
        Energy in Ry
    species: Tuple[str, ...]
        Atomic species present
    tau: Sequence[np.ndarray]
        Atomic positions in crystal basis
    """
    @property
    def nsteps(self):
        if self.basis:
            return len(self.basis)
        elif self.tau:
            return len(self.tau)
        elif self.energy:
            return len(self.energy)
        else:
            return None

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

    @GeometryData.energy.setter
    def energy(self, e):
        if self.nsteps: assert(len(e) == self.nsteps)
        self._energy = e

    @classmethod
    def from_poscar(cls, poscar):
        raise NotImplementedError

    def __or__(self, other):
        """Overloaded to join datasets."""
        assert isinstance(other, self.__class__)
        assert self.prefix == other.prefix
        assert self.species == other.species
        new_energy = self.energy + other.energy
        new_basis = self.basis + other.basis
        new_tau = self.tau + other.tau

        return self.__class__(self.prefix, new_basis, self.species, new_tau, new_energy)

    def __len__(self):
        return self.nsteps

    def get_init(self):
        return GeometryData(self.prefix, self.basis[0], self.species, self.tau[0], self.energy[0])

    def to_xsf(self):
        from geometry.xsf import gen_xsf_animate
        return gen_xsf_animate(self.basis, self.species, self.tau)
