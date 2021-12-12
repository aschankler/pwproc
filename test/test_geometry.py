from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    ContextManager,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pytest

import pwproc.geometry
from pwproc.geometry.cell import Basis, Species, Tau, MovableFlags

# pylint: disable=import-outside-toplevel


@contextmanager
def no_exception() -> Generator[None, None, None]:
    yield


Geometry = Tuple[Basis, Species, Tau]

GEOMETRY_DATA = {
    "BN_crystal": {
        "basis": 3.57 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        "species": ("B", "N"),
        "tau": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
    },
}


def check_geometry(geom: Geometry, geom_ref: Geometry) -> None:
    basis, species, tau = geom
    basis_ref, species_ref, tau_ref = geom_ref
    assert (basis == basis_ref).all()
    assert species == species_ref
    assert (tau == tau_ref).all()


@pytest.fixture
def geometry_directory() -> Path:
    return Path(__file__).parent.joinpath("data", "geometry")


@pytest.fixture
def geometry_ref(request) -> Geometry:
    data = GEOMETRY_DATA[request.param]
    basis = pwproc.geometry.cell.Basis(data["basis"].copy())
    species = pwproc.geometry.cell.Species(data["species"])
    tau = pwproc.geometry.cell.Tau(data["tau"].copy())
    return basis, species, tau


# ------------------
# Test `format_util` module
# ------------------


@pytest.mark.parametrize(
    "value,precision,formatted,expect_raise",
    [
        (1.0, 1, "1.0", no_exception()),
        (1.0, 2, "1.00", no_exception()),
        (1.0, 0, "1.", no_exception()),
        (2.34, 2, "2.34", no_exception()),
        (2.34, 1, "2.3", no_exception()),
        (-2.5, 1, "-2.5", no_exception()),
        (1.0, -1, None, pytest.raises(ValueError)),
    ],
)
def test_fixed_precision(
    value: float, precision: int, formatted: str, expect_raise: ContextManager
) -> None:
    from pwproc.geometry.format_util import as_fixed_precision

    with expect_raise:
        assert formatted == as_fixed_precision(value, precision)


@pytest.mark.parametrize(
    "value,width,precision,formatted,expect_raise",
    [
        (1.0, 0, None, None, pytest.raises(ValueError)),
        (1.0, 3, None, "1.0", no_exception()),
        (1.0, 5, None, "1.000", no_exception()),
        (1.0, 5, 2, "01.00", no_exception()),
        (1.0, 5, -2, "01.00", pytest.raises(ValueError)),
        (1.0, 2, None, "1.", no_exception()),
        (1.0, 1, None, "1", no_exception()),
        (1.0, 2, 1, "1", pytest.raises(ValueError)),
        (12.2, 3, None, "12.", no_exception()),
        (12.2, 4, None, "12.2", no_exception()),
        (12.2, 2, None, "12", no_exception()),
        (12.2, 2, 1, "12", pytest.raises(ValueError)),
        (12.2, 1, None, "12", pytest.raises(ValueError)),
        (12020.2, 3, None, "1e4", no_exception()),
        (1e12, 3, None, "1e12", pytest.raises(ValueError)),
        (12020.2, 5, None, "1.2e4", no_exception()),
        (12020.2, 7, 2, "01.20e4", no_exception()),
        (12020.2, 6, 2, "1.20e4", no_exception()),
        (12020.2, 5, 2, "1.20e4", pytest.raises(ValueError)),
        (0.356, 5, None, "0.356", no_exception()),
        (0.356, 3, None, "0.4", no_exception()),
        (0.356, 2, None, "0.", no_exception()),
        (0.00356, 5, None, "4.e-3", no_exception()),
        (0.00356, 3, None, "0.0", no_exception()),
        (0.00356, 6, 2, "3.56e-3", pytest.raises(ValueError)),
    ],
)
def test_fixed_width(
    value: float,
    width: int,
    precision: Optional[int],
    formatted: str,
    expect_raise: ContextManager,
) -> None:
    from pwproc.geometry.format_util import as_fixed_width

    with expect_raise:
        result = as_fixed_width(value, width, precision=precision)
        assert result == formatted
        assert len(result) == width


@pytest.mark.parametrize(
    "value,width,kwargs,formatted,expect_raise",
    [
        (1.0, 5, {}, "1.000", no_exception()),
        (1.0, 5, {"signed": True}, " 1.00", no_exception()),
        (1.0, 5, {"precision": 1, "signed": True}, " 01.0", no_exception()),
        (-1.0, 5, {}, "-1.00", no_exception()),
        (-1.0, 5, {}, "-1.00", no_exception()),
        (-1.0, 5, {"precision": 1}, "-01.0", no_exception()),
        (-1.0, 5, {"signed": True}, "-1.00", no_exception()),
        (-1.0, 5, {"precision": 1, "signed": True}, "-01.0", no_exception()),
        (1.0, 3, {"signed": True}, " 1.", no_exception()),
        (1.0, 2, {"signed": True}, " 1", no_exception()),
        (1.0, 1, {"signed": True}, " 1", pytest.raises(ValueError)),
        (-1.0, 3, {}, "-1.", no_exception()),
        (-1.0, 2, {}, "-1", no_exception()),
        (-1.0, 1, {}, "-1", pytest.raises(ValueError)),
        (-12.2, 3, {}, "-12", no_exception()),
        (-12.2, 2, {}, "-12", pytest.raises(ValueError)),
        (-12020.2, 4, {}, "-1e4", no_exception()),
        (12020.2, 4, {"signed": True}, " 1e4", no_exception()),
        (1e12, 4, {}, "1e12", no_exception()),
        (-1e12, 4, {}, "-1e12", pytest.raises(ValueError)),
        (1.234e4, 6, {"precision": 2, "signed": True}, None, pytest.raises(ValueError)),
        (1.234e4, 7, {"precision": 2, "signed": True}, " 1.23e4", no_exception()),
        (1.234e4, 8, {"precision": 2, "signed": True}, " 01.23e4", no_exception()),
        (-0.356, 6, {}, "-0.356", no_exception()),
        (-0.356, 4, {}, "-0.4", no_exception()),
        (-0.356, 2, {}, "-0", no_exception()),
        (-0.00356, 5, {}, "-4e-3", no_exception()),
        (-0.00356, 7, {}, "-3.6e-3", no_exception()),
    ],
)
def test_fixed_width_signed(
    value: float,
    width: int,
    kwargs: Mapping[str, Any],
    formatted: str,
    expect_raise: ContextManager,
) -> None:
    from pwproc.geometry.format_util import as_fixed_width

    with expect_raise:
        result = as_fixed_width(value, width, **kwargs)
        assert result == formatted
        assert len(result) == width


@pytest.mark.parametrize(
    "value,precision,formatted,expect_raise",
    [
        (1.0, 2, "1.00d0", no_exception()),
        (1.0, 3, "1.000d0", no_exception()),
        (256.0, 3, "2.560d2", no_exception()),
        (256.0, 1, "2.6d2", no_exception()),
        (256.0, 0, "3.d2", no_exception()),
        (-0.035, 2, "-3.50d-2", no_exception()),
        (1.0, -1, "-3.50d-2", pytest.raises(ValueError)),
    ],
)
def test_as_fort_exp(
    value: float, precision: int, formatted: str, expect_raise: ContextManager
) -> None:
    from pwproc.geometry.format_util import as_fortran_exp

    with expect_raise:
        assert formatted == as_fortran_exp(value, precision)


@pytest.mark.parametrize(
    "formatted,value,expect_raise",
    [
        ("1.0d0", 1.0, no_exception()),
        ("1.00d0", 1.0, no_exception()),
        ("1.d0", 1.0, no_exception()),
        ("2.3d-2", 0.023, no_exception()),
        ("2.3d2", 230.0, no_exception()),
        ("2.3d02", 230.0, no_exception()),
        ("-2", -2.0, no_exception()),
        ("-2.02", -2.02, no_exception()),
        ("1e5", None, pytest.raises(ValueError)),
    ],
)
def test_from_fort_exp(
    formatted: str, value: float, expect_raise: ContextManager
) -> None:
    import math

    from pwproc.geometry.format_util import from_fortran_exp

    with expect_raise:
        assert math.isclose(value, from_fortran_exp(formatted))


@pytest.mark.parametrize(
    "data,kwargs,formatted",
    [
        (["ab", "cd"], {}, ["   a   b\n", "   c   d\n"]),
        (["ab", "cd"], {"min_space": 1}, [" a b\n", " c d\n"]),
        (["ab", "cd"], {"min_space": 1, "sep": "-"}, ["-a-b\n", "-c-d\n"]),
        (["ab", "cd"], {"min_space": 1, "left_pad": 0}, ["a b\n", "c d\n"]),
        (["ab", ["cc", "d"]], {"min_space": 1}, [" a  b\n", " cc d\n"]),
        (
            ["ab", ["cc", "d"]],
            {"min_space": 1, "align": "right"},
            ["  a b\n", " cc d\n"],
        ),
        (["ab", "cd"], {"min_space": 1, "convert_fn": str.upper}, [" A B\n", " C D\n"]),
        ([[1, 2], [3, 4]], {"min_space": 1}, [" 1 2\n", " 3 4\n"]),
        (
            [[1, 2], [3, 4]],
            {"min_space": 1, "convert_fn": lambda x: str(x + 1)},
            [" 2 3\n", " 4 5\n"],
        ),
    ],
)
def test_columns(
    data: Sequence[Sequence[Any]], kwargs: Mapping[str, Any], formatted: Sequence[str]
) -> None:
    from pwproc.geometry.format_util import columns

    result = columns(data, **kwargs)
    assert len(result) == len(formatted)
    for tst, ref in zip(result, formatted):
        assert tst == ref


@pytest.mark.parametrize(
    "data,out,expect_raise",
    [
        (["ab", "cd"], ["   a   b\n", "   c   d\n"], no_exception()),
        (
            ["ab", "cd", "ef"],
            ["   a   b\n", "   c   d\n", "   e   f\n"],
            no_exception(),
        ),
        (["abc", "de"], None, pytest.raises(ValueError)),
    ],
)
def test_columns_fail(
    data: Sequence[Sequence[Any]], out: Sequence[str], expect_raise: ContextManager
) -> None:
    from pwproc.geometry.format_util import columns

    with expect_raise:
        result = columns(data)
        assert len(result) == len(result)
        for tst, ref in zip(result, result):
            assert tst == ref


# Todo: test cell
# Todo: test xsf

# ------------------
# Test POSCAR module
# ------------------


class TestPoscar:
    geom_read_data = [
        {"id": "BN_crys", "comment": "Cubic BN", "scale": 3.57, "geom": "BN_crystal"},
        {"id": "BN_cart", "comment": "Cubic BN", "scale": 3.57, "geom": "BN_crystal"},
        {
            "id": "BN_crys_sd",
            "comment": "Cubic BN",
            "scale": 3.57,
            "geom": "BN_crystal",
        },
        {
            "id": "BN_cart_sd",
            "comment": "Cubic BN",
            "scale": 3.57,
            "geom": "BN_crystal",
        },
    ]

    @pytest.fixture
    def file(self, request, geometry_directory) -> Iterable[str]:
        geom_file = geometry_directory.joinpath("POSCAR_" + request.param)
        with open(geom_file) as f:
            return f.readlines()

    @pytest.mark.parametrize(
        ["comment", "file"],
        [
            pytest.param(dat["comment"], dat["id"], id=dat["id"])
            for dat in geom_read_data
        ],
        indirect=["file"],
    )
    def test_comment(self, comment: str, file: Iterable[str]) -> None:
        comment_test = pwproc.geometry.poscar.read_poscar_comment(file)
        assert comment == comment_test

    @pytest.mark.parametrize(
        ["scale", "file"],
        [pytest.param(dat["scale"], dat["id"], id=dat["id"]) for dat in geom_read_data],
        indirect=["file"],
    )
    def test_scale(self, scale: float, file: Iterable[str]) -> None:
        scale_test = pwproc.geometry.poscar.read_poscar_scale(file)
        assert scale == scale_test

    @pytest.mark.parametrize(
        ["geometry_ref", "file"],
        [pytest.param(dat["geom"], dat["id"], id=dat["id"]) for dat in geom_read_data],
        indirect=["geometry_ref", "file"],
    )
    def test_read(self, geometry_ref: Geometry, file: Iterable[str]) -> None:
        geom_test = pwproc.geometry.poscar.read_poscar(file, out_type="crystal")
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.parametrize("coord", ["direct", "cartesian"])
    @pytest.mark.parametrize("scale", [1.0, 3.57])
    @pytest.mark.parametrize(
        ["geometry_ref", "comment"],
        [(k, k) for k in GEOMETRY_DATA],
        indirect=["geometry_ref"],
    )
    def test_write(
        self, geometry_ref: Geometry, comment: str, scale: float, coord: str
    ) -> None:
        """Test writer assuming reader is correct."""
        out_lines = pwproc.geometry.poscar.gen_poscar(
            *geometry_ref,
            comment=comment,
            scale=scale,
            in_type="crystal",
            coordinate_type=coord,
        )
        out_lines = "".join(out_lines).split("\n")
        assert comment == pwproc.geometry.poscar.read_poscar_comment(out_lines)
        assert scale == pwproc.geometry.poscar.read_poscar_scale(out_lines)
        geom_test = pwproc.geometry.poscar.read_poscar(out_lines, out_type="crystal")
        check_geometry(geom_test, geometry_ref)

    @pytest.mark.parametrize(
        "file,dynamics",
        [
            pytest.param("BN_cart", (None, None), id="BN_cart"),
            pytest.param("BN_crys", (None, None), id="BN_crys"),
            pytest.param(
                "BN_cart_sd",
                ((True, True, False), (False, False, False)),
                id="BN_cart_sd",
            ),
            pytest.param("BN_crys_sd", ((True, False, True), None), id="BN_crys_sd"),
        ],
        indirect=["file"],
    )
    def test_read_dynamics(self, file: Iterable[str], dynamics: MovableFlags) -> None:
        result = pwproc.geometry.poscar.read_poscar_selective_dynamics(file)
        assert len(result) == len(dynamics)
        for test, ref in zip(result, dynamics):
            assert test == ref

    @pytest.mark.parametrize(
        "geometry_ref,dynamics",
        [
            pytest.param("BN_crystal", None, marks=pytest.mark.xfail),
            ("BN_crystal", (None, None)),
            ("BN_crystal", (None, (True, True, True))),
            ("BN_crystal", (None, (False, True, True))),
            ("BN_crystal", ((False, False, False), (True, True, True))),
        ],
        indirect=["geometry_ref"],
    )
    def test_write_dynamics(
        self, geometry_ref: Geometry, dynamics: MovableFlags
    ) -> None:
        out_lines = pwproc.geometry.poscar.gen_poscar(
            *geometry_ref, selective_dynamics=dynamics
        )
        out_lines = "".join(out_lines).split("\n")
        dynamics_out = pwproc.geometry.poscar.read_poscar_selective_dynamics(out_lines)
        assert len(dynamics) == len(dynamics_out)
        for test, ref in zip(dynamics_out, dynamics):
            assert test == ref


# writer/reader (something with more atoms) BTO (tetrahedral), LiNbO3,
# Error tests


# Test espresso
# Test data module
