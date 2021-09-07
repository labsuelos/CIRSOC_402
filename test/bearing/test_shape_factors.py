'''test module for the shape factors in the bearing capacity
equation for shallow foundations'''


import itertools
import numpy as np
import pytest

from cirsoc_402.bearing.bearing_factors import bearing_factor_nc
from cirsoc_402.bearing.bearing_factors import bearing_factor_nq

from cirsoc_402.bearing.shape import shape_factors
from cirsoc_402.bearing.shape import shape_factor_c
from cirsoc_402.bearing.shape import shape_factor_q
from cirsoc_402.bearing.shape import shape_factor_g

from cirsoc_402.bearing.shape import cirsoc_factor_c
from cirsoc_402.bearing.shape import cirsoc_factor_g
from cirsoc_402.bearing.shape import cirsoc_factor_q

from cirsoc_402.bearing.shape import canada_factor_c
from cirsoc_402.bearing.shape import canada_factor_g
from cirsoc_402.bearing.shape import canada_factor_q

from cirsoc_402.bearing.shape import meyerhof_factor_c
from cirsoc_402.bearing.shape import meyerhof_factor_g
from cirsoc_402.bearing.shape import meyerhof_factor_q

from cirsoc_402.bearing.shape import hansen_factor_c
from cirsoc_402.bearing.shape import hansen_factor_g
from cirsoc_402.bearing.shape import hansen_factor_q

from cirsoc_402.bearing.shape import vesic_factor_c
from cirsoc_402.bearing.shape import vesic_factor_g
from cirsoc_402.bearing.shape import vesic_factor_q

def test_shape_factors():
    pass


def test_shape_factor_c():
    pass


def test_shape_factor_q():
    pass


def test_shape_factor_g():
    pass


def test_cirsoc_factor_c():
    pass


def test_cirsoc_factor_q():
    pass


def test_cirsoc_factor_g():
    pass


def test_canada_factor_c():
    '''test function for cirsoc_402.bearing.shape.canada_factor_c
    '''
    assert np.isnan(canada_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_c(30, 2, np.nan)) == True
    assert np.isnan(canada_factor_c(30, np.nan, 3)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(canada_factor_c(np.nan, np.nan, 3)) == True

    # return nan when effective_length=0
    assert np.isnan(canada_factor_c(30, 2, 0)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(canada_factor_c(30, np.nan, 0)) == True
    assert np.isnan(canada_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    for w, l, p in itertools.product(widths, lengths, phi):
        nc = bearing_factor_nc(p)
        nq = bearing_factor_nq(p)
        expected = 1 + w / l * nq / nc
        assert canada_factor_c(p, w, l) == expected


def test_canada_factor_q():
    '''test function for cirsoc_402.bearing.shape.canada_factor_q
    '''
    assert np.isnan(canada_factor_q(np.nan, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_q(30, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_q(30, 2, np.nan)) == True
    assert np.isnan(canada_factor_q(30, np.nan, 3)) == True
    assert np.isnan(canada_factor_q(np.nan, 2, 3)) == True
    assert np.isnan(canada_factor_q(np.nan, 2, np.nan)) == True
    assert np.isnan(canada_factor_q(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(canada_factor_q(30, 2, 0)) == True
    assert np.isnan(canada_factor_q(np.nan, 2, 0)) == True
    assert np.isnan(canada_factor_q(30, np.nan, 0)) == True
    assert np.isnan(canada_factor_q(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    for w, l, p in itertools.product(widths, lengths, phi):
        expected = 1 + w / l * np.tan(np.radians(p))
        assert pytest.approx(canada_factor_q(p, w, l), 0.01) == expected


def test_canada_factor_g():
    '''test function for cirsoc_402.bearing.shape.canada_factor_g
    '''
    assert np.isnan(canada_factor_g(np.nan, np.nan)) == True
    assert np.isnan(canada_factor_g(2, np.nan)) == True
    assert np.isnan(canada_factor_g(np.nan, 2)) == True

    # return nan when width=0
    assert np.isnan(canada_factor_g(2, 0)) == True
    assert np.isnan(canada_factor_g(np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    for w, l in itertools.product(widths, lengths):
        expected = 1 - 0.4 *  w / l
        assert canada_factor_g(w, l) == expected


def test_meyerhof_factor_c():
    '''test function for cirsoc_402.bearing.shape.meyerhof_factor_c
    '''
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, 3)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, 3)) == True

    # return nan when effective_length=0
    assert np.isnan(meyerhof_factor_c(30, 2, 0)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, 0)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    for w, l, p in itertools.product(widths, lengths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected = 1 + 0.2 * nphi * w / l
        assert meyerhof_factor_c(p, w, l) == expected


def test_meyerhof_factor_q():
    '''test function for cirsoc_402.bearing.shape.meyerhof_factor_q
    '''
    assert np.isnan(meyerhof_factor_q(np.nan, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_q(30, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_q(30, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_q(30, np.nan, 3)) == True
    assert np.isnan(meyerhof_factor_q(np.nan, 2, 3)) == True
    assert np.isnan(meyerhof_factor_q(np.nan, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_q(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(meyerhof_factor_q(30, 2, 0)) == True
    assert np.isnan(meyerhof_factor_q(np.nan, 2, 0)) == True
    assert np.isnan(meyerhof_factor_q(30, np.nan, 0)) == True
    assert np.isnan(meyerhof_factor_q(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(10, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert meyerhof_factor_q(0, w, l) == 1
    # phi > 10
    for w, l, p in itertools.product(widths, lengths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected = 1 + 0.1 * nphi * w / l
        assert meyerhof_factor_q(p, w, l) == expected
    # 0 < phi < 10
    for w, l, p in itertools.product(widths, lengths, phi):
        factor0 = meyerhof_factor_q(0, w, l)
        factor10 = meyerhof_factor_q(10, w, l)
        for p in np.linspace(1, 9, 9):
            expected = (factor10 - factor0) * p / 10 + factor0
            assert meyerhof_factor_q(p, w, l) == expected


def test_meyerhof_factor_g():
    '''test function for cirsoc_402.bearing.shape.meyerhof_factor_g
    '''
    assert np.isnan(meyerhof_factor_g(np.nan, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_g(30, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_g(30, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_g(30, np.nan, 3)) == True
    assert np.isnan(meyerhof_factor_g(np.nan, 2, 3)) == True
    assert np.isnan(meyerhof_factor_g(np.nan, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_g(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(meyerhof_factor_g(30, 2, 0)) == True
    assert np.isnan(meyerhof_factor_g(np.nan, 2, 0)) == True
    assert np.isnan(meyerhof_factor_g(30, np.nan, 0)) == True
    assert np.isnan(meyerhof_factor_g(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(10, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert meyerhof_factor_g(0, w, l) == 1
    # phi > 10
    for w, l, p in itertools.product(widths, lengths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected = 1 + 0.1 * nphi * w / l
        assert meyerhof_factor_g(p, w, l) == expected
    # 0 < phi < 10
    for w, l, p in itertools.product(widths, lengths, phi):
        factor0 = meyerhof_factor_g(0, w, l)
        factor10 = meyerhof_factor_g(10, w, l)
        for p in np.linspace(1, 9, 9):
            expected = (factor10 - factor0) * p / 10 + factor0
            assert meyerhof_factor_g(p, w, l) == expected


def test_hansen_factor_c():
    '''test function for cirsoc_402.bearing.shape.hansen_factor_c
    '''
    assert np.isnan(hansen_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, 2, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, 3)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(hansen_factor_c(np.nan, np.nan, 3)) == True

    # return nan when effective_length=0
    assert np.isnan(hansen_factor_c(30, 2, 0)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, 0)) == True
    assert np.isnan(hansen_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        expected = 0.2 * w / l
        assert hansen_factor_c(0, w, l) == expected
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        nq = bearing_factor_nq(p)
        nc = bearing_factor_nc(p)
        expected = 1 + nq / nc * w / l
        assert pytest.approx(hansen_factor_c(p, w, l), 0.01) == expected


def test_hansen_factor_q():
    '''test function for cirsoc_402.bearing.shape.hansen_factor_q
    '''
    assert np.isnan(hansen_factor_q(np.nan, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_q(30, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_q(30, 2, np.nan)) == True
    assert np.isnan(hansen_factor_q(30, np.nan, 3)) == True
    assert np.isnan(hansen_factor_q(np.nan, 2, 3)) == True
    assert np.isnan(hansen_factor_q(np.nan, 2, np.nan)) == True
    assert np.isnan(hansen_factor_q(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(hansen_factor_q(30, 2, 0)) == True
    assert np.isnan(hansen_factor_q(np.nan, 2, 0)) == True
    assert np.isnan(hansen_factor_q(30, np.nan, 0)) == True
    assert np.isnan(hansen_factor_q(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert hansen_factor_q(0, w, l) == 1
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        expected = 1 +  w / l * np.tan(np.radians(p))
        assert hansen_factor_q(p, w, l) == expected


def test_hansen_factor_g():
    '''test function for cirsoc_402.bearing.shape.hansen_factor_g
    '''
    assert np.isnan(hansen_factor_g(np.nan, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_g(30, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_g(30, 2, np.nan)) == True
    assert np.isnan(hansen_factor_g(30, np.nan, 3)) == True
    assert np.isnan(hansen_factor_g(np.nan, 2, 3)) == True
    assert np.isnan(hansen_factor_g(np.nan, 2, np.nan)) == True
    assert np.isnan(hansen_factor_g(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(hansen_factor_g(30, 2, 0)) == True
    assert np.isnan(hansen_factor_g(np.nan, 2, 0)) == True
    assert np.isnan(hansen_factor_g(30, np.nan, 0)) == True
    assert np.isnan(hansen_factor_g(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert hansen_factor_g(0, w, l) == 1
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        expected = 1 - 0.4 * w / l
        assert hansen_factor_g(p, w, l) == expected


def test_vesic_factor_c():
    '''test function for cirsoc_402.bearing.shape.vesic_factor_c
    '''
    assert np.isnan(vesic_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, 2, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, 3)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(vesic_factor_c(np.nan, np.nan, 3)) == True

    # return nan when effective_length=0
    assert np.isnan(vesic_factor_c(30, 2, 0)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, 0)) == True
    assert np.isnan(vesic_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        expected = 0.2 * w / l
        assert vesic_factor_c(0, w, l) == expected
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        nq = bearing_factor_nq(p)
        nc = bearing_factor_nc(p)
        expected = 1 + nq / nc * w / l
        assert pytest.approx(vesic_factor_c(p, w, l), 0.01) == expected


def test_vesic_factor_q():
    '''test function for cirsoc_402.bearing.shape.vesic_factor_q
    '''
    assert np.isnan(vesic_factor_q(np.nan, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_q(30, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_q(30, 2, np.nan)) == True
    assert np.isnan(vesic_factor_q(30, np.nan, 3)) == True
    assert np.isnan(vesic_factor_q(np.nan, 2, 3)) == True
    assert np.isnan(vesic_factor_q(np.nan, 2, np.nan)) == True
    assert np.isnan(vesic_factor_q(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(vesic_factor_q(30, 2, 0)) == True
    assert np.isnan(vesic_factor_q(np.nan, 2, 0)) == True
    assert np.isnan(vesic_factor_q(30, np.nan, 0)) == True
    assert np.isnan(vesic_factor_q(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert vesic_factor_q(0, w, l) == 1
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        expected = 1 +  w / l * np.tan(np.radians(p))
        assert vesic_factor_q(p, w, l) == expected


def test_vesic_factor_g():
    '''test function for cirsoc_402.bearing.shape.vesic_factor_g
    '''
    assert np.isnan(vesic_factor_g(np.nan, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_g(30, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_g(30, 2, np.nan)) == True
    assert np.isnan(vesic_factor_g(30, np.nan, 3)) == True
    assert np.isnan(vesic_factor_g(np.nan, 2, 3)) == True
    assert np.isnan(vesic_factor_g(np.nan, 2, np.nan)) == True
    assert np.isnan(vesic_factor_g(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(vesic_factor_g(30, 2, 0)) == True
    assert np.isnan(vesic_factor_g(np.nan, 2, 0)) == True
    assert np.isnan(vesic_factor_g(30, np.nan, 0)) == True
    assert np.isnan(vesic_factor_g(np.nan, np.nan, 0)) == True


    widths = np.linspace(0.5, 5, 23)
    lengths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, l in itertools.product(widths, lengths):
        assert vesic_factor_g(0, w, l) == 1
    # phi > 0
    for w, l, p in itertools.product(widths, lengths, phi):
        expected = 1 - 0.4 * w / l
        assert vesic_factor_g(p, w, l) == expected

