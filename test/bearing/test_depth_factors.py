'''test module for the depth factors in the bearing capacity
equation for shallow foundations'''


import itertools
import numpy as np
import pytest

from cirsoc_402.constants import BEARINGFACTORS
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc

from cirsoc_402.bearing.depth import depth_factors
from cirsoc_402.bearing.depth import depth_factor_c
from cirsoc_402.bearing.depth import depth_factor_q
from cirsoc_402.bearing.depth import depth_factor_g

from cirsoc_402.bearing.depth import cirsoc_factor_c
from cirsoc_402.bearing.depth import cirsoc_factor_g
from cirsoc_402.bearing.depth import cirsoc_factor_q

from cirsoc_402.bearing.depth import canada_factor_k
from cirsoc_402.bearing.depth import canada_factor_c
from cirsoc_402.bearing.depth import canada_factor_g
from cirsoc_402.bearing.depth import canada_factor_q

from cirsoc_402.bearing.depth import meyerhof_factor_c
from cirsoc_402.bearing.depth import meyerhof_factor_g
from cirsoc_402.bearing.depth import meyerhof_factor_q

from cirsoc_402.bearing.depth import hansen_factor_c
from cirsoc_402.bearing.depth import hansen_factor_g
from cirsoc_402.bearing.depth import hansen_factor_q
from cirsoc_402.bearing.depth import hansen_factor_k

from cirsoc_402.bearing.depth import vesic_factor_c
from cirsoc_402.bearing.depth import vesic_factor_g
from cirsoc_402.bearing.depth import vesic_factor_q
from cirsoc_402.bearing.depth import vesic_factor_k



def test_depth_factors():
    '''test function for cirsoc_402.bearing.depth.depth_factors
    '''

    # nan imput
    factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, np.nan)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, np.nan, np.nan)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, 2, np.nan)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(5, np.nan, 5)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    
    factor_c, factor_q, factor_g = depth_factors(np.nan, 2, 3)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(np.nan, 2, np.nan)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    
    factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, 3)
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    
    # function must retunr nan when widht=0 because division by zero
    for factors in BEARINGFACTORS:
        factor_c, factor_q, factor_g = depth_factors(3, 2, 0, factors=factors)
        assert np.isnan(factor_c) == True
        assert np.isnan(factor_q) == True
        assert np.isnan(factor_g) == True

        factor_c, factor_q, factor_g = depth_factors(np.nan, 2, 0, factors=factors)
        assert np.isnan(factor_c) == True
        assert np.isnan(factor_q) == True
        assert np.isnan(factor_g) == True

        factor_c, factor_q, factor_g = depth_factors(3, np.nan, 0, factors=factors)
        assert np.isnan(factor_c) == True
        assert np.isnan(factor_q) == True
        assert np.isnan(factor_g) == True

        factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, 0, factors=factors)
        assert np.isnan(factor_c) == True
        assert np.isnan(factor_q) == True
        assert np.isnan(factor_g) == True

    # unsuported factor family
    factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, np.nan, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, np.nan, np.nan, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, 2, np.nan, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(5, np.nan, 5, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    
    factor_c, factor_q, factor_g = depth_factors(np.nan, 2, 3, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(np.nan, 2, np.nan, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    
    factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, 3, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, 2, 0, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(np.nan, 2, 0, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(3, np.nan, 0, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True

    factor_c, factor_q, factor_g = depth_factors(np.nan, np.nan, 0, factors='xxxx')
    assert np.isnan(factor_c) == True
    assert np.isnan(factor_q) == True
    assert np.isnan(factor_g) == True
    

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 40, 21)
    # Cirsoc
    for w, d, p in itertools.product(widths, depths, phi):
        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='cirsoc')
        expected_c = cirsoc_factor_c(p, d, w)
        expected_q = cirsoc_factor_q(p, d, w)
        expected_g = cirsoc_factor_g()
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='CIRSOC')
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

    # Canada
    for w, d, p in itertools.product(widths, depths, phi):
        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='canada')
        expected_c = canada_factor_c(p, d, w)
        expected_q = canada_factor_q(p, d, w)
        expected_g = canada_factor_g()
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='Canada')
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g
    
    # Meyerhof
    for w, d, p in itertools.product(widths, depths, phi):
        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='meyerhof')
        expected_c = meyerhof_factor_c(p, d, w)
        expected_q = meyerhof_factor_q(p, d, w)
        expected_g = meyerhof_factor_g(p, d, w)
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='Meyerhof')
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g
    
    # Hansen
    for w, d, p in itertools.product(widths, depths, phi):
        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='hansen')
        expected_c = hansen_factor_c(p, d, w)
        expected_q = hansen_factor_q(p, d, w)
        expected_g = hansen_factor_g()
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='Hansen')
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g
    
    # Vesic
    for w, d, p in itertools.product(widths, depths, phi):
        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='vesic')
        expected_c = vesic_factor_c(p, d, w)
        expected_q = vesic_factor_q(p, d, w)
        expected_g = vesic_factor_g()
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g

        factor_c, factor_q, factor_g = depth_factors(p, d, w, factors='Vesic')
        assert factor_c == expected_c
        assert factor_q == expected_q
        assert factor_g == expected_g


def test_depth_factor_c():
    '''test function for cirsoc_402.bearing.depth.depth_factor_c
    '''

    # nan imput
    assert np.isnan(depth_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(depth_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(depth_factor_c(30, 2, np.nan)) == True
    assert np.isnan(depth_factor_c(30, np.nan, 3)) == True
    assert np.isnan(depth_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(depth_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(depth_factor_c(np.nan, np.nan, 3)) == True
    
    # function must retunr nan when widht=0 because division by zero
    for factors in BEARINGFACTORS:
        assert np.isnan(depth_factor_c(30, 2, 0, factors=factors)) == True
        assert np.isnan(depth_factor_c(np.nan, 2, 0, factors=factors)) == True
        assert np.isnan(depth_factor_c(30, np.nan, 0, factors=factors)) == True
        assert np.isnan(depth_factor_c(np.nan, np.nan, 0, factors=factors)) == True

    # unsuported factor family
    assert np.isnan(depth_factor_c(np.nan, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(30, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(30, 2, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(30, np.nan, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, 2, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, 2, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, np.nan, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(30, 2, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, 2, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(30, np.nan, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_c(np.nan, np.nan, 0, factors='xxxx')) == True
    

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 40, 21)
    # Cirsoc
    for w, d, p in itertools.product(widths, depths, phi):
        expected = cirsoc_factor_c(p, d, w)
        computed = depth_factor_c(p, d, w, factors='cirsoc')
        assert  computed == expected
        computed = depth_factor_c(p, d, w, factors='CIRSOC')
        assert  computed == expected

    # Canada
    for w, d, p in itertools.product(widths, depths, phi):
        expected = canada_factor_c(p, d, w)
        computed = depth_factor_c(p, d, w, factors='canada')
        assert  computed == expected
        computed = depth_factor_c(p, d, w, factors='Canada')
        assert  computed == expected
    
    # Meyerhof
    for w, d, p in itertools.product(widths, depths, phi):
        expected = meyerhof_factor_c(p, d, w)
        computed = depth_factor_c(p, d, w, factors='meyerhof')
        assert  computed == expected
        computed = depth_factor_c(p, d, w, factors='Meyerhof')
        assert  computed == expected
    
    # Hansen
    for w, d, p in itertools.product(widths, depths, phi):
        expected = hansen_factor_c(p, d, w)
        computed = depth_factor_c(p, d, w, factors='hansen')
        assert  computed == expected
        computed = depth_factor_c(p, d, w, factors='Hansen')
        assert  computed == expected

    # Vesic
    for w, d, p in itertools.product(widths, depths, phi):
        expected = vesic_factor_c(p, d, w)
        computed = depth_factor_c(p, d, w, factors='vesic')
        assert  computed == expected
        computed = depth_factor_c(p, d, w, factors='vesic')
        assert  computed == expected


def test_depth_factor_q():
    '''test function for cirsoc_402.bearing.depth.depth_factor_c
    '''

    # nan imput
    assert np.isnan(depth_factor_q(np.nan, np.nan, np.nan)) == True
    assert np.isnan(depth_factor_q(30, np.nan, np.nan)) == True
    assert np.isnan(depth_factor_q(30, 2, np.nan)) == True
    assert np.isnan(depth_factor_q(30, np.nan, 3)) == True
    assert np.isnan(depth_factor_q(np.nan, 2, 3)) == True
    assert np.isnan(depth_factor_q(np.nan, 2, np.nan)) == True
    assert np.isnan(depth_factor_q(np.nan, np.nan, 3)) == True
    
    # function must retunr nan when widht=0 because division by zero
    for factors in BEARINGFACTORS:
        assert np.isnan(depth_factor_q(30, 2, 0, factors=factors)) == True
        assert np.isnan(depth_factor_q(np.nan, 2, 0, factors=factors)) == True
        assert np.isnan(depth_factor_q(30, np.nan, 0, factors=factors)) == True
        assert np.isnan(depth_factor_q(np.nan, np.nan, 0, factors=factors)) == True

    # unsuported factor family
    assert np.isnan(depth_factor_q(np.nan, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(30, np.nan, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(30, 2, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(30, np.nan, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, 2, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, 2, np.nan, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, np.nan, 3, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(30, 2, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, 2, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(30, np.nan, 0, factors='xxxx')) == True
    assert np.isnan(depth_factor_q(np.nan, np.nan, 0, factors='xxxx')) == True
    

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 40, 21)
    # Cirsoc
    for w, d, p in itertools.product(widths, depths, phi):
        expected = cirsoc_factor_q(p, d, w)
        computed = depth_factor_q(p, d, w, factors='cirsoc')
        assert  computed == expected
        computed = depth_factor_q(p, d, w, factors='CIRSOC')
        assert  computed == expected

    # Canada
    for w, d, p in itertools.product(widths, depths, phi):
        expected = canada_factor_q(p, d, w)
        computed = depth_factor_q(p, d, w, factors='canada')
        assert  computed == expected
        computed = depth_factor_q(p, d, w, factors='Canada')
        assert  computed == expected
    
    # Meyerhof
    for w, d, p in itertools.product(widths, depths, phi):
        expected = meyerhof_factor_q(p, d, w)
        computed = depth_factor_q(p, d, w, factors='meyerhof')
        assert  computed == expected
        computed = depth_factor_q(p, d, w, factors='Meyerhof')
        assert  computed == expected
    
    # Hansen
    for w, d, p in itertools.product(widths, depths, phi):
        expected = hansen_factor_q(p, d, w)
        computed = depth_factor_q(p, d, w, factors='hansen')
        assert  computed == expected
        computed = depth_factor_q(p, d, w, factors='Hansen')
        assert  computed == expected
    
    # Vesic
    for w, d, p in itertools.product(widths, depths, phi):
        expected = vesic_factor_q(p, d, w)
        computed = depth_factor_q(p, d, w, factors='vesic')
        assert  computed == expected
        computed = depth_factor_q(p, d, w, factors='Vesic')
        assert  computed == expected


def test_depth_factors_g():
    pass


def test_canada_factor_k():
    '''test function for cirsoc_402.bearing.depth.canada_factor_k
    '''

    assert np.isnan(canada_factor_k(np.nan, np.nan)) == True
    assert np.isnan(canada_factor_k(np.nan, 10)) == True
    assert np.isnan(canada_factor_k(4, np.nan)) == True

    # return nan when width=0
    assert np.isnan(canada_factor_k(2, 0)) == True
    assert np.isnan(canada_factor_k(np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 46)
    depths = np.linspace(0.5, 5, 46)
    for w, d in itertools.product(widths, depths):
        if d <= w:
            assert canada_factor_k(d, w) == d / w
        else:
            assert canada_factor_k(d, w) == np.arctan(d / w)


def test_canada_factor_c():
    '''test function for cirsoc_402.bearing.depth.canada_factor_c
    '''
    assert np.isnan(canada_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(canada_factor_c(30, 2, np.nan)) == True
    assert np.isnan(canada_factor_c(30, np.nan, 3)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(canada_factor_c(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(canada_factor_c(30, 2, 0)) == True
    assert np.isnan(canada_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(canada_factor_c(30, np.nan, 0)) == True
    assert np.isnan(canada_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        kfactor = canada_factor_k(d, w)
        expected = 1 + 0.4 * kfactor
        assert canada_factor_c(0, d, w) == expected
    # phi > 0
    for w, d, p in itertools.product(widths, depths, phi):
        sqd = canada_factor_q(p, d, w)
        nc = bearing_factor_nc(p)
        expected = sqd - (1 - sqd) / (nc * np.tan(np.radians(p)))
        assert canada_factor_c(p, d, w) == expected


def test_canada_factor_q():
    '''test function for cirsoc_402.bearing.depth.canada_factor_g
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
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    for w, d, p in itertools.product(widths, depths, phi):
        kfactor = canada_factor_k(d, w)
        expected = 1 + 2 * np.tan(np.radians(p)) * (1 - np.sin(np.radians(p)))**2 * kfactor
        assert canada_factor_q(p, d, w) == expected


def test_canada_factor_g():
    '''test function for cirsoc_402.bearing.depth.canada_factor_g
    '''
    assert canada_factor_g() == 1


def test_meyerhof_factor_c():
    '''test function for cirsoc_402.bearing.depth.meyerhof_factor_c
    '''
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, 3)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(meyerhof_factor_c(30, 2, 0)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(meyerhof_factor_c(30, np.nan, 0)) == True
    assert np.isnan(meyerhof_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    for w, d, p in itertools.product(widths, depths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected = 1 + 0.2 * nphi**(1/2) * d / w
        assert meyerhof_factor_c(p, d, w) == expected


def test_meyerhof_factor_q():
    '''test function for cirsoc_402.bearing.depth.meyerhof_factor_g
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
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(10, 45, 20)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        assert meyerhof_factor_q(0, d, w) == 1
    # phi > 10
    for w, d, p in itertools.product(widths, depths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected =  1 + 0.1 * nphi**(1/2) * d / w
        assert meyerhof_factor_q(p, d, w) == expected
    # 0 < phi < 10
    for w, d in itertools.product(widths, depths):
        factor0 = meyerhof_factor_q(0, d, w)
        factor10 = meyerhof_factor_q(10, d, w)
        for p in np.linspace(1, 9, 9):
            expected = (factor10 - factor0) * p / 10 + factor0
            assert meyerhof_factor_q(p, d, w) == expected


def test_meyerhof_factor_g():
    '''test function for cirsoc_402.bearing.depth.meyerhof_factor_g
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
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(10, 45, 20)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        assert meyerhof_factor_g(0, d, w) == 1
    # phi > 10
    for w, d, p in itertools.product(widths, depths, phi):
        nphi = np.tan(np.radians(45 + p / 2))**2
        expected =  1 + 0.1 * nphi**(1/2) * d / w
        assert meyerhof_factor_g(p, d, w) == expected
    # 0 < phi < 10
    for w, d in itertools.product(widths, depths):
        factor0 = meyerhof_factor_g(0, d, w)
        factor10 = meyerhof_factor_g(10, d, w)
        for p in np.linspace(1, 9, 9):
            expected = (factor10 - factor0) * p / 10 + factor0
            assert meyerhof_factor_g(p, d, w) == expected


def test_hansen_factor_k():
    '''test function for cirsoc_402.bearing.depth.hansen_factor_k
    '''

    assert np.isnan(hansen_factor_k(np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_k(np.nan, 10)) == True
    assert np.isnan(hansen_factor_k(4, np.nan)) == True

    # return nan when width=0
    assert np.isnan(hansen_factor_k(2, 0)) == True
    assert np.isnan(hansen_factor_k(np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 46)
    depths = np.linspace(0.5, 5, 46)
    for w, d in itertools.product(widths, depths):
        if d <= w:
            assert hansen_factor_k(d, w) == d / w
        else:
            assert hansen_factor_k(d, w) == np.arctan(d / w)


def test_hansen_factor_c():
    '''test function for cirsoc_402.bearing.depth.hansen_factor_c
    '''
    assert np.isnan(hansen_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, 2, np.nan)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, 3)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(hansen_factor_c(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(hansen_factor_c(30, 2, 0)) == True
    assert np.isnan(hansen_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(hansen_factor_c(30, np.nan, 0)) == True
    assert np.isnan(hansen_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        kfactor = hansen_factor_k(d, w)
        expected =  0.4 * kfactor
        assert hansen_factor_c(0, d, w) == expected
    # phi > 0
    for w, d, p in itertools.product(widths, depths, phi):
        kfactor = hansen_factor_k(d, w)
        expected = 1 + 0.4 * kfactor
        assert hansen_factor_c(p, d, w) == expected


def test_hansen_factor_q():
    '''test function for cirsoc_402.bearing.depth.hansen_factor_g
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
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        assert hansen_factor_q(0, d, w) == 1
    for w, d, p in itertools.product(widths, depths, phi):
        kfactor = hansen_factor_k(d, w)
        expected = 1 + 2 * np.tan(np.radians(p)) * (1 - np.sin(np.radians(p)))**2 * kfactor
        assert hansen_factor_q(p, d, w) == expected


def test_hansen_factor_g():
    '''test function for cirsoc_402.bearing.depth.hansen_factor_g
    '''
    assert hansen_factor_g() == 1


def test_vesic_factor_c():
    '''test function for cirsoc_402.bearing.depth.vesic_factor_c
    '''
    assert np.isnan(vesic_factor_c(np.nan, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, 2, np.nan)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, 3)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, 3)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, np.nan)) == True
    assert np.isnan(vesic_factor_c(np.nan, np.nan, 3)) == True

    # return nan when width=0
    assert np.isnan(vesic_factor_c(30, 2, 0)) == True
    assert np.isnan(vesic_factor_c(np.nan, 2, 0)) == True
    assert np.isnan(vesic_factor_c(30, np.nan, 0)) == True
    assert np.isnan(vesic_factor_c(np.nan, np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 23)
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(0, 45, 23)
    # phi > 0
    for w, d, p in itertools.product(widths, depths, phi):
        kfactor = vesic_factor_k(d, w)
        expected = 1 + 0.4 * kfactor
        assert vesic_factor_c(p, d, w) == expected


def test_vesic_factor_q():
    '''test function for cirsoc_402.bearing.depth.vesic_factor_g
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
    depths = np.linspace(0.5, 5, 23)
    phi = np.linspace(1, 45, 23)
    # phi = 0
    for w, d in itertools.product(widths, depths):
        assert vesic_factor_q(0, d, w) == 1
    for w, d, p in itertools.product(widths, depths, phi):
        kfactor = vesic_factor_k(d, w)
        expected = 1 + 2 * np.tan(np.radians(p)) * (1 - np.sin(np.radians(p)))**2 * kfactor
        assert vesic_factor_q(p, d, w) == expected


def test_vesic_factor_g():
    '''test function for cirsoc_402.bearing.depth.vesic_factor_g
    '''
    assert vesic_factor_g() == 1


def test_vesic_factor_k():
    '''test function for cirsoc_402.bearing.depth.vesic_factor_k
    '''

    assert np.isnan(vesic_factor_k(np.nan, np.nan)) == True
    assert np.isnan(vesic_factor_k(np.nan, 10)) == True
    assert np.isnan(vesic_factor_k(4, np.nan)) == True

    # return nan when width=0
    assert np.isnan(vesic_factor_k(2, 0)) == True
    assert np.isnan(vesic_factor_k(np.nan, 0)) == True

    widths = np.linspace(0.5, 5, 46)
    depths = np.linspace(0.5, 5, 46)
    for w, d in itertools.product(widths, depths):
        if d <= w:
            assert vesic_factor_k(d, w) == d / w
        else:
            assert vesic_factor_k(d, w) == np.arctan(d / w)
