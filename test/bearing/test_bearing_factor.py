'''test module for the bearing factors in the bearing capacity
equation for shallow foundations'''


import numpy as np
import pytest

from cirsoc_402.bearing.bearing_factors import bearing_factor_nq
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc
from cirsoc_402.bearing.bearing_factors import bearing_factor_ng

from cirsoc_402.constants import BEARINGMETHOD


def test_bearing_factor_nq():
    '''Test function for cirsoc_402.bearing.bearing_factors.bearing_factor_nq
    '''

    # nan behavior
    assert np.isnan(bearing_factor_nq(np.nan)) == True

    # USACE EM 1110-1-1905 Bearing capacity of soils, Table 4-4
    phi = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
           34, 36, 38, 40, 42, 44, 46, 48, 50]
    expected = [1, 1.2, 1.43, 1.72, 2.06, 2.47, 2.97, 3.59, 4.34, 5.26, 6.4, 
                7.82, 9.6, 11.85, 14.72, 18.4, 23.18, 29.44, 37.75, 48.93,
                64.19, 85.37, 115.31, 158.5, 222.3, 319.05]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nq(fric), 0.01)

    # Canadian Foundation Engineerig Manual Table 10-1
    phi = [0, 10, 15, 20, 21, 23, 23, 24]
    expected = [1, 2.5, 3.9, 6.4, 7.1, 7.8, 8.7, 9.6] 
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nq(fric), 0.1)
    
    phi = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    expected = [11, 12, 13, 15, 16, 18, 21, 23, 26, 29, 33, 38, 43, 49, 56, 64]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nq(fric), 1)


def test_bearing_factor_nc():
    '''Test function for cirsoc_402.bearing.bearing_factors.bearing_factor_nc
    '''

    # nan behavior
    assert np.isnan(bearing_factor_nc(np.nan)) == True

    # USACE EM 1110-1-1905 Bearing capacity of soils, Table 4-4
    phi = [0, 2, 4, 6, 8, 10,12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
           34, 36, 38, 40, 42, 44, 46, 48, 50]
    expected = [5.14, 5.63, 6.18, 6.81, 7.53, 8.34, 9.28, 10.37, 11.63, 13.1,
                14.83, 16.88, 19.32, 22.25, 25.8, 30.14, 35.49, 42.16, 50.59,
                61.35, 75.31, 93.71, 118.37, 152.1, 199.26, 266.88]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nc(fric), 0.01)

    # Canadian Foundation Engineerig Manual Table 10-1
    phi = [0, 10]
    expected = [5.1, 8.3] 
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nc(fric), 0.1)
    
    phi = [15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
           36, 37, 38, 39, 40]
    expected = [11, 15, 16, 17, 18, 19, 21, 22, 24, 26, 28, 30, 33, 35, 39, 42,
                46, 51, 56, 61, 68, 75]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_nc(fric), 1)


def test_bearing_factor_ng():
    '''Test function for cirsoc_402.bearing.bearing_factors.bearing_factor_ng
    '''

    # nan behavior
    for method in BEARINGMETHOD:
        assert np.isnan(bearing_factor_ng(np.nan, method)) == True
        assert np.isnan(bearing_factor_ng(np.nan, method, ground_inclination=np.nan)) == True

    for method in ['Eurocode 7',  'Meyerhof',  'Vesic',  'Hansen', 'Canada']:
        assert np.isnan(bearing_factor_ng(np.nan, method)) == True
        assert np.isnan(bearing_factor_ng(np.nan, method, ground_inclination=np.nan)) == True

    for method in ['canada', 'Canada']: 
        assert np.isnan(bearing_factor_ng(np.nan, method, ground_inclination=20)) == True
        assert np.isnan(bearing_factor_ng(5, method, ground_inclination=np.nan)) == True
        assert np.isnan(bearing_factor_ng(30, method, ground_inclination=np.nan)) == True

    # unsuported bearing method returns nan
    assert np.isnan(bearing_factor_ng(10, 'xasdq')) == True

    # USACE EM 1110-1-1905 Bearing capacity of soils, Table 4-4
    assert bearing_factor_ng(0, 'meyerhof') == 0
    assert bearing_factor_ng(0, 'Meyerhof') == 0
    phi = [0, 2, 4, 6, 8, 10,12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
           34, 36, 38, 40, 42, 44, 46, 48, 50]
    expected = [0, 0.01, 0.04, 0.11, 0.21, 0.37, 0.6, 0.92, 1.37, 2, 2.87, 4.07,
                5.72, 8, 11.19, 15.67, 22.02, 31.15, 44.43, 64.07, 93.69,
                139.32, 211.41, 328.73, 526.44, 873.84]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_ng(fric, 'meyerhof'), abs=0.01, rel=0.01)
        assert exp == pytest.approx(bearing_factor_ng(fric, 'Meyerhof'), abs=0.01, rel=0.01)

    assert bearing_factor_ng(0, 'hansen') == 0
    assert bearing_factor_ng(0, 'Hansen') == 0
    phi = [0, 2, 4, 6, 8, 10,12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
           34, 36, 38, 40, 42, 44, 46, 48, 50]
    expected = [0, 0.01, 0.05, 0.11, 0.22, 0.39, 0.63, 0.97, 1.43, 2.08, 2.95,
                4.13, 5.75, 7.94, 10.94, 15.07, 20.79, 28.77, 40.05, 56.17,
                79.54, 113.95, 165.58, 244.64, 368.88, 568.56]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_ng(fric, 'hansen'), abs=0.01, rel=0.01)
        assert exp == pytest.approx(bearing_factor_ng(fric, 'Hansen'), abs=0.01, rel=0.01)
    

    assert bearing_factor_ng(0, 'vesic') == 0
    assert bearing_factor_ng(0, 'Vesic') == 0
    phi = [0, 2, 4, 6, 8, 10,12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 
           34, 36, 38, 40, 42, 44, 46, 48, 50]
    expected = [0, 0.15, 0.34, 0.57, 0.86, 1.22, 1.69, 2.29, 3.06, 4.07, 5.39,
                7.13, 9.44, 12.54, 16.72, 22.4, 30.21, 41.06, 56.31, 78.02,
                109.41, 155.54, 224.63, 330.33, 495.99, 762.85]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_ng(fric, 'vesic'), abs=0.01, rel=0.01)
        assert exp == pytest.approx(bearing_factor_ng(fric, 'Vesic'), abs=0.01, rel=0.01)
    
    # Canadian Foundation Engineerig Manual Table 10-1
    phi = [0, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27]
    expected = [0, 0.6, 1.3, 3.0, 3.6, 4.2, 5.0, 5.9, 7.0, 8.2, 9.7]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_ng(fric, 'canada'), abs=0.1)
        assert exp == pytest.approx(bearing_factor_ng(fric, 'Canada'), abs=0.1)

    phi = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    expected = [11, 14, 16, 19, 22, 27, 31, 37, 44, 52, 61, 73, 86]
    for fric, exp in zip(phi, expected):
        assert exp == pytest.approx(bearing_factor_ng(fric, 'canada'), abs=1)
        assert exp == pytest.approx(bearing_factor_ng(fric, 'Canada'), abs=1)

    # Canadian Foundation Engineerig Manual Table 10-2 inclined base
    ground_inc = np.linspace(0, 45, 46)
    expected = -2 * np.sin(np.radians(ground_inc))
    for incl, exp in zip(ground_inc, expected):
        assert exp == pytest.approx(bearing_factor_ng(0, 'canada', ground_inclination=incl), abs=0.1)

