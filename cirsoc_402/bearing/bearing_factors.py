'''Module with the bearing factors formulas for the bearing capacity
equation
'''

from constants import  BEARINGMETHOD
from exceptions import BearingMethodError

import numpy as np

def bearing_factor_nq(phi):
    phi = np.radians(phi)
    nq = (np.exp(np.pi * np.tan(phi))) * ((np.tan(0.25 * np.pi + 0.5 * phi))**2)
    return nq


def bearing_factor_nc(phi):
    if phi == 0:
        nc = (2 + np.pi)
    else:
        nc = (bearing_factor_nq(phi) - 1) * (1 / np.tan(np.radians(phi)))
    return nc


def bearing_factor_ng(phi, method):
    nq = bearing_factor_nq(phi)
    method = method.lower()
    if method not in BEARINGMETHOD:
        raise BearingMethodError(method)
    elif method == 'vesic':
        ng = 2 * (nq + 1) * np.tan(np.radians(phi)) #Vesic (1975)
    elif method == 'hansen':
        ng = 1.5 * (nq - 1) * np.tan(np.radians(phi)) #Hansen (1970)
    elif method == 'eurocode 7':
        ng = 2 * (nq - 1) * np.tan(np.radians(phi)) #Eurocode 7 (CEN 2004)
    elif method == 'meyerhof':
        ng = (nq - 1) * np.tan(1.4 * np.radians(phi)) #Meyerhof (1963)
    return ng