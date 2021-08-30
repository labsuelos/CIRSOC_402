'''Module with the depth factors formulas for the bearing capacity
equation
'''
import numpy as np
from bearing_factors import bearing_factor_nc

from constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingFactorsError

def depth_factors(depth, phi, width, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(depth, width, phi)
        factor_q = cirsoc_factor_q(depth, width, phi)
        factor_g = cirsoc_factor_g(phi)
    elif factors == 'canada':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()

    return factor_c, factor_q, factor_g


def depth_factor_c(depth, phi, width, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(depth, width, phi)
    elif factors == 'canada':
        factor_c = canada_factor_c()
    elif factors == 'usace':
        factor_c = canada_factor_c()

    return factor_c


def depth_factor_q(depth, phi, width, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q(depth, width, phi)
    elif factors == 'canada':
        factor_q = canada_factor_q()
    elif factors == 'usace':
        factor_q = canada_factor_q()

    return factor_q


def depth_factor_g(depth, phi, width, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g(phi)
    elif factors == 'canada':
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_g = canada_factor_g()

    return factor_g


def cirsoc_factor_c(depth, width, phi):
    factor_q = cirsoc_factor_q(depth, width, phi)
    if phi > 0:
        factor = factor_q- ((1 - factor_q) / (bearing_factor_nc(phi) * np.tan(np.radians(phi))))
    else:
        factor = 1 + 0.33 * np.arctan(depth/width)
    return factor


def cirsoc_factor_q(depth, width, phi):
    factor = 1 + 2 * np.tan(np.radians(phi)) * ((1 - np.sin(np.radians(phi))) ** 2) * np.arctan(depth/width)
    return factor


def cirsoc_factor_g(phi):
    return 1


def canada_factor_c():
    return 1


def canada_factor_q():
    return 1


def canada_factor_g():
    return 1


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1

