'''Module with the ground inclination factors formulas for the bearing
capacity equation
'''
import numpy as np

from constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingFactorsError


def ground_inclination_factors(factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'usace':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()

    return factor_c, factor_q, factor_g


def ground_inclination_factor_c(factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
    elif factors == 'canada':
        factor_c = cirsoc_factor_c()
    elif factors == 'usace':
        factor_c = cirsoc_factor_c()

    return factor_c


def ground_inclination_factor_q(factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q()
    elif factors == 'canada':
        factor_q = cirsoc_factor_q()
    elif factors == 'usace':
        factor_q = cirsoc_factor_q()

    return factor_q


def ground_inclination_factor_g( factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_g = cirsoc_factor_g()
    elif factors == 'usace':
        factor_g = cirsoc_factor_g()

    return factor_g


def cirsoc_factor_c():
    return 1


def cirsoc_factor_q():
    return 1


def cirsoc_factor_g():
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