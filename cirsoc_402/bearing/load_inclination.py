'''Module with the load inclination factors formulas for the bearing
capacity equation
'''
import numpy as np
from bearing_factors import bearing_factor_nc

from constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingFactorsError


def load_inclination_factors(phi, cohesion, width, length, vertical_load,
                             horizontal_load, load_theta,
                             factors=DEFAULTBEARINGFACTORS):

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
        factor_c = canada_factor_c(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
        factor_g = canada_factor_g(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
    elif factors == 'usace':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()

    return factor_c, factor_q, factor_g


def load_inclination_factor_c(phi, cohesion, width, length, vertical_load,
                              horizontal_load, load_theta,
                              factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
    elif factors == 'usace':
        factor_c = canada_factor_c()

    return factor_c


def load_inclination_factor_q(phi, cohesion, width, length, vertical_load,
                              horizontal_load, load_theta,
                              factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q()
    elif factors == 'canada':
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
    elif factors == 'usace':
        factor_q = canada_factor_q()

    return factor_q


def load_inclination_factor_g(phi, cohesion, width, length, vertical_load,
                              horizontal_load, load_theta,
                              factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_g = canada_factor_g(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_theta)
    elif factors == 'usace':
        factor_g = canada_factor_g()

    return factor_g


def cirsoc_factor_c():
    return 1


def cirsoc_factor_q():
    return 1


def cirsoc_factor_g():
    return 1


def canada_factor_c(phi, cohesion, width, length, vertical_load, horizontal_load, load_theta):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(load_theta):
        return np.nan
    if phi<0 or np.isnan(phi) or cohesion<0 or np.isnan(cohesion):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan
    
    factor_m = canada_factor_m(width, length, load_theta)
    
    if phi==0:
        return 1 - factor_m * horizontal_load / (width * length * cohesion * bearing_factor_nc(phi))
    else:
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load, horizontal_load, load_theta)
        factor = factor_q - (1 - factor_q) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))
        return factor


def canada_factor_q(phi, cohesion, width, length, vertical_load, horizontal_load, load_theta):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(load_theta):
        return np.nan
    if phi<0 or np.isnan(phi) or cohesion<0 or np.isnan(cohesion):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan

    factor_m = canada_factor_m(width, length, load_theta)
    factor = (1 - horizontal_load / (vertical_load + width * length * cohesion / np.tan(np.radians(phi)))) ** factor_m
    return factor


def canada_factor_g(phi, cohesion, width, length, vertical_load, horizontal_load, load_theta):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(load_theta):
        return np.nan
    if phi<0 or np.isnan(phi) or cohesion<0 or np.isnan(cohesion):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan

    factor_m = canada_factor_m(width, length, load_theta)
    factor = (1 - horizontal_load / (vertical_load + width * length * cohesion / np.tan(np.radians(phi)))) ** (factor_m + 1)
    return factor


def canada_factor_m(width, length, load_theta):
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan
    if np.isnan(load_theta):
        return np.nan
    mb = (2 + width / length) / (1 + width / length)
    ml = (2 + length / width) / (1 + length / width)
    return ml * np.cos(np.radians(load_theta))**2 + mb * np.sin(np.radians(load_theta))**2


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1
