'''Module with the shape factors formulas for the bearing capacity
equation
'''
import numpy as np
from bearing_factors import bearing_factor_nq, bearing_factor_nc

from constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingFactorsError

def shape_factors(shape, phi, width=np.nan, length=np.nan, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    
    if shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(shape, phi, width=width, length=length)
        factor_q = cirsoc_factor_q(shape, phi, width=width, length=length)
        factor_g = cirsoc_factor_g(shape, width=width, length=length)
    elif factors == 'canada':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()

    return factor_c, factor_q, factor_g


def shape_factor_c(shape, phi, width=np.nan, length=np.nan, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    
    if shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(shape, phi, width=width, length=length)
    elif factors == 'canada':
        factor_c = canada_factor_c()
    elif factors == 'usace':
        factor_c = canada_factor_c()

    return factor_c


def shape_factor_q(shape, phi, width=np.nan, length=np.nan, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    
    if shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q(shape, phi, width=width, length=length)
    elif factors == 'canada':
        factor_q = canada_factor_q()
    elif factors == 'usace':
        factor_q = canada_factor_q()

    return factor_q


def shape_factor_g(shape, phi, width=np.nan, length=np.nan, factors=DEFAULTBEARINGFACTORS):

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    
    if shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g(shape, width=width, length=length)
    elif factors == 'canada':
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_g = canada_factor_g()

    return factor_g


def cirsoc_factor_c(shape, phi, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (width/length) * (bearing_factor_nq(phi) \
                 / bearing_factor_nc(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    return factor


def cirsoc_factor_q(shape, phi, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (width / length) * np.tan(np.radians(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * np.tan(np.radians(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * np.tan(np.radians(phi))        
    return factor


def cirsoc_factor_g(shape, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 - 0.4 * (width / length)
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 - 0.4 * (1)
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 - 0.4 * (0)
    return factor


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