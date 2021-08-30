'''Module with the functions necessary to compute the bearing capacity.
'''

import numpy as np

from constants import LANGUAGE
from constants import BEARINGSHAPE, BEARINGMETHOD, BEARINGFACTORS
from exceptions import BearingShapeError
from exceptions import BearingMethodError
from exceptions import BearingWidthError
from exceptions import BearingLengthError
from exceptions import BearingFactorsError


def bearingcapacity(shape, method, gamma_Ng, gamma_Nq, phi, cohesion, depth,
                    load, width, length=np.nan, base_inclination=np.nan
                    vertical_load=np.nan, horizontal_load=np.nan,
                    factors='cirsoc204'):
    
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)

    if method not in BEARINGMETHOD:
        raise BearingMethodError(method)
    
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)

    qu_c = bearing_c(shape, method, phi, cohesion, depth, width, length=length,
              base_inclination=base_inclination,
              vertical_load=vertical_load, horizontal_load=horizontal_load,
              factors=factors)
    qu_q = bearing_q(shape, gamma_Nq, phi, depth, load, width, length=length,
                     base_inclination=base_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load, factors=factors)
    qu_g = bearing_g(shape, method, gamma_Ng, phi, width, length=length,
                     base_inclination=base_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load, factors=factors)

    return qu_c + qu_q + qu_g


def bearing_c(shape, method, phi, cohesion, depth, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, factors='cirsoc204'):
    qu_c = cohesion * nc(phi) * c_cs(shape, phi, width=width, length=length) * c_cd(depth, width, phi)
    return qu_c

def bearing_q(shape, gamma_Nq, phi, depth, load, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, factors='cirsoc204'):
    qu_q = (load + depth * gamma_Nq) * nq(phi) * c_qs(shape, phi, width=width, length=length) * c_qd(depth, width, phi)
    return qu_q

def bearing_g(shape, method, gamma_Ng, phi, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, factors='cirsoc204'):
    if shape == ['rectangle', 'rectangulo']:
        qu_g_width = 0.5 * gamma_Ng * width * ng(phi, method) * c_gs(shape, width=width, length=length) * c_gd(phi)
        qu_g_length = 0.5 * gamma_Ng * length * ng(phi, method) * c_gs(shape, width=width, length=length) * c_gd(phi)
        qu_g = np.minimum(qu_g_width, qu_g_length)
    else:
        qu_g = 0.5 * gamma_Ng * width * ng(phi, method) * c_gs(shape, width=width, length=length) * c_gd(phi)
    return qu_g


def bearing_factor_nq(phi):
    phi = np.radians(phi)
    nq = (np.exp(np.pi * np.tan(phi))) * ((np.tan(0.25 * np.pi + 0.5 * phi)) ** 2)
    return nq


def bearing_factor_nc(phi):
    if phi == 0:
        nc = (2 + np.pi)
    else:
        nc = (nq(phi) - 1) * (1 / np.tan(np.radians(phi)))
    return nc


def bearing_factor_ng(phi, method):
    method = method.lower()
    if method not in BEARINGMETHOD:
        raise BearingMethodError(method)
    elif method == 'vesic':
        ng = 2 * (nq(phi) + 1) * np.tan(np.radians(phi)) #Vesic (1975)
    elif method == 'hansen':
        ng = 1.5 * (nq(phi) - 1) * np.tan(np.radians(phi)) #Hansen (1970)
    elif method == 'eurocode 7':
        ng = 2 * (nq(phi) - 1) * np.tan(np.radians(phi)) #Eurocode 7 (CEN 2004)
    elif method == 'meyerhof':
        ng = (nq(phi) - 1) * np.tan(1.4 * np.radians(phi)) #Meyerhof (1963)
    return ng


def shape_factor_c(shape, phi, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    elif shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
        c_cs = 1 + (width/length) * (Nq(phi) / Nc(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        c_cs = 1 + (1) * (Nq(phi) / Nc(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        c_cs = 1 + (0) * (Nq(phi) / Nc(phi))
    return c_cs


def shape_factor_q(shape, phi, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    elif shape in ['rectangle', 'rectangulo']:
        c_qs = 1 + (width / length) * np.tan(np.radians(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        c_qs = 1 + (1) * np.tan(np.radians(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        c_qs = 1 + (0) * np.tan(np.radians(phi))        
    return c_qs


def shape_factor_g(shape, width=np.nan, length=np.nan):
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    elif shape in ['rectangle', 'rectangulo']:
        c_gs = 1 - 0.4 * (width / length)
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        c_gs = 1 - 0.4 * (1)
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        c_gs = 1 - 0.4 * (0)
    return c_gs


def depth_factor_q(depth, width, phi):
    c_qd = 1 + 2 * np.tan(np.radians(phi)) * ((1 - np.sin(np.radians(phi))) ** 2) * np.arctan(depth/width)
    return c_qd

def depth_factor_g(phi):
    return 1

def depth_factor_c(depth, width, phi):
    if phi > 0:
        c_cd = c_qd(depth, width, phi) - ((1 - c_qd(depth, width, phi)) / (Nc(phi) * np.tan(np.radians(phi))))
    else:
        c_cd = 1 + 0.33 * np.arctan(depth/width)
    return c_cd

    
def load_inclination_factor_q(vertical_load, horizontal_load, base_inclination, width, length, phi, c):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(base_inclination):
        return np.nan
    if phi<0 or np.isnan(phi) or c<0 or np.isnan(c):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan

    m = inclination_factor_m(base_inclination, width, length)
    sqi = (1 - horizontal_load / (vertical_load + width * length * c / np.tan(np.radians(phi)))) ** m
    return np.max([sqi, 0])


def load_inclination_factor_c(vertical_load, horizontal_load, base_inclination, width, length, phi, c):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(base_inclination):
        return np.nan
    if phi<0 or np.isnan(phi) or c<0 or np.isnan(c):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan
    
    m = inclination_factor_m(base_inclination, width, length)
    
    if phi==0:
        return 1 - m * horizontal_load / (width * length * c * nc(phi))
    else:
        sqi = Sqi(vertical_load, horizontal_load, base_inclination, width, length, phi, cohesion)
        sci = sqi - (1 - sqi) / (nc(phi) * np.tan(np.radians(phi)))
        return np.max([sci, 0])


def load_inclination_factor_g(vertical_load, horizontal_load, base_inclination, width, length, phi, cohesion):
    if vertical_load<=0 or horizontal_load<0 or np.isnan(vertical_load) or np.isnan(horizontal_load) or np.isnan(base_inclination):
        return np.nan
    if phi<0 or np.isnan(phi) or cohesion<0 or np.isnan(cohesion):
        return no.nan
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan

    m = inclination_factor_m(base_inclination, width, length)
    sgi = (1 - horizontal_load / (vertical_load + width * length * cohesion / np.tan(np.radians(phi)))) ** (m + 1)
    return np.max([sgi, 0])


def load_inclination_factor_m(base_inclination, width, length):
    if width>length or width<=0 or length<=0 or np.isnan(width) or np.isnan(length):
        return np.nan
    if np.isnan(base_inclination):
        return np.nan
    mb = (2 + width / length) / (1 + width / length)
    ml = (2 + length / width) / (1 + length / width)
    return ml * np.cos(np.radians(base_inclination))**2 + mb * np.sin(np.radians(base_inclination))**2


def base_inclination_factor_c():
    return 1


def base_inclination_factor_q():
    return 1


def base_inclination_factor_g():
    return 1


def ground_inclination_factor_c():
    return 1


def ground_inclination_factor_q():
    return 1


def ground_inclination_factor_g():
    return 1