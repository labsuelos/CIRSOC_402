'''Module with the functions necessary to compute the bearing capacity.
'''

import numpy as np

from constants import BEARINGSHAPE, BEARINGMETHOD, BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingShapeError
from exceptions import BearingMethodError
from exceptions import BearingWidthError
from exceptions import BearingLengthError
from exceptions import BearingFactorsError

from bearing_factors import bearing_factor_nc, bearing_factor_nq, bearing_factor_ng
from shape import shape_factor_c, shape_factor_q, shape_factor_g
from depth import depth_factor_c, depth_factor_q, depth_factor_g
from load_inclination import load_inclination_factor_c, load_inclination_factor_q, load_inclination_factor_g
from ground_inclination import ground_inclination_factor_c, ground_inclination_factor_q, ground_inclination_factor_g
from base_inclination_factors import base_inclination_factor_c, base_inclination_factor_q, base_inclination_factor_g


def bearingcapacity(shape, method, gamma_ng, gamma_nq, phi, cohesion, depth,
                    load, width, length=np.nan, base_inclination=np.nan
                    vertical_load=np.nan, horizontal_load=np.nan,
                    load_theta=np.nan, ground_inclination=np.nan,
                    factors=DEFAULTBEARINGFACTORS):
    
    shape = shape.lower()
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)

    if method not in BEARINGMETHOD:
        raise BearingMethodError(method)
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)

    qu_c = bearing_c(shape, method, phi, cohesion, depth, width, length=length,
              base_inclination=base_inclination,
              vertical_load=vertical_load, horizontal_load=horizontal_load,
              ground_inclination=ground_inclination, , factors=factors)
    qu_q = bearing_q(shape, gamma_nq, phi, depth, load, width, length=length,
                     base_inclination=base_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load,
                     ground_inclination=ground_inclination, factors=factors)
    qu_g = bearing_g(shape, method, gamma_ng, phi, width, length=length,
                     base_inclination=base_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load,
                     ground_inclination=ground_inclination, factors=factors)

    return qu_c + qu_q + qu_g


def bearing_c(shape, method, phi, cohesion, depth, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, ground_inclination=np.nan,
              factors='cirsoc204'):
    
    shape_factor = shape_factor_c(shape, phi, width=width, length=length, factors)
    depth_factor = depth_factor_c(depth, phi, width, factors=factors)
    load_factor = load_inclination_factor_c(phi, cohesion, width, length,
                                            vertical_load, horizontal_load,
                                            load_theta, factors)
    base_factor = base_inclination_factor_c(factors)
    ground_factor = ground_inclination_factor_c(factors)
    
    
    qu_c = cohesion * bearing_factor_nc(phi) * shape_factor * depth_factor  /
           * load_factor * base_factor * ground_factor
    return qu_c


def bearing_q(shape, gamma_nq, phi, depth, load, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, ground_inclination=np.nan,
              factors='cirsoc204'):
    
    shape_factor = shape_factor_q(shape, phi, width=width, length=length, factors)
    depth_factor = depth_factor_q(depth, phi, width, factors=factors)
    load_factor = load_inclination_factor_q(phi, cohesion, width, length,
                                            vertical_load, horizontal_load,
                                            load_theta, factors)
    base_factor = base_inclination_factor_q(factors)
    ground_factor = ground_inclination_factor_c(factors)
    
    qu_q = (load + depth * gamma_nq) * bearing_factor_nq(phi) * shape_factor  /
           * depth_factor * load_factor * base_factor * ground_factor
    return qu_q


def bearing_g(shape, method, gamma_ng, phi, width, length=np.nan,
              base_inclination=np.nan, vertical_load=np.nan,
              horizontal_load=np.nan, ground_inclination=np.nan,
              factors='cirsoc204'):

    shape_factor = shape_factor_g(shape, phi, width=width, length=length, factors)
    depth_factor = depth_factor_g(depth, phi, width, factors=factors)
    load_factor = load_inclination_factor_g(phi, cohesion, width, length,
                                            vertical_load, horizontal_load,
                                            load_theta, factors)
    base_factor = base_inclination_factor_g(factors)
    ground_factor = ground_inclination_factor_g(factors)
    
    bearing_factor = bearing_factor_ng(phi, method)

    if shape == ['rectangle', 'rectangulo']:
        qu_g_width = 0.5 * gamma_ng * width * bearing_factor 
        qu_g_length = 0.5 * gamma_ng * length * bearing_factor
        qu_g = np.minimum(qu_g_width, qu_g_length)
    else:
        qu_g = 0.5 * gamma_ng * width * bearing_factor
    qu_g = qu_g * bearing_factor_nq(phi) * shape_factor * depth_factor \
           * load_factor * base_factor * ground_factor
    return qu_g