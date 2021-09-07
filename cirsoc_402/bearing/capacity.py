'''Module with the functions necessary to compute the bearing capacity.
'''

import numpy as np

from cirsoc_402.constants import BEARINGSHAPE, BEARINGMETHOD, BEARINGFACTORS, DEFAULTBEARINGFACTORS
from cirsoc_402.exceptions import BearingShapeError
from cirsoc_402.exceptions import BearingMethodError
from cirsoc_402.exceptions import BearingWidthError
from cirsoc_402.exceptions import BearingLengthError
from cirsoc_402.exceptions import BearingFactorsError

from cirsoc_402.bearing.bearing_factors import bearing_factor_nc, bearing_factor_nq, bearing_factor_ng
from cirsoc_402.bearing.shape import shape_factor_c, shape_factor_q, shape_factor_g
from cirsoc_402.bearing.depth import depth_factor_c, depth_factor_q, depth_factor_g
from cirsoc_402.bearing.load_inclination import load_inclination_factor_c, load_inclination_factor_q, load_inclination_factor_g
from cirsoc_402.bearing.ground_inclination import ground_inclination_factor_c, ground_inclination_factor_q, ground_inclination_factor_g
from cirsoc_402.bearing.base_inclination import base_inclination_factor_c, base_inclination_factor_q, base_inclination_factor_g


def bearingcapacity(shape, method, gamma_nq, gamma_ng, phi, cohesion, depth,
                    width, effective_width, effective_length, surface_load=0,
                    base_inclination=0, ground_inclination=0, vertical_load=0,
                    horizontal_load=0, load_orientation=0, 
                    factors=DEFAULTBEARINGFACTORS):
    """Bearing capacity of shallow foundations calculation according to
    the CIRSOC 402 code, the USACE manual [1]_, Eurocode 7 [2]_,
    Canadian Foundation Manual [3]_ or the combination
    of soil weight bearing factor Ngamma and modification
    factors selected by the user. By default the function computes the
    bearing capacity acording to CIRSOC 402.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    method : str
        Calculation method for the soil weight bearing capacity factor.
        The supported methods can be seen with
        cirsoc_402.constants.BEARINGMETHOD.
    gamma_nq : float, int
        Unit weight used in the surcharge term in the bearing capacity
        equation [kN/m3]
    gamma_ng : float, int
        Unit weight used in the soil weight term in the bearing capacity
        equation [kN/m3]
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
    surface_load : float, int, optional
        Load acting on the ground surface considered in the surcharge
        term of the bearing capacity equation. by default 0 [kPa]
    base_inclination : float, int, optional
        Base slope relative to the horizontal plane, by default 0 [deg]
    ground_inclination : float, int, optional
        Ground slope relative to the horizontal plane, by default 0
        [deg]
    vertical_load : float, int, optional
        Vertical load acting on the foundation level. Positive for
        compression, by default 0 [kN]
    horizontal_load : float, int, optional
        Horizontal load acting on the foundation level, by default 0
        [kN]
    load_orientation : float, int, optional
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        By default 0 [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float
        Bearing capacity of a shallow foundation [kN]

    Raises
    ------
    BearingShapeError
        Exception raised when the shape in the bearing capacity
        calculation requested by the user is not supported by the code.
    BearingMethodError
        Exception raised when the bearing capacity calculation method
        requested by the user is not supported by the code.
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code.
    """

    qu_c = bearing_c(shape, method, phi, cohesion, depth, width, effective_width,
                     effective_length, base_inclination=base_inclination,
                     ground_inclination=ground_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load,
                     load_orientation=load_orientation, factors=factors)
    qu_q = bearing_q(shape, gamma_nq, phi, cohesion, depth, width,
                     effective_width, effective_length,
                     surface_load=surface_load,
                     base_inclination=base_inclination,
                     ground_inclination=ground_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load,
                     load_orientation=load_orientation, factors=factors)
    qu_g = bearing_g(shape, method, gamma_ng, phi, cohesion, depth, width,
                     effective_width, effective_length,
                     base_inclination=base_inclination,
                     ground_inclination=ground_inclination,
                     vertical_load=vertical_load,
                     horizontal_load=horizontal_load,
                     load_orientation=load_orientation, factors=factors)

    return qu_c + qu_q + qu_g


def bearing_c(shape, method, phi, cohesion, depth, width, effective_width,
              effective_length, base_inclination=0, ground_inclination=0,
              vertical_load=0, horizontal_load=0, load_orientation=0,
              factors=DEFAULTBEARINGFACTORS):
    '''Cohesion term of the bearing capacity equation for shallow
    foundations according to the CIRSOC 402 code, the USACE manual [1]_,
    Eurocode 7 [2]_, Canadian Foundation Manual [3]_ or the combination
    of soil weight bearing factor Ngamma and modification
    factors selected by the user. By default the function computes the
    bearing capacity acording to CIRSOC 402.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    method : str
        Calculation method for the soil weight bearing capacity factor.
        The supported methods can be seen with
        cirsoc_402.constants.BEARINGMETHOD.
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        total Width of the foundation. In ciruclar foundations the
        diameter[m]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int, optional
        effective length of the equivalent rectangular load area [m] 
    base_inclination : float, int, optional
        Base slope relative to the horizontal plane, by default 0 [deg]
    ground_inclination : float, int, optional
        Ground slope relative to the horizontal plane, by default 0
        [deg]
    vertical_load : float, int, optional
        Vertical load acting on the foundation level. Positive for
        compression, by default 0 [kN]
    horizontal_load : float, int, optional
        Horizontal load acting on the foundation level, by default 0
        [kN]
    load_orientation : float, int, optional
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        By default 0 [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float
        Cohesion term of the bearing capacity equation for shallow
        foundations [kN]
    '''

    shape_factor = shape_factor_c(shape, phi, effective_width, effective_length,
                                  factors=factors)
    depth_factor = depth_factor_c(phi, depth, width, factors=factors)
    load_factor = load_inclination_factor_c(phi, cohesion,
                                            effective_width, effective_length,
                                            vertical_load, horizontal_load,
                                            load_orientation, factors=factors)
    base_factor = base_inclination_factor_c(phi, base_inclination, factors)
    ground_factor = ground_inclination_factor_c(phi, ground_inclination,
                                                factors=factors)
    
    
    # ref [3] eq. (10.9)
    qu_c = cohesion * bearing_factor_nc(phi) * shape_factor * depth_factor \
           * load_factor * base_factor * ground_factor
    return qu_c


def bearing_q(shape, gamma_nq, phi, cohesion, depth, width, effective_width,
              effective_length, surface_load=0, base_inclination=0,
              ground_inclination=0, vertical_load=0, horizontal_load=0,
              load_orientation=0, factors=DEFAULTBEARINGFACTORS):
    '''Surcharge term of the bearing capacity equation for shallow
    foundations according to the CIRSOC 402 code, the USACE manual [1]_,
    Eurocode 7 [2]_, Canadian Foundation Manual [3]_ or the combination
    of soil weight bearing factor Ngamma and modification
    factors selected by the user. By default the function computes the
    bearing capacity acording to CIRSOC 402.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    gamma_nq : float, int
        Unit weight used in the surcharge term in the bearing capacity
        equation [kN/m3]
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        total Width of the foundation. In ciruclar foundations the
        diameter[m]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int, optional
        effective length of the equivalent rectangular load area [m] 
    surface_load : float, int, optional
        Load acting on the ground surface considered in the surcharge
        term of the bearing capacity equation. by default 0 [kPa]
    base_inclination : float, int, optional
        Base slope relative to the horizontal plane, by default 0 [deg]
    ground_inclination : float, int, optional
        Ground slope relative to the horizontal plane, by default 0
        [deg]
    vertical_load : float, int, optional
        Vertical load acting on the foundation level. Positive for
        compression, by default 0 [kN]
    horizontal_load : float, int, optional
        Horizontal load acting on the foundation level, by default 0
        [kN]
    load_orientation : float, int, optional
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        By default 0 [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float
        Surcharge term of the bearing capacity equation for shallow
        foundations [kN]
    '''
    shape_factor = shape_factor_q(shape, phi, effective_width, effective_length,
                                  factors=factors)
    depth_factor = depth_factor_q(phi, depth, width, factors=factors)
    load_factor = load_inclination_factor_q(phi, cohesion,
                                            effective_width, effective_length,
                                            vertical_load, horizontal_load,
                                            load_orientation, factors)
    base_factor = base_inclination_factor_q(phi, base_inclination,
                                            factors=factors)
    ground_factor = ground_inclination_factor_c(phi, ground_inclination,
                                                factors=factors)
    
    # ref [3] eq. (10.10)
    qu_q = (surface_load + depth * gamma_nq) * bearing_factor_nq(phi) * \
           shape_factor * depth_factor * load_factor * base_factor * ground_factor
    return qu_q


def bearing_g(shape, method, gamma_ng, phi, cohesion, depth, width,
              effective_width, effective_length, base_inclination=0,
              ground_inclination=0, vertical_load=0, horizontal_load=0,
              load_orientation=0, factors=DEFAULTBEARINGFACTORS):
    '''Soil weight term of the bearing capacity equation for shalow
    foundations according to the CIRSOC 402 code, the USACE manual [1]_,
    Eurocode 7 [2]_, Canadian Foundation Manual [3]_ or the combination
    of soil weight bearing factor Ngamma and modification
    factors selected by the user. By default the function computes the
    bearing capacity acording to CIRSOC 402.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    method : str
        Calculation method for the soil weight bearing capacity factor.
        The supported methods can be seen with
        cirsoc_402.constants.BEARINGMETHOD.
    gamma_ng : float, int
        Unit weight used in the soil weight term in the bearing capacity
        equation [kN/m3]
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        total Width of the foundation. In ciruclar foundations the
        diameter[m]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int, optional
        effective length of the equivalent rectangular load area [m]
    base_inclination : float, int, optional
        Base slope relative to the horizontal plane, by default 0 [deg]
    ground_inclination : float, int, optional
        Ground slope relative to the horizontal plane, by default 0
        [deg]
    vertical_load : float, int, optional
        Vertical load acting on the foundation level. Positive for
        compression, by default 0 [kN]
    horizontal_load : float, int, optional
        Horizontal load acting on the foundation level, by default 0
        [kN]
    load_orientation : float, int, optional
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        By default 0 [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float
        Soil weight term of the bearing capacity equation for shallow
        foundations [kN]
    '''

    shape_factor = shape_factor_g(shape, phi, effective_width, effective_length,
                                  factors=factors)
    depth_factor = depth_factor_g(phi, depth, width, factors=factors)
    load_factor = load_inclination_factor_g(phi, cohesion,
                                            effective_width, effective_length,
                                            vertical_load, horizontal_load,
                                            load_orientation, factors)
    base_factor = base_inclination_factor_g(phi, base_inclination,
                                            factors=factors)
    ground_factor = ground_inclination_factor_g(ground_inclination,
                                                factors=factors)
    
    bearing_factor = bearing_factor_ng(phi, method,
                                       ground_inclination=ground_inclination)

    if shape == ['rectangle', 'rectangulo']:
        qu_g_width = 0.5 * gamma_ng * effective_width * bearing_factor 
        qu_g_length = 0.5 * gamma_ng * effective_length * bearing_factor
        qu_g = np.minimum(qu_g_width, qu_g_length)
    else:
        qu_g = 0.5 * gamma_ng * effective_width * bearing_factor
    qu_g = qu_g * bearing_factor_nq(phi) * shape_factor * depth_factor \
           * load_factor * base_factor * ground_factor
    return qu_g