'''Module with the load inclination factors formulas for the bearing
capacity equation
'''
import numpy as np

from cirsoc_402.constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from cirsoc_402.exceptions import BearingFactorsError
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc
from cirsoc_402.bearing.shape import shape_checks



def load_inclination_factors(shape, phi, cohesion, width, length, vertical_load,
                             horizontal_load, load_orientation,
                             factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the width. by default np.nan [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length
        [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    tuple
        tuple with the following content:
        - float : Load inclination factor for the cohesion term in the
        bearing capacity equation [ ]
        - float : Load inclination factor for the surcharge term in the
        bearing capacity equation [ ]
        - float : Load inclination factor for the soil weight term in the
        bearing capacity equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)

    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
        factor_g = canada_factor_g(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'usace':
        factor_c = usace_factor_c()
        factor_q = usace_factor_q()
        factor_g = usace_factor_g()

    return factor_c, factor_q, factor_g


def load_inclination_factor_c(shape, phi, cohesion, width, length,
                              vertical_load, horizontal_load, load_orientation,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the width. by default np.nan [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length
        [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float, int
        Depth factor for the cohesion term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)

    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'usace':
        factor_c = usace_factor_c()

    return factor_c


def load_inclination_factor_q(shape, phi, cohesion, width, length,
                              vertical_load, horizontal_load, load_orientation,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the width. by default np.nan [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length
        [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float, int
        Depth factor for the surcharge term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)

    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q()
    elif factors == 'canada':
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'usace':
        factor_q = usace_factor_q()

    return factor_q


def load_inclination_factor_g(shape, phi, cohesion, width, length,
                              vertical_load, horizontal_load, load_orientation,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the width. by default np.nan [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length
        [deg]
    factors : str, optional
        Set of dimensionless correction factors for the cohesion,
        surcharge and soil weight terms due to the depth, shape, load
        inclination, ground inclination and base inclination to be used
        in the calculation. The supported factor families can be seen
        with cirsoc_402.constants.BEARINGFACTORS. By default 
        cirsoc_402.constants.DEFAULTBEARINGFACTORS

    Returns
    -------
    float, int
        Depth factor for the soil weight term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)

    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_g = canada_factor_g(phi, cohesion, width, length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'usace':
        factor_g = usace_factor_g()

    return factor_g


def cirsoc_factor_c():
    return 1


def cirsoc_factor_q():
    return 1


def cirsoc_factor_g():
    return 1


def canada_factor_c(phi, cohesion, width, length, vertical_load,
                    horizontal_load, load_orientation):
    '''Load inclination factor for the cohesion term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation [m]
    length : float, int
        Length of the foundation [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression. [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        [deg]

    Returns
    -------
    int, float
        Load inclination factor for the cohesion term in the bearing
        capacity equation [ ]
    '''

    # ref [3] table 10.2 factor Sci
    factor_m = canada_factor_m(width, length, load_orientation)
    if phi==0:
        return 1 - factor_m * horizontal_load / (width * length * cohesion * bearing_factor_nc(phi))
    else:
        factor_q = canada_factor_q(phi, cohesion, width, length, vertical_load, horizontal_load, load_orientation)
        factor = factor_q - (1 - factor_q) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))
        return factor


def canada_factor_q(phi, cohesion, width, length, vertical_load,
                    horizontal_load, load_orientation):
    '''Load inclination factor for the surcharge term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation [m]
    length : float, int
        Length of the foundation [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression. [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        [deg]

    Returns
    -------
    int, float
        Load inclination factor for the surcharge term in the bearing
        capacity equation [ ]
    '''

    # ref [3] table 10.2 factor Sqi
    factor_m = canada_factor_m(width, length, load_orientation)
    factor = (1 - horizontal_load / (vertical_load + width * length * cohesion / np.tan(np.radians(phi)))) ** factor_m
    return factor


def canada_factor_g(phi, cohesion, width, length, vertical_load,
                    horizontal_load, load_orientation):
    '''Load inclination factor for the soil weight term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    width : float, int
        Width of the foundation [m]
    length : float, int
        Length of the foundation [m]
    vertical_load : float, int
        Vertical load acting on the foundation level. Positive for
        compression. [kN]
    horizontal_load : float, int
        Horizontal load acting on the foundation level [kN]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        [deg]

    Returns
    -------
    int, float
        Load inclination factor for the soil weight term in the bearing
        capacity equation [ ]
    '''

    # ref [3] table 10.2 factor Sgi
    factor_m = canada_factor_m(width, length, load_orientation)
    factor = (1 - horizontal_load / (vertical_load + width * length * cohesion / np.tan(np.radians(phi)))) ** (factor_m + 1)
    return factor


def canada_factor_m(width, length, load_orientation):
    '''Auxiliary factor n used in the computation of the load
    inclination factors in the bearing capacity equation according to
    the Canadian Engineering Foundation Manual [3]_
    (table 10.2, note 2).

    Parameters
    ----------
    width : float, int
        Width of the foundation [m]
    length : float, int
        Length of the foundation [m]
    load_orientation : float, int
        Orientation of the horizontal load in the foundation plane.
        If load_orientation=0 the load acts parallel to the width,
        if load_orientation=90 the load acts parallel to the length.
        [deg]

    Returns
    -------
    float, int
        Auxiliary factor n used in the computation of the load
        inclination factors [ ]
    '''
    # ref [3] table 10.2 note [1]
    mb = (2 + width / length) / (1 + width / length)
    ml = (2 + length / width) / (1 + length / width)
    return ml * np.cos(np.radians(load_orientation))**2 + mb * np.sin(np.radians(load_orientation))**2


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1
