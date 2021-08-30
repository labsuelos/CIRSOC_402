'''Module with the shape factors formulas for the bearing capacity
equation
'''
import numpy as np
from bearing_factors import bearing_factor_nq, bearing_factor_nc

from constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from exceptions import BearingFactorsError
from exceptions import BearingLengthSquareError
from exceptions import BearingLengthCircularError
from exceptions import BearingSizeError


def shape_factors(shape, phi, width, length=np.nan,
                  factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation shape
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the widht. by default np.nan [m]
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
        - float : Shape factor for the cohesion term in the bearing
        capacity equation [ ]
        - float : Shape factor for the surcharge term in the bearing
        capacity equation [ ]
        - float : Shape factor for the soil weight term in the bearing
        capacity equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code.
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(shape, phi, width=width, length=length)
        factor_q = cirsoc_factor_q(shape, phi, width=width, length=length)
        factor_g = cirsoc_factor_g(shape, width=width, length=length)
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, widht, length)
        factor_q = canada_factor_q(phi, widht, length)
        factor_g = canada_factor_g(phi, widht, length)
    elif factors == 'usace':
        factor_c = canada_factor_c()
        factor_q = canada_factor_q()
        factor_g = canada_factor_g()

    return factor_c, factor_q, factor_g


def shape_factor_c(shape, phi, width=np.nan, length=np.nan,
                   factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation shape
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the widht. by default np.nan [m]
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
        Shape factor for the cohesion term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code.
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(shape, phi, width=width, length=length)
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, widht, length)
    elif factors == 'usace':
        factor_c = canada_factor_c()

    return factor_c


def shape_factor_q(shape, phi, width=np.nan, length=np.nan, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation shape
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the widht. by default np.nan [m]
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
        Shape factor for the surcharge term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code.
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q(shape, phi, width=width, length=length)
    elif factors == 'canada':
        factor_q = canada_factor_q(phi, widht, length)
    elif factors == 'usace':
        factor_q = canada_factor_q()

    return factor_q


def shape_factor_g(shape, phi, width=np.nan, length=np.nan,
                   factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation shape
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int, optional
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the widht. by default np.nan [m]
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
        Shape factor for the soil weight term in the bearing capacity
        equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code.
    '''

    # All errors must be catch here, following functions do not have
    # checks
    shape = shape.lower()
    length = shape_checks(shape, width, length)
    
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g(shape, width=width, length=length)
    elif factors == 'canada':
        factor_g = canada_factor_g(phi, widht, length)
    elif factors == 'usace':
        factor_g = canada_factor_g()

    return factor_g


def shape_checks(shape, width, length):
    '''Chekcs input values before computing the dimensionless correction
    factor due to the foundation shape in the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
    length : float, int
        Length of the foundation for rectangular foundations. For
        circular foundations no value needs to be provided or np.nan.
        For square foundations no value needs to be provided or the same
        value as the widht.
    
    Returns
    -------
    float, int
        Length of the foundation for rectangular foundations. For
        circular and square foundations length=width

    Raises
    ------
    BearingShapeError
        Exception raised when the shape in the bearing capacity
        calculation requested by the user is not supported by the code.
    BearingWidthError
        Exception raised when the user failed to provide the width in for
        bearing capacity formula
    BearingLengthError
        Exception raised when the user failed to provide the length in
        for the bearing capacity formula
    BearingLengthCircularError
        Exception raised when the the bearing capacity of a circular base
        is requested but the length doesn't match the width
    BearingLengthSquareError
        Exception raised when the the bearing capacity of a square base
        is requested but the length doesn't match the width
    BearingSizeError
        Exception raised when the the bearing capacity of a circular base
        is requested but the length doesn't match the width
    '''
    if shape not in BEARINGSHAPE:
        raise BearingShapeError(shape)
    
    if shape in ['rectangle', 'rectangulo']:
        if np.isnan(width):
            raise BearingWidthError()
        if np.isnan(length):
            raise BearingLengthError()
    
    if shape in ['circular', 'circular', 'circulo'] and np.isnan(lenght):
        length = width
    if shape in ['circular', 'circular', 'circulo'] and not np.isnan(lenght):
        raise BearingLengthCircularError
    if shape in ['square', 'cuadrado', 'cuadrada'] and np.isnan(lenght):
        length = width
    if shape in ['square', 'cuadrado', 'cuadrada'] and not np.isnan(lenght):
        raise BearingLengthSquareError
    
    if length < width:
        raise BearingSizeError
    return length


def cirsoc_factor_c(shape, phi, width, length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (width/length) * (bearing_factor_nq(phi) \
                 / bearing_factor_nc(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    return factor


def cirsoc_factor_q(shape, phi, width, length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (width / length) * np.tan(np.radians(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * np.tan(np.radians(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * np.tan(np.radians(phi))        
    return factor


def cirsoc_factor_g(shape, width, length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 - 0.4 * (width / length)
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 - 0.4 * (1)
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 - 0.4 * (0)
    return factor


def canada_factor_c(phi, width, length):
    '''Shape factor for the cohesion term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    length : float, int
        Length of the foundation for rectangular foundations. For
        circular and square foundations length=width [m]

    Returns
    -------
    float, int
        Shape factor for the cohesion term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Scs factor
    if phi==0:
        return 1 + width / (5 * length)
    else:
        return 1 + (width / length) * (bearing_factor_nq(phi)/bearing_factor_nc(phi))


def canada_factor_q(phi, widht, length):
    '''Shape factor for the surcharge term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    length : float, int
        Length of the foundation for rectangular foundations. For
        circular and square foundations length=width [m]

    Returns
    -------
    float, int
        Shape factor for the surcharge term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Sqs factor
    if phi == 0:
        return 1
    else:
        return 1 + width / length * np.tan(np.radians(phi))


def canada_factor_g(phi, width, length):
    '''Shape factor for the soil weight term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    length : float, int
        Length of the foundation for rectangular foundations. For
        circular and square foundations length=width [m]

    Returns
    -------
    float, int
        Shape factor for the soil weight term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Sqs factor
    if phi == 0:
        return 1
    else:
        return 1 - 0.4 * width / length


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1