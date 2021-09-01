'''Module with the depth factors formulas for the bearing capacity
equation
'''
import numpy as np

from cirsoc_402.constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from cirsoc_402.exceptions import BearingFactorsError
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc

def depth_factors(depth, phi, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
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
        - float : Depth factor for the cohesion term in the bearing
        capacity equation [ ]
        - float : Depth factor for the surcharge term in the bearing
        capacity equation [ ]
        - float : Depth factor for the soil weight term in the bearing
        capacity equation [ ]

    Raises
    ------
    BearingFactorsError
        Exception raised when the the bearing capacity factor group
        requested by the user is not supported by the code
    '''

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(depth, width, phi)
        factor_q = cirsoc_factor_q(depth, width, phi)
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, depth, width)
        factor_q = canada_factor_q(phi, depth, width)
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_c = usace_factor_c()
        factor_q = usace_factor_q()
        factor_g = usace_factor_g()

    return factor_c, factor_q, factor_g


def depth_factor_c(depth, phi, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
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
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_c = cirsoc_factor_c(depth, width, phi)
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, depth, width)
    elif factors == 'usace':
        factor_c = usace_factor_c()

    return factor_c


def depth_factor_q(depth, phi, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
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
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_q = cirsoc_factor_q(depth, width, phi)
    elif factors == 'canada':
        factor_q = canada_factor_q(phi, depth, width)
    elif factors == 'usace':
        factor_q = usace_factor_q()

    return factor_q


def depth_factor_g(depth, phi, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    phi : float, int
        Soil friction angle [deg]
    width : float, int
        Width of the foundation. In ciruclar foundations the diameter
        [m]
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
    factors = factors.lower()
    if factors not in BEARINGFACTORS:
        raise BearingFactorsError(factors)
    elif factors == 'cirsoc':
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_g = canada_factor_g()
    elif factors == 'usace':
        factor_g = usace_factor_g()

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


def cirsoc_factor_g():
    return 1


def canada_factor_c(phi, depth, width):
    '''Depth factor for the cohesion term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Width of the foundation [m]

    Returns
    -------
    float, int
        Depth factor for the cohesion term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Scd factor
    kfactor = canada_factor_k(depth, width)
    if phi==0:
        return 1 + 0.4 * kfactor
    else:
        sqd = canada_factor_q(phi, depth, width)
        return sqd - (1 - sqd) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))


def canada_factor_q(phi, depth, width):
    '''Depth factor for the surcharge term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Width of the foundation [m]

    Returns
    -------
    float, int
        Depth factor for the surcharge term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Sqd factor
    kfactor = canada_factor_k(depth, width)
    return 1 + 2 * kfactor * np.tan(np.radians(phi)) * (1 - np.sin(np.radians(phi)))**2


def canada_factor_g():
    '''Depth factor for the soil weigth term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------

    Returns
    -------
    float, int
        Depth factor for the weigth term term in the bearing capacity
        equation [ ]
    '''
    # ref [3] table 10.2 Sgd factor
    return 1


def canada_factor_k(depth, width):
    '''Auxiliary factor k used in the computation of the depth factor
    for the cohesion and surcharge terms in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2).

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Width of the foundation [m]

    Returns
    -------
    float, int
        Auxiliary factor k used in the computation of the depth factor
        for the cohesion and surcharge terms [ ]
    '''

    # ref [3] table 10.2 note 2
    if depth <= width:
        kfactor = depth/width
    else:
        kfactor = np.arctan(depth / width)
    return kfactor


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1

