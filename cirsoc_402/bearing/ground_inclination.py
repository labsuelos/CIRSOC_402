'''Module with the ground inclination factors formulas for the bearing
capacity equation
'''
import numpy as np

from cirsoc_402.constants import DEFAULTBEARINGFACTORS
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc


def ground_inclination_factors(phi, ground_inclination,
                               factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the ground inclination
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]
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
        - float : Ground inclination factor for the cohesion term in the
        bearing capacity equation [ ]
        - float : Ground inclination factor for the surcharge term in the
        bearing capacity equation [ ]
        - float : Ground inclination factor for the soil weight term in the
        bearing capacity equation [ ]
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, ground_inclination)
        factor_q = canada_factor_q(ground_inclination)
        factor_g = canada_factor_g(ground_inclination)
    elif factors == 'meyerhof':
        factor_c = meyerhof_factor_c()
        factor_q = meyerhof_factor_q()
        factor_g = meyerhof_factor_g()
    elif factors == 'hansen':
        factor_c = hansen_factor_c()
        factor_q = hansen_factor_q()
        factor_g = hansen_factor_g()
    elif factors == 'vesic':
        factor_c = vesic_factor_c()
        factor_q = vesic_factor_q()
        factor_g = vesic_factor_g()
    else:
        factor_c = np.nan
        factor_q = np.nan
        factor_g = np.nan

    return factor_c, factor_q, factor_g


def ground_inclination_factor_c(phi, ground_inclination,
                                factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the ground inclination
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]
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
        Ground inclination factor for the cohesion term in the bearing
        capacity equation [ ]
    '''

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_c()
    elif factors == 'canada':
        return canada_factor_c(phi, ground_inclination)
    elif factors == 'meyerhof':
        return meyerhof_factor_c()
    elif factors == 'hansen':
        return hansen_factor_c()
    elif factors == 'vesic':
        return vesic_factor_c()
    else:
        return np.nan


def ground_inclination_factor_q(ground_inclination,
                                factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the ground inclination
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]
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
        Ground inclination factor for the surcharge term in the bearing
        capacity equation [ ]
    '''

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_q()
    elif factors == 'canada':
        return canada_factor_q(ground_inclination)
    elif factors == 'meyerhof':
        return meyerhof_factor_q()
    elif factors == 'hansen':
        return hansen_factor_q()
    elif factors == 'vesic':
        return vesic_factor_q()
    else:
        return np.nan


def ground_inclination_factor_g(ground_inclination,
                                factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the ground inclination
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]
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
        Ground inclination factor for the soil weight term in the bearing
        capacity equation [ ]
    '''

    # All errors must be catch here, following functions do not have
    # checks
    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_g()
    elif factors == 'canada':
        return canada_factor_g(ground_inclination)
    elif factors == 'meyerhof':
        return meyerhof_factor_g()
    elif factors == 'hansen':
        return hansen_factor_g()
    elif factors == 'vesic':
        return vesic_factor_g()
    else:
        return np.nan


def cirsoc_factor_c():
    return 1


def cirsoc_factor_q():
    return 1


def cirsoc_factor_g():
    return 1


def canada_factor_c(phi, ground_inclination):
    '''Ground inclination factor for the cohesion term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Ground inclination factor for the cohesion term in the bearing
        capacity equation [ ]
    '''

    # ref [3] table 10.2 factor Scb
    if phi == 0:
        factor = 1 - (2 * np.radians(ground_inclination)) / (np.pi + 2)
    else:
        qfactor = canada_factor_q(ground_inclination)
        factor = qfactor - (1 - qfactor) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))
    return factor


def canada_factor_q(ground_inclination):
    '''Ground inclination factor for the surcharge term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Ground inclination factor for the surcharge term in the bearing
        capacity equation [ ]
    '''
    # ref [3] table 10.2 factor Sqb
    return (1 - np.tan(np.radians(ground_inclination)))**2


def canada_factor_g(ground_inclination):
    '''Ground inclination factor for the soil weight term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    ground_inclination : float, int
        Ground slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Ground inclination factor for the soil weight term in the
        bearing capacity equation [ ]
    '''
    # ref [3] table 10.2 factor Scb
    return (1 - np.tan(np.radians(ground_inclination)))**2


def meyerhof_factor_c():
    return 1


def meyerhof_factor_q():
    return 1


def meyerhof_factor_g():
    return 1


def hansen_factor_c():
    return 1


def hansen_factor_q():
    return 1


def hansen_factor_g():
    return 1


def vesic_factor_c():
    return 1


def vesic_factor_q():
    return 1


def vesic_factor_g():
    return 1
