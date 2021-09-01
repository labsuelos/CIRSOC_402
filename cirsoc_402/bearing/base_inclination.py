'''Module with the base inclination factors formulas for the bearing
capacity equation
'''
import numpy as np


from cirsoc_402.constants import BEARINGFACTORS, DEFAULTBEARINGFACTORS
from cirsoc_402.exceptions import BearingFactorsError
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc


def base_inclination_factors(phi, base_inclination,
                             factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the base inclination
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]
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
        - float : Base inclination factor for the cohesion term in the
        bearing capacity equation [ ]
        - float : Base inclination factor for the surcharge term in the
        bearing capacity equation [ ]
        - float : Base inclination factor for the soil weight term in the
        bearing capacity equation [ ]

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
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, base_inclination)
        factor_q = canada_factor_q(phi, base_inclination)
        factor_g = canada_factor_g(phi, base_inclination)
    elif factors == 'usace':
        factor_c = usace_factor_c()
        factor_q = usace_factor_q()
        factor_g = usace_factor_g()

    return factor_c, factor_q, factor_g


def base_inclination_factor_c(phi, base_inclination,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the base inclination
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]
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
        Base inclination factor for the cohesion term in the bearing
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
        factor_c = cirsoc_factor_c()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, base_inclination)
    elif factors == 'usace':
        factor_c = usace_factor_c()

    return factor_c


def base_inclination_factor_q(phi, base_inclination,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the base inclination
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]
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
        Base inclination factor for the surcharge term in the bearing
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
        factor_q = cirsoc_factor_q()
    elif factors == 'canada':
        factor_q = canada_factor_q(phi, base_inclination)
    elif factors == 'usace':
        factor_q = usace_factor_q()

    return factor_q


def base_inclination_factor_g(phi, base_inclination,
                              factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the base inclination
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]
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
        Base inclination factor for the soil weight term in the bearing
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
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_g = canada_factor_g(phi, base_inclination)
    elif factors == 'usace':
        factor_g = usace_factor_g()

    return factor_g


def cirsoc_factor_c():
    return 1


def cirsoc_factor_q():
    return 1


def cirsoc_factor_g():
    return 1


def canada_factor_c(phi, base_inclination):
    '''Base inclination factor for the cohesion term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Base inclination factor for the cohesion term in the bearing
        capacity equation  [ ]
    '''

    # ref [3] table 10.2 factor Scdelta
    if phi == 0:
        factor = 1 - (2 * np.radians(base_inclination)) / (np.pi + 2)
    else:
        qfactor = canada_factor_q(phi, base_inclination)
        factor = qfactor - (1 - qfactor) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))
    return factor


def canada_factor_q(phi, base_inclination):
    '''Base inclination factor for the surcharge term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Base inclination factor for the surcharge term in the bearing
        capacity equation  [ ]
    '''

    # ref [3] table 10.2 factor Sqdelta
    return (1 - np.radians(base_inclination) * np.tan(np.radians(phi)))**2


def canada_factor_g(phi, base_inclination):
    '''Base inclination factor for the soil weight term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    base_inclination : float, int
        Base slope relative to the horizontal plane [deg]

    Returns
    -------
    float, int
        Base inclination factor for the soil weight term in the bearing
        capacity equation  [ ]
    '''

    # ref [3] table 10.2 factor Scdelta
    return  (1 - np.radians(base_inclination) * np.tan(np.radians(phi)))**2


def usace_factor_c():
    return 1


def usace_factor_q():
    return 1


def usace_factor_g():
    return 1