'''Module with the depth factors formulas for the bearing capacity
equation
'''
import numpy as np

from cirsoc_402.constants import DEFAULTBEARINGFACTORS
from cirsoc_402.exceptions import BearingFactorsError
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc

def depth_factors(phi, depth, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]
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
    '''

    if any(np.isnan([phi, depth, width])) or width==0:
        return np.nan, np.nan, np.nan
    
    factors = factors.lower()
    if factors == 'cirsoc':
        factor_c = cirsoc_factor_c(phi, depth, width)
        factor_q = cirsoc_factor_q(phi, depth, width)
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, depth, width)
        factor_q = canada_factor_q(phi, depth, width)
        factor_g = canada_factor_g()
    elif factors == 'meyerhof':
        factor_c = meyerhof_factor_c(phi, depth, width)
        factor_q = meyerhof_factor_q(phi, depth, width)
        factor_g = meyerhof_factor_g(phi, depth, width)
    elif factors == 'hansen':
        factor_c = hansen_factor_c(phi, depth, width)
        factor_q = hansen_factor_q(phi, depth, width)
        factor_g = hansen_factor_g()
    elif factors == 'vesic':
        factor_c = vesic_factor_c(phi, depth, width)
        factor_q = vesic_factor_q(phi, depth, width)
        factor_g = vesic_factor_g()
    else:
        factor_c = np.nan
        factor_q = np.nan
        factor_g = np.nan

    return factor_c, factor_q, factor_g


def depth_factor_c(phi, depth, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_c(phi, depth, width,)
    elif factors == 'canada':
        return canada_factor_c(phi, depth, width)
    elif factors == 'meyerhof':
        return meyerhof_factor_c(phi, depth, width)
    elif factors == 'hansen':
        return hansen_factor_c(phi, depth, width)
    elif factors == 'vesic':
        return vesic_factor_c(phi, depth, width)
    else:
        return np.nan


def depth_factor_q(phi, depth, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_q(phi, depth, width)
    elif factors == 'canada':
        return canada_factor_q(phi, depth, width)
    elif factors == 'meyerhof':
        return meyerhof_factor_q(phi, depth, width)
    elif factors == 'hansen':
        return hansen_factor_q(phi, depth, width)
    elif factors == 'vesic':
        return vesic_factor_q(phi, depth, width)
    else:
        return np.nan


def depth_factor_g(phi, depth, width, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation depth
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]
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
    '''

    if any(np.isnan([phi, depth, width])) or width==0:
        return np.nan

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_g()
    elif factors == 'canada':
        return canada_factor_g()
    elif factors == 'meyerhof':
        return meyerhof_factor_g(phi, depth, width)
    elif factors == 'hansen':
        return hansen_factor_g()
    elif factors == 'vesic':
        return vesic_factor_g()
    else:
        return np.nan


def cirsoc_factor_c(phi, depth, width):
    if width == 0:
        return np.nan
    if phi > 0:
        factor_q = cirsoc_factor_q(phi, depth, width)
        return factor_q - ((1 - factor_q) / (bearing_factor_nc(phi) * np.tan(np.radians(phi))))
    elif phi == 0:
        return 1 + 0.33 * np.arctan(depth / width)
    else:
        return np.nan


def cirsoc_factor_q(phi, depth, width):
    if width == 0:
        return np.nan
    return 1 + 2 * np.tan(np.radians(phi)) * ((1 - np.sin(np.radians(phi))) ** 2) * np.arctan(depth / width)


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
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

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
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the surcharge term in the bearing capacity
        equation [ ]
    '''

    # ref [3] table 10.2 Sqd factor
    kfactor = canada_factor_k(depth, width)
    return 1 + 2 * np.tan(np.radians(phi)) * (1 - np.sin(np.radians(phi)))**2 * kfactor


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
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Auxiliary factor k used in the computation of the depth factor
        for the cohesion and surcharge terms [ ]
    '''
    if width == 0:
        return np.nan
    # ref [3] table 10.2 note 2
    if depth <= width:
        return depth / width
    else:
        return np.arctan(depth / width)


def meyerhof_factor_c(phi, depth, width):
    '''Depth factor for the cohesion term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3).

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the cohesion term term in the bearing capacity
        equation [ ]
    '''

    # ref [1] table 4-3
    if width == 0:
        return np.nan
    return 1 + 0.2 * np.tan(np.radians(45 + phi / 2)) * depth / width


def meyerhof_factor_q(phi, depth, width):
    '''Depth factor for the surcharge term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the surcharge term term in the bearing capacity
        equation [ ]
    '''
    # ref [1] table 4-3
    if width == 0:
        return np.nan
    if phi == 0:
        return 1
    elif phi > 10:
        return 1 + 0.1 * np.tan(np.radians(45 + phi / 2)) * depth / width
    elif 0 < phi and phi <=10:
        factor10 = 1 + 0.1 * np.tan(np.radians(45 + 10 / 2)) * depth / width
        factor0 = 1
        return (factor10 - factor0) * phi / 10 + factor0
    else:
        return np.nan


def meyerhof_factor_g(phi, depth, width):
    '''Depth factor for the soil weight term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the soil weight term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-3
    return meyerhof_factor_q(phi, depth, width)


def hansen_factor_c(phi, depth, width):
    '''Depth factor for the cohesion term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-5). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the cohesion term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-5
    kfactor = hansen_factor_k(depth, width)
    if phi == 0:
        return 0.4 * kfactor
    elif phi > 0:
        return 1 + 0.4 * kfactor
    else:
        return np.nan


def hansen_factor_q(phi, depth, width):
    '''Depth factor for the surcharge term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-5). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the surcharge term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-5
    if phi == 0:
        return 1
    elif phi > 0:
        kfactor = hansen_factor_k(depth, width)
        return 1 + 2 * np.tan(np.radians(phi)) * (1 - np.sin(np.radians(phi)))**2 * kfactor
    else:
        return np.nan


def hansen_factor_g():
    '''Depth factor for the soil weight term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-5). 

    Parameters
    ----------

    Returns
    -------
    float, int
        Depth factor for the soil weight term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-5
    return 1


def hansen_factor_k(depth, width):
    '''Auxiliary factor k used in the computation of the depth factors
    in the bearing capacity according to Hansen [7]_ as stated in the
    USACE manual [1]_ (table 4-5).

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Auxiliary factor k used in the computation of the depth factors
        [ ]
    '''
    if width == 0:
        return np.nan
    # ref [1] table 4-5
    if depth <= width:
        return depth / width
    else:
        return np.arctan(depth / width)


def vesic_factor_c(phi, depth, width):
    '''Depth factor for the cohesion term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the cohesion term term in the bearing
        capacity equation [ ]
    '''

    # ref [1] table 4-6
    kfactor = vesic_factor_k(depth, width)
    if phi >= 0:
        return 1 + 0.4 * kfactor
    else:
        return np.nan


def vesic_factor_q(phi, depth, width):
    '''Depth factor for the surcharge term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Depth factor for the surcharge term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-6
    if phi == 0:
        return 1
    elif phi > 0:
        kfactor = vesic_factor_k(depth, width)
        return 1 + 2 * np.tan(np.radians(phi)) * (1 - np.sin(np.radians(phi)))**2 * kfactor
    else:
        return np.nan


def vesic_factor_g():
    '''Depth factor for the soil weight term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------

    Returns
    -------
    float, int
        Depth factor for the soil weight term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-6
    return 1


def vesic_factor_k(depth, width):
    '''Auxiliary factor k used in the computation of the depth factors
    in the bearing capacity according to Vesic [8]_ [9]_ as stated in
    the USACE manual [1]_ (table 4-6).

    Parameters
    ----------
    depth : float, int
        Depth of the foundation level [m]
    width : float, int
        Total Width of the foundation. In ciruclar foundations the
        diameter[m]

    Returns
    -------
    float, int
        Auxiliary factor k used in the computation of the depth factors
        [ ]
    '''
    if width == 0:
        return np.nan
    # ref [1] table 4-6
    if depth <= width:
        return depth / width
    else:
        return np.arctan(depth / width)

