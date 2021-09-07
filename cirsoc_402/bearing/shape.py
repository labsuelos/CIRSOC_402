'''Module with the shape factors formulas for the bearing capacity
equation
'''
import numpy as np

from cirsoc_402.constants import DEFAULTBEARINGFACTORS
from cirsoc_402.bearing.bearing_factors import bearing_factor_nq, bearing_factor_nc


def shape_factors(shape, phi, effective_width, effective_length,
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
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 
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
    '''

    if any(np.isnan([phi, effective_length, effective_width])):
        return np.nan, np.nan, np.nan
    
    factors = factors.lower()
    if factors == 'cirsoc':
        factor_c = cirsoc_factor_c(shape, phi, effective_width, effective_length)
        factor_q = cirsoc_factor_q(shape, phi, effective_width, effective_length)
        factor_g = cirsoc_factor_g(shape, effective_width, effective_length)
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, effective_width, effective_length)
        factor_q = canada_factor_q(phi, effective_width, effective_length)
        factor_g = canada_factor_g(effective_width, effective_length)
    elif factors == 'meyerhof ':
        factor_c = meyerhof_factor_c(phi, effective_width, effective_length)
        factor_q = meyerhof_factor_q(phi, effective_width, effective_length)
        factor_g = meyerhof_factor_g(phi, effective_width, effective_length)
    elif factors == 'hansen ':
        factor_c = hansen_factor_c(phi, effective_width, effective_length)
        factor_q = hansen_factor_q(phi, effective_width, effective_length)
        factor_g = hansen_factor_g(phi, effective_width, effective_length)
    elif factors == 'vesic ':
        factor_c = vesic_factor_c(phi, effective_width, effective_length)
        factor_q = vesic_factor_q(phi, effective_width, effective_length)
        factor_g = vesic_factor_g(phi, effective_width, effective_length)
    else:
        factor_c = np.nan
        factor_q = np.nan
        factor_g = np.nan

    return factor_c, factor_q, factor_g


def shape_factor_c(shape, phi, effective_width, effective_length,
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
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 
    factors : str
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
    '''

    shape = shape.lower()
    
    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_c(shape, phi, effective_width, effective_length)
    elif factors == 'canada':
        return canada_factor_c(phi, effective_width, effective_length)
    elif factors == 'meyerhof':
        return meyerhof_factor_c(phi, effective_width, effective_length)
    elif factors == 'hansen':
        return hansen_factor_c(phi, effective_width, effective_length)
    elif factors == 'vesic':
        return vesic_factor_c(phi, effective_width, effective_length)
    else:
        return np.nan


def shape_factor_q(shape, phi, effective_width, effective_length,
                   factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the foundation shape
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    shape : str
        Shape of the foundation. The supported shapes can be seen with
        cirsoc_402.constants.BEARINGSHAPE.
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 
    factors : str
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
    '''

    if any(np.isnan([phi, effective_length, effective_width])):
        return np.nan

    shape = shape.lower()
    
    factors = factors.lower()
    if  factors == 'cirsoc':
        return cirsoc_factor_q(shape, phi, effective_width, effective_length)
    elif factors == 'canada':
        return canada_factor_q(phi, effective_width, effective_length)
    elif factors == 'meyerhof':
        return meyerhof_factor_q(phi, effective_width, effective_length)
    elif factors == 'hansen':
        return hansen_factor_q(phi, effective_width, effective_length)
    elif factors == 'vesic':
        return vesic_factor_q(phi, effective_width, effective_length)
    else:
        return np.nan


def shape_factor_g(shape, phi, effective_width, effective_length,
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
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 
    factors : str
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
    '''

    shape = shape.lower()
    
    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_g(shape, effective_width, effective_length)
    elif factors == 'canada':
        return canada_factor_g(effective_width, effective_length)
    elif factors == 'meyerhof':
        return meyerhof_factor_g(phi, effective_width, effective_length)
    elif factors == 'hansen':
        return hansen_factor_g(phi, effective_width, effective_length)
    elif factors == 'vesic':
        return vesic_factor_g(phi, effective_width, effective_length)
    else:
        return np.nan


def cirsoc_factor_c(shape, phi, effective_width, effective_length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (effective_width/effective_length) * (bearing_factor_nq(phi) \
                 / bearing_factor_nc(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * (bearing_factor_nq(phi) / bearing_factor_nc(phi))
    return factor


def cirsoc_factor_q(shape, phi, effective_width, effective_length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 + (effective_width / effective_length) * np.tan(np.radians(phi))
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 + (1) * np.tan(np.radians(phi))
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 + (0) * np.tan(np.radians(phi))        
    return factor


def cirsoc_factor_g(shape, effective_width, effective_length):
    shape = shape.lower()
    if shape in ['rectangle', 'rectangulo']:
        factor = 1 - 0.4 * (effective_width / effective_length)
    elif shape in ['square', 'cuadrado', 'cuadrada']:
        factor = 1 - 0.4 * (1)
    elif shape in ['circular', 'circle', 'circulo', 'circular']:
        factor = 1 - 0.4 * (0)
    return factor


def canada_factor_c(phi, effective_width, effective_length):
    '''Shape factor for the cohesion term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the cohesion term in the bearing capacity
        equation [ ]
    '''

    if effective_length == 0:
        return np.nan
    # ref [3] table 10.2 Scs factor
    return 1 + effective_width / effective_length * bearing_factor_nq(phi) /bearing_factor_nc(phi)


def canada_factor_q(phi, effective_width, effective_length):
    '''Shape factor for the surcharge term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the surcharge term in the bearing capacity
        equation [ ]
    '''
    if effective_length == 0:
        return np.nan
    # ref [3] table 10.2 Sqs factor
    return 1 + effective_width / effective_length * np.tan(np.radians(phi))


def canada_factor_g(effective_width, effective_length):
    '''Shape factor for the soil weight term in the bearing capacity
    equation according to the Canadian Engineering Foundation Manual
    [3]_ (table 10.2). 

    Parameters
    ----------
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m]

    Returns
    -------
    float, int
        Shape factor for the soil weight term in the bearing capacity
        equation [ ]
    '''
    if effective_length == 0:
        return np.nan
    # ref [3] table 10.2 Sgs factor
    return 1 - 0.4 * effective_width / effective_length


def meyerhof_factor_c(phi, effective_width, effective_length):
    '''Shape factor for the cohesion term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3).

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the cohesion term in the bearing capacity
        equation [ ]
    '''
    # ref [1] table 4-3
    if effective_length == 0:
        return np.nan
    nphi = np.tan(np.radians(45 + phi / 2))**2
    return 1 + 0.2 * nphi * effective_width / effective_length


def meyerhof_factor_q(phi, effective_width, effective_length):
    '''Shape factor for the surcharge term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3).

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the surcharge term in the bearing capacity
        equation [ ]
    '''

    # ref [1] table 4-3
    if effective_length == 0:
        return np.nan
    if phi == 0:
        return 1
    elif phi > 10:
        nphi = np.tan(np.radians(45 + phi / 2))**2
        return 1 + 0.1 * nphi * effective_width / effective_length
    elif 0 < phi and phi <=10:
        nphi = np.tan(np.radians(45 + 10 / 2))**2
        factor10 = 1 + 0.1 * nphi * effective_width / effective_length
        factor0 = 1
        return (factor10 - factor0) * phi / 10 + factor0
    else:
        return np.nan


def meyerhof_factor_g(phi, effective_width, effective_length):
    '''Shape factor for the soil weight term in the bearing capacity
    equation according to Meyerhof [5]_ [6]_ as stated in the USACE
    manual [1]_ (table 4-3).

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the soil weight term in the bearing capacity
        equation [ ]
    '''
    # ref [1] table 4-3
    return meyerhof_factor_q(phi, effective_width, effective_length)


def hansen_factor_c(phi, effective_width, effective_length):
    '''Shape factor for the cohesion term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-3). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the cohesion term term in the bearing
        capacity equation [ ]
    '''

    # ref [1] table 4-5
    if effective_length == 0:
        return np.nan
    if phi == 0 :
        return 0.2 * effective_width / effective_length
    elif phi > 0:
        nqfactor = bearing_factor_nq(phi)
        ncfactor = bearing_factor_nc(phi)
        return 1 + nqfactor / ncfactor * effective_width / effective_length
    else:
        return np.nan


def hansen_factor_q(phi, effective_width, effective_length):
    '''Shape factor for the surcharge term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-3). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the surcharge term term in the bearing
        capacity equation [ ]
    '''

    # ref [1] table 4-5
    if effective_length == 0:
        return np.nan
    if phi == 0 :
        return 1
    elif phi > 0:
        return 1 + effective_width / effective_length * np.tan(np.radians(phi))
    else:
        return np.nan


def hansen_factor_g(phi, effective_width, effective_length):
    '''Shape factor for the soil weight term in the bearing capacity
    equation according to Hansen [7]_ as stated in the USACE
    manual [1]_ (table 4-5). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the soil weight term term in the bearing
        capacity equation [ ]
    '''

    # ref [1] table 4-5
    if effective_length == 0:
        return np.nan
    if phi == 0:
        return 1
    elif phi > 0:
        return 1 - 0.4 * effective_width / effective_length
    else:
        return np.nan


def vesic_factor_c(phi, effective_width, effective_length):
    '''Shape factor for the cohesion term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the cohesion term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-6
    if effective_length == 0:
        return np.nan
    if phi == 0 :
        return 0.2 * effective_width / effective_length
    elif phi > 0:
        nqfactor = bearing_factor_nq(phi)
        ncfactor = bearing_factor_nc(phi)
        return 1 + nqfactor * effective_width / (ncfactor * effective_length)
    else:
        return np.nan


def vesic_factor_q(phi, effective_width, effective_length):
    '''Shape factor for the surcharge term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the surcharge term term in the bearing
        capacity equation [ ]
    '''
    # ref [1] table 4-6
    if effective_length == 0:
        return np.nan
    if phi == 0 :
        return 1
    elif phi > 0:
        return 1 + effective_width / effective_length * np.tan(np.radians(phi))
    else:
        return np.nan


def vesic_factor_g(phi, effective_width, effective_length):
    '''Shape factor for the soil weight term in the bearing capacity
    equation according to Vesic [8]_ [9]_  as stated in the USACE
    manual [1]_ (table 4-6). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 

    Returns
    -------
    float, int
        Shape factor for the soil weight term term in the bearing
        capacity equation [ ]
    '''

    # ref [1] table 4-6
    if effective_length == 0:
        return np.nan
    if phi == 0:
        return 1
    elif phi > 0:
        return 1 - 0.4 * effective_width / effective_length
    else:
        return np.nan