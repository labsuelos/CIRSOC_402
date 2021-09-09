'''Module with the load inclination factors formulas for the bearing
capacity equation
'''
import numpy as np

from cirsoc_402.constants import DEFAULTBEARINGFACTORS
from cirsoc_402.bearing.bearing_factors import bearing_factor_nc
from cirsoc_402.bearing.bearing_factors import bearing_factor_nq



def load_inclination_factors(phi, cohesion, effective_width,
                             effective_length, vertical_load,
                             horizontal_load, load_orientation,
                             factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the cohesion, surcharge and soil weight terms of the bearing
    capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        effective width of the equivalent rectangular load area [m] 
    effective_length : float, int
        effective length of the equivalent rectangular load area [m] 
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        factor_c = cirsoc_factor_c()
        factor_q = cirsoc_factor_q()
        factor_g = cirsoc_factor_g()
    elif factors == 'canada':
        factor_c = canada_factor_c(phi, cohesion, effective_width,
                                   effective_length, vertical_load,
                                   horizontal_load, load_orientation)
        factor_q = canada_factor_q(phi, cohesion, effective_width,
                                   effective_length, vertical_load,
                                   horizontal_load, load_orientation)
        factor_g = canada_factor_g(phi, cohesion, effective_width,
                                   effective_length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'meyerhof':
        factor_c = meyerhof_factor_c(phi, vertical_load, horizontal_load, load_orientation)
        factor_q = meyerhof_factor_q(phi, vertical_load, horizontal_load, load_orientation)
        factor_g = meyerhof_factor_g(phi, vertical_load, horizontal_load)
    elif factors == 'hansen':
        factor_c = hansen_factor_c(phi, base_adhesion, effective_width, effective_length,
                                   vertical_load, horizontal_load)
        factor_q = hansen_factor_q(phi, base_adhesion, effective_width,
                                   effective_length, vertical_load, horizontal_load)
        factor_g = hansen_factor_g(phi, base_adhesion, effective_width,
                                   effective_length, vertical_load, horizontal_load,
                                   base_inclination)
    elif factors == 'vesic':
        factor_c = vesic_factor_c()
        factor_q = vesic_factor_q()
        factor_g = vesic_factor_g()
    else:
        factor_c = np.nan
        factor_q = np.nan
        factor_g = np.nan

    return factor_c, factor_q, factor_g


def load_inclination_factor_c(phi, cohesion, effective_width,
                              effective_length, vertical_load, horizontal_load,
                              load_orientation, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the cohesion term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_c()
    elif factors == 'canada':
        return canada_factor_c(phi, cohesion, effective_width,
                               effective_length, vertical_load,
                               horizontal_load, load_orientation)
    elif factors == 'meyerhof':
        return meyerhof_factor_c(phi, vertical_load, horizontal_load, load_orientation)
    elif factors == 'hansen':
        return hansen_factor_c(phi, base_adhesion, effective_width, effective_length,
                               vertical_load, horizontal_load)
    elif factors == 'vesic':
        return vesic_factor_c()
    else:
        return np.nan


def load_inclination_factor_q(phi, cohesion, effective_width,
                              effective_length, vertical_load, horizontal_load,
                              load_orientation, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the surcharge term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_q()
    elif factors == 'canada':
        return canada_factor_q(phi, cohesion, effective_width,
                                  effective_length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'meyerhof':
        return meyerhof_factor_q(phi, vertical_load, horizontal_load)
    elif factors == 'meyerhof':
        return hansen_factor_q(phi, base_adhesion, effective_width,
                               effective_length, vertical_load, horizontal_load)
    elif factors == 'meyerhof':
        return vesic_factor_q()
    else:
        return np.nan


def load_inclination_factor_g(phi, cohesion, effective_width,
                              effective_length, vertical_load, horizontal_load,
                              load_orientation, factors=DEFAULTBEARINGFACTORS):
    '''Dimensionless correction factors due to the load inclination
    for the soil weight term of the bearing capacity equation.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    '''

    factors = factors.lower()
    if factors == 'cirsoc':
        return cirsoc_factor_g()
    elif factors == 'canada':
        return canada_factor_g(phi, cohesion, effective_width,
                                   effective_length, vertical_load,
                                   horizontal_load, load_orientation)
    elif factors == 'meyerhof':
        return meyerhof_factor_g(phi, vertical_load, horizontal_load)
    elif factors == 'hansen':
        return hansen_factor_g(phi, base_adhesion, effective_width,
                               effective_length, vertical_load, horizontal_load,
                               base_inclination)
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


def canada_factor_c(phi, cohesion, effective_width, effective_length,
                    vertical_load, horizontal_load, load_orientation):
    '''Load inclination factor for the cohesion term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
   effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    if vertical_load == 0  and horizontal_load == 0:
        return 1
    # ref [3] table 10.2 factor Sci
    factor_m = canada_factor_m(effective_width, effective_length, load_orientation)
    if phi==0:
        return 1 - factor_m * horizontal_load / (effective_width * effective_length * cohesion * bearing_factor_nc(phi))
    else:
        factor_q = canada_factor_q(phi, cohesion, effective_width, effective_length, vertical_load, horizontal_load, load_orientation)
        factor = factor_q - (1 - factor_q) / (bearing_factor_nc(phi) * np.tan(np.radians(phi)))
        return factor


def canada_factor_q(phi, cohesion, effective_width, effective_length,
                    vertical_load, horizontal_load, load_orientation):
    '''Load inclination factor for the surcharge term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    factor_m = canada_factor_m(effective_width, effective_length, load_orientation)
    factor = (1 - horizontal_load / (vertical_load + effective_width * effective_length * cohesion / np.tan(np.radians(phi)))) ** factor_m
    return factor


def canada_factor_g(phi, cohesion, effective_width, effective_length,
                    vertical_load, horizontal_load, load_orientation):
    '''Load inclination factor for the soil weight term in the bearing
    capacity equation according to the Canadian Engineering Foundation
    Manual [3]_ (table 10.2). 

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    cohesion : float, int
        Soil cohesion [kPa]
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    factor_m = canada_factor_m(effective_width, effective_length, load_orientation)
    factor = (1 - horizontal_load / (vertical_load + effective_width * effective_length * cohesion / np.tan(np.radians(phi)))) ** (factor_m + 1)
    return factor


def canada_factor_m(effective_width, effective_length, load_orientation):
    '''Auxiliary factor n used in the computation of the load
    inclination factors in the bearing capacity equation according to
    the Canadian Engineering Foundation Manual [3]_
    (table 10.2, note 2).

    Parameters
    ----------
    effective_width : float, int
        Effective width of the equivalent rectangular load area [m]
    effective_length : float, int
        Effective length of the equivalent rectangular load area [m]
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
    mb = (2 + effective_width / effective_length) / (1 + effective_width / effective_length)
    ml = (2 + effective_length /effective_width) / (1 + effective_length / effective_width)
    return ml * np.cos(np.radians(load_orientation))**2 + mb * np.sin(np.radians(load_orientation))**2


def meyerhof_factor_c(phi, vertical_load, horizontal_load, load_orientation):
    if vertical_load == 0  and horizontal_load == 0:
        return 1
    elif vertical_load == 0 or load_orientation != 90:
        return np.nan
    theta = np.rad2deg(np.arctan(horizontal_load / vertical_load))
    if phi == 0:
        return 1 - theta / 90
    elif phi > 0:
        return (1 - theta / 90)**2
    else:
        return np.nan


def meyerhof_factor_q(phi, vertical_load, horizontal_load, load_orientation):
    return meyerhof_factor_c(phi, vertical_load, horizontal_load, load_orientation)


def meyerhof_factor_g(phi, vertical_load, horizontal_load):
    if vertical_load == 0  and horizontal_load == 0:
        return 1
    elif vertical_load == 0:
        return np.nan
    theta = np.rad2deg(np.arctan(horizontal_load / vertical_load))
    if phi == 0:
        return 1
    elif theta <= phi:
        return (1 - theta / phi)**2
    elif phi < theta:
        return 0
    else:
        return np.nan


def hansen_factor_c(phi, base_adhesion, effective_width, effective_length,
                    vertical_load, horizontal_load):
    if phi == 0:
        area = effective_width * effective_length
        return (1 - (1 - horizontal_load / (area * base_adhesion))**(1/2) ) / 2 
    elif phi > 0:
        qfactor = hansen_factor_q(phi, base_adhesion, effective_width,
                                  effective_length, vertical_load,
                                  horizontal_load)
        nqfactor = bearing_factor_nq(phi)
        return qfactor - (1 - qfactor) / (nqfactor - 1)
    else:
        return np.nan


def hansen_factor_q(phi, base_adhesion, effective_width, effective_length,
                    vertical_load, horizontal_load):
    area = effective_width * effective_length
    return (1 - 0.5 * horizontal_load / (vertical_load + area * base_adhesion / np.tan(np.radians(phi))))**5


def hansen_factor_g(phi, base_adhesion, effective_width, effective_length,
                    vertical_load, horizontal_load, base_inclination):
    if base_inclination == 0:
        area = effective_width * effective_length
        return (1 - 0.7 * horizontal_load / (vertical_load + area * base_adhesion / np.tan(np.radians(phi))))**5
    elif base_inclination > 0:
        return (1 - (0.7  - base_inclination / 450) * horizontal_load / (vertical_load + area * base_adhesion / np.tan(np.radians(phi))))**5
    else:
        return np.nan


def vesic_factor_c():
    return 1


def vesic_factor_q():
    return 1


def vesic_factor_g():
    return 1


def vesic_factor_m():
    return 1
