'''Module with the bearing factors formulas for the bearing capacity
equation
'''
import numpy as np

from cirsoc_402.constants import  BEARINGMETHOD


def bearing_factor_nq(phi):
    '''Bearing factor Nq of the surcharge term in the bearing
    capacity equation for shallow foundations [1]_ [2]_ [3]_ [4]_.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]

    Returns
    -------
    float
        Bearing factor of the surcharge term in the bearing capacity
        equation for shallow foundations [ ].
    '''
    if phi == 0:
        # ref [3] eq. 10.7
        return 1
    phi = np.radians(phi)
    factor = (np.exp(np.pi * np.tan(phi))) * ((np.tan(0.25 * np.pi + 0.5 * phi))**2)
    return factor


def bearing_factor_nc(phi):
    '''Bearing factor Nc of the cohesion term in the bearing
    capacity equation for shallow foundations [1]_ [2]_ [3]_ [4]_.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]

    Returns
    -------
    float
        Bearing factor of the cohesion term in the bearing capacity
        equation for shallow foundations [ ].
    '''
    if phi == 0:
        # ref [3] eq. 10.7
        factor = 2 + np.pi
    else:
        factor = (bearing_factor_nq(phi) - 1) * (1 / np.tan(np.radians(phi)))
    return factor


def bearing_factor_ng(phi, method, ground_inclination=0):
    '''Bearing factor Ngamma of the soil weight term in the
    bearing capacity equation for shallow foundations [1]_ [2]_ [3]_
    [4]_.

    Parameters
    ----------
    phi : float, int
        Soil friction angle [deg]
    method : str
        Calculation method for the soil weight bearing capacity factor.
        The supported methods can be seen with
        cirsoc_402.constants.BEARINGMETHOD.
    ground_inclination : float, int, optional
        Ground slope relative to the horizontal plane. Only used by the 
        canadian version of the bearing factor when phi=0. By default 0
        [deg]

    Returns
    -------
    float
        Bearing factor of the soil weight term in the bearing capacity
        equation for shallow foundations [ ]
    '''
    qfactor = bearing_factor_nq(phi)
    method = method.lower()
    if method == 'vesic':
        factor = 2 * (qfactor + 1) * np.tan(np.radians(phi)) #Vesic (1975)
    elif method == 'hansen':
        factor = 1.5 * (qfactor - 1) * np.tan(np.radians(phi)) #Hansen (1970)
    elif method == 'eurocode 7':
        factor = 2 * (qfactor - 1) * np.tan(np.radians(phi)) #Eurocode 7 (CEN 2004)
    elif method == 'meyerhof':
        factor = (qfactor - 1) * np.tan(1.4 * np.radians(phi)) #Meyerhof (1963)
    elif method == 'canada':
        if np.isnan(ground_inclination):
            return np.nan
        elif phi==0 and ground_inclination==0:
            # ref [3] eq. 10.8
            factor = 0
        elif phi==0 and ground_inclination>0:
            # ref [3] table 10.2 note 4
            factor = -2 * np.sin(np.radians(ground_inclination))
        else:
            # ref [3] eq. 10.4
            # factor = 0.0663 * np.exp(0.1623 * phi)
            # ref [4] eq. 10.5
            factor = 0.1054 * np.exp(0.1675 * phi)
    else:
        return np.nan
    return factor