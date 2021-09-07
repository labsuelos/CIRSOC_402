'''Module with the functions necessary to compute the load capacity
of shallow foundations
'''

import numpy as np
import warnings

from cirsoc_402.constants import LANGUAGE
from cirsoc_402.constants import DEFAULTSTANDARD
from cirsoc_402.constants import BEARINGSHAPE
from cirsoc_402.constants import STANDARD
from cirsoc_402.constants import BEARINGMETHOD
from cirsoc_402.constants import BEARINGFACTORS

from cirsoc_402.exceptions import BearingShapeError
from cirsoc_402.exceptions import BearingWidthError
from cirsoc_402.exceptions import BearingLengthError
from cirsoc_402.exceptions import BearingLengthSquareError
from cirsoc_402.exceptions import BearingLengthCircularError
from cirsoc_402.exceptions import BearingDepthError
from cirsoc_402.exceptions import BearingGammaNgError
from cirsoc_402.exceptions import BearingGammaNqError
from cirsoc_402.exceptions import BearingPhiError
from cirsoc_402.exceptions import BearingCohesionError
from cirsoc_402.exceptions import StandardError
from cirsoc_402.exceptions import BearingMethodError
from cirsoc_402.exceptions import BearingFactorsError

class Footing():
    
    def __init__(self, shape, gamma_nq, gamma_ng, phi, cohesion, depth,
                 width, length=np.nan, base_inclination=0,
                 ground_inclination=0, standard=DEFAULTSTANDARD,
                 method=None ,factors=None):
        
        self._set_shape(shape, width, length)
        self._set_depth(depth)
        self._set_inclination(base_inclination, ground_inclination)
        self._set_soil_parameters(gamma_nq, gamma_ng, phi, cohesion)
        self._set_method(standard, method, factors)

    def _set_shape(self, shape, width, length):
        ''' Check that the foundation shape are valid and set the
        ``_shape``, ``_widht`` and ``_length`` parameters values.

        Parameters
        ----------
        shape : str
            Shape of the foundation. The supported shapes can be seen
            with cirsoc_402.constants.BEARINGSHAPE.
        width : float, int
            Width of the foundation. In ciruclar foundations the
            diameter [m]
        length : float, int
            Length of the foundation for rectangular foundations. For
            circular foundations no value needs to be provided or
            np.nan. For square foundations no value needs to be provided
            or the same value as the width.

        Raises
        ------
        BearingShapeError
            Exception raised when the shape in the bearing capacity
            calculation requested by the user is not supported by the
            code.
        BearingWidthError
            Exception raised when the user failed to provide the width
            in for bearing capacity formula
        BearingLengthError
            Exception raised when the user failed to provide the length
            in for the bearing capacity formula
        BearingLengthCircularError
            Exception raised when the the bearing capacity of a circular
            base is requested but the length doesn't match the width
        BearingLengthSquareError
            Exception raised when the the bearing capacity of a square
            base is requested but the length doesn't match the width
        '''
        if shape not in BEARINGSHAPE:
            raise BearingShapeError(shape)
    
        if width is not isinstance(float, int) or np.isnan(width):
            raise BearingWidthError()
        if width <=0:
            raise BearingWidthError()
        
        if shape in ['rectangle', 'rectangulo']:
            if np.isnan(length) or length==0:
                raise BearingLengthError()

        
        if shape in ['circular', 'circular', 'circulo']:
            if np.isnan(length):
                length = width
            elif length is not isinstance(float, int):
                raise BearingLengthError()
            elif length != width:
                raise BearingLengthCircularError()

        if shape in ['square', 'cuadrado', 'cuadrada']:
            if np.isnan(length):
                length = width
            elif length is not isinstance(float, int):
                raise BearingLengthError()
            elif length != width:
                raise BearingLengthSquareError()
        
        self._shape = shape
        if length >= width:
            self._length = length
            self._width = width
        else:
            self._length = width
            self._width = length
    
    def _set_depth(self, depth):

        if depth is not isinstance(float, int) or np.isnan(depth):
            raise BearingDepthError()
        elif depth<=0:
            raise BearingDepthError()

        self._depth = depth
        if depth > 2.5 * self._width:
            if LANGUAGE == 'EN':
                mesage = "Base depth is larger than 2.5 times it's depth."
            elif LANGUAGE == 'ES':
                mesage = 'Profundidad (depth) de la base es mayor a 2.5 veces su ancho (width).'
            warnings.warn(mesage)
    
    def _set_inclination(self, base_inclination, ground_inclination):

        if not isinstance(base_inclination, (int, float)):
            raise TypeError("Base inclination must be specified by a real number.")
        
        if not isinstance(ground_inclination, (int, float)):
            raise TypeError("Ground inclination must be specified by a real number.")
        
        self._base_inclination = base_inclination
        self._ground_inclination = ground_inclination

    def _set_soil_parameters(self, gamma_nq, gamma_ng, phi, cohesion):

        if gamma_nq is not isinstance(float, int) or np.isnan(gamma_nq):
            raise BearingGammaNqError()
        elif gamma_nq < 0 :
            raise BearingGammaNqError()
        self._gamma_nq = gamma_nq
        
        if gamma_ng is not isinstance(float, int) or np.isnan(gamma_ng):
            raise BearingGammaNgError()
        elif gamma_ng < 0 :
            raise BearingGammaNgError()
        self._gamma_ng = gamma_ng

        if phi is not isinstance(float, int):
            raise BearingPhiError()
        elif np.isnan(phi):
            phi = 0
        elif phi < 0:
            raise BearingPhiError()
        self._phi = phi

        if cohesion is not isinstance(float, int):
            raise BearingCohesionError()
        elif np.isnan(cohesion):
            cohesion = 0
        elif cohesion < 0:
            raise BearingCohesionError()
        self._cohesion = cohesion
    
    def _set_method(self, standard, method, factors):

        
        if standard is None and method is None and factors is None:
            raise RuntimeError()

        if method is not None and not isinstance(method, str):
            raise RuntimeError()
        elif isinstance(method, str):
            method = method.lower()
        if method is not None and method not in BEARINGMETHOD:
            raise BearingMethodError(method)
        
        if factors is not None and not isinstance(factors, str):
            raise RuntimeError()
        elif isinstance(factors, str):
            factors = factors.lower() 
        if factors is not None and factors not in BEARINGFACTORS:
            raise BearingFactorsError(factors)

        if standard is not None and not isinstance(factors, str):
            raise RuntimeError()
        elif isinstance(factors, str):
            standard = standard.lower()
        if standard is not None and standard not in STANDARD:
            raise StandardError(standard)
        
        
        if standard == 'cirsoc':
            factors = 'cirsoc'
            method = 'vesic'
        elif standard == 'canada':
            factors = 'canada'
            method = 'canada'
        elif standard == 'usace':
            factors = 'usace'
            method = 'vesic'
        
        self._standard = standard
        self._method = method
        self._factors = factors








