    '''Custom exception in cirsoc_402
    '''

from constants import LANGUAGE
from constants import BEARINGSHAPE, BEARINGMETHOD, BEARINGFACTORS

class BearingShapeError(Exception):
    '''Exception raised when the shape in the bearing capacity
    calculation requested by the user is not supported by the code.

    Attributes
    ----------
        shape :  str
            bearing capacity shape requested by the user
        mesage : str
            explanation of the error
    '''

    def __init__(self, shape):        
        self.shape = shape
        if LANGUAGE == 'EN':
            mesage = "Unsuported bering capacity shape ''{}'. Suported shapes are: "
            mesage = mesage + ', '.join(BEARINGSHAPEDICT['EN'])
        elif LANGUAGE == 'ES':
            mesage = "Forma de base '{}' no disponible. Las formas disponibles son: "
            mesage = mesage + ', '.join(BEARINGSHAPEDICT['ES'])
        self.mesage = mesage.format(shape)
        super().__init__(self.mesage)


class BearingMethodError(Exception):
    '''Exception raised when the bearing capacity calculation method
    requested by the user is not supported by the code.

    Attributes
    ----------
        method : str
            bering capacity method requested by the user
        mesage : str
            explanation of the error
    '''

    def __init__(self, method):        
        self.method = method
        if LANGUAGE == 'EN':
            mesage = "Unsuported bearing capacity method '{}'. Suported metod are: "
            mesage = mesage + ', '.join(BEARINGMETHOD['EN'])
        elif LANGUAGE == 'ES':
            mesage = ("Metodo de capacidad de carga '{}' no disponible. Los "
                      "metodos disponibles son")
            mesage = mesage + ', '.join(BEARINGMETHOD['ES'])
        self.mesage = mesage.format(method)
        super().__init__(self.mesage)


class BearingWidthError(Exception):
    '''Exception raised when the user failed to provide the width in for
    the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Input width not provided for a rectangular base. "
                      "Provide this value as keyword argument.")
        elif LANGUAGE == 'ES':
            mesage = ("El valor de la varaible width no fue provisto para una "
                      "base rectangular. Provea este valor como un keyword "
                      "argument")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingLengthError(Exception):
    '''Exception raised when the user failed to provide the length in
    for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Input length not provided for a rectangular base. "
                      "Provide this value as keyword argument.")
        elif LANGUAGE == 'ES':
            mesage = ("El valor de la varaible length no fue provisto para una "
                      "base rectangular. Provea este valor como un keyword "
                      "argument")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingFactorsError(Exception):
    '''Exception raised when the the bearing capacity factor group
    requested by the user is not supported by the code.

    Attributes
    ----------
        factor : str
            bering capacity factor group requested by the user
        mesage : str
            explanation of the error
    '''

    def __init__(self, method):        
        self.factor = factor
        if LANGUAGE == 'EN':
            mesage = "Unsuported bearing capacity factors '{}'. Suported factors are: "
            mesage = mesage + ', '.join(BEARINGFACTORS)
        elif LANGUAGE == 'ES':
            mesage = ("Grupo de factores para capacidad de carga '{}' no "
                      "disponible. Las factores disponibles son: ")
            mesage = mesage + ', '.join(BEARINGFACTORS)
        self.mesage = mesage.format(method)
        super().__init__(self.mesage)