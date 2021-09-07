'''Custom exception in cirsoc_402
'''

from cirsoc_402.constants import LANGUAGE
from cirsoc_402.constants import BEARINGSHAPE, BEARINGMETHOD, BEARINGFACTORS
from cirsoc_402.constants import BEARINGSHAPEDICT, BEARINGMETHODDICT
from cirsoc_402.constants import STANDARD, DEFAULTSTANDARD

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
            mesage = mesage + ', '.join(BEARINGMETHODDICT['EN'])
        elif LANGUAGE == 'ES':
            mesage = ("Metodo de capacidad de carga '{}' no disponible. Los "
                      "metodos disponibles son")
            mesage = mesage + ', '.join(BEARINGMETHODDICT['ES'])
        self.mesage = mesage.format(method)
        super().__init__(self.mesage)


class BearingWidthError(Exception):
    '''Exception raised when the user failed to provide a valid width
    for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid width for a shallow foundation. Please "
                     "provide a positive real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Ancho (width) de base invalido. Por favor especifique un "
                      "numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingLengthError(Exception):
    '''Exception raised when the user failed to provide a valid length
    for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid length for a shallow foundation. Please "
                     "provide a positive real number or np.nan if the shape is "
                     "ciruclar or rectangular.")
        elif LANGUAGE == 'ES':
            mesage = ("Largo (length) de base invalido. Por favor provea un numero "
                      "real positivo o np.nan para bases ciruclares or cuadradas.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingDepthError(Exception):
    '''Exception raised when the user failed to provide a valid depth in
    for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid depth for a shallow foundation. Please "
                     "provide a positive real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Profundidad (depth) de base invalido. Por favor especifique "
                      "un numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingGammaNgError(Exception):
    '''Exception raised when the user failed to provide a valid unit
    weight for the soil weight term in for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid soil weight for the Ng term (gamma_ng) in the bearing "
                      "capacity equation. Please provide a positive real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Peso especifico invalido para el termino Ng de la "
                      "formula de capacidad de carga (gamma_ng). Por favor "
                      "especifique un numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingGammaNqError(Exception):
    '''Exception raised when the user failed to provide a valid unit
    weight for the surcharge term in for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid soil weight for the Nq term (gamma_nq) in the bearing "
                      "capacity equation. Please provide a positive real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Peso especifico invalido para el termino Nq de la "
                      "formula de capacidad de carga (gamma_nq). Por favor "
                      "especifique un numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingPhiError(Exception):
    '''Exception raised when the user failed to provide a valid firction
    angle for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid friction angle (phi) input. Please provide a"
                      " positive real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Angulo de friction (phi) invalido. Por favor "
                      "especifique un numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingCohesionError(Exception):
    '''Exception raised when the user failed to provide a valid cohesion
    for the bearing capacity formula

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = ("Invalid cohesion input. Please provide a positive"
                      " real number.")
        elif LANGUAGE == 'ES':
            mesage = ("Valor de cohesion invalido. Por favor "
                      "especifique un numero real positivo.")
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingFactorsError(Exception):
    '''Exception raised when the the bearing capacity factor group
    requested by the user is not supported by the code

    Attributes
    ----------
        factor : str
            bering capacity factor group requested by the user
        mesage : str
            explanation of the error
    '''

    def __init__(self, factor):        
        self.factor = factor
        if LANGUAGE == 'EN':
            mesage = "Unsuported bearing capacity factors '{}'. Suported factors are: "
            mesage = mesage + ', '.join(BEARINGFACTORS)
        elif LANGUAGE == 'ES':
            mesage = ("Grupo de factores para capacidad de carga '{}' no "
                      "disponible. Las factores disponibles son: ")
            mesage = mesage + ', '.join(BEARINGFACTORS)
        self.mesage = mesage.format(factor)
        super().__init__(self.mesage)


class BearingLengthSquareError(Exception):
    '''Exception raised when the the bearing capacity of a square base
    is requested but the length doesn't match the width

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''

    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = "The length and width of a square base must be the same."
        elif LANGUAGE == 'ES':
            mesage = "El larngo (length) y ancho (width) de una base cuadrada deben ser iguales."
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingLengthCircularError(Exception):
    '''Exception raised when the the bearing capacity of a circular base
    is requested but the length doesn't match the width

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''
    
    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = "The length and width of a circular base must be the same."
        elif LANGUAGE == 'ES':
            mesage = "El larngo (length) y ancho (width) de una base circular deben ser iguales."
        self.mesage = mesage
        super().__init__(self.mesage)


class BearingSizeError(Exception):
    '''Exception raised when the the bearing capacity of a circular base
    is requested but the length doesn't match the width

    Attributes
    ----------
        mesage : str
            explanation of the error
    '''
    
    def __init__(self):        
        if LANGUAGE == 'EN':
            mesage = "The width of the base must be smaller than the length."
        elif LANGUAGE == 'ES':
            mesage = "El ancho (width) de la base debe ser menor que el largo (length)."
        self.mesage = mesage
        super().__init__(self.mesage)


class StandardError(Exception):
    '''Exception raised when the standard requested by the user is not
    supported by the code.

    Attributes
    ----------
        standard :  str
            standard requested by the user
        mesage : str
            explanation of the error
    '''

    def __init__(self, standard):        
        self.standard = standard
        if LANGUAGE == 'EN':
            mesage = "Unsuported desgin standard ''{}'. Suported design standards are: "
        elif LANGUAGE == 'ES':
            mesage = "Estandard de diseno '{}' no disponible. Las estandares disponbles son son: "
        mesage = mesage + ', '.join(DEFAULTSTANDARD)
        self.mesage = mesage.format(standard)
        super().__init__(self.mesage)

