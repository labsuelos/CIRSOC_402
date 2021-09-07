'''Module with all the constants
'''

# Error and warning messages language
#LANGUAGE = 'EN'
LANGUAGE = 'ES'

# Supported shapes for the bearing capacity
BEARINGSHAPEDICT = {}
BEARINGSHAPEDICT['EN'] = ['rectangle', 'square',  'circular']
BEARINGSHAPEDICT['ES'] = ['rectangulo', 'cuadrado', 'cuadrada', 'circular', 'circulo']
BEARINGSHAPE = list(set(sum(BEARINGSHAPEDICT.values(), [])))
# Supported methods for the bearing capacity
BEARINGMETHODDICT = {}
BEARINGMETHODDICT['EN'] = ['vesic', 'hansen', 'meyerhof', 'eurocode 7', 'canada'] # , 'vesic pile']
BEARINGMETHODDICT['ES'] = ['vesic', 'hansen', 'meyerhof', 'eurocode 7', 'canada'] #, 'vesic pilote']
BEARINGMETHOD = list(set(sum(BEARINGMETHODDICT.values(), [])))
# Supported methods for bearing factods
BEARINGFACTORS = ['cirsoc', 'canada', 'meyerhof', 'hansen', 'vesic']
DEFAULTBEARINGFACTORS = 'cirsoc'

# Supported codes
STANDARD = ['cirsoc', 'asce', 'eurocode', 'canada']
DEFAULTSTANDARD = 'cirsoc'

# Load states
LOAD = ['D', 'Di', 'E', 'F', 'Fa', 'H', 'L', 'Lr', 'R', 'S', 'T', 'W', 'Wi']