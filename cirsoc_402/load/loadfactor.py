'''Module with the classess used for storing load factors for of the
load combination equations
'''

from dataclasses import dataclass, asdict, astuple

@dataclass
class LoadFactors():
    '''Load factors for each load in a load combination that defines a
    service or ultimate load state.

    Attributes
    -------
    D : float
        load factor for the dead load [ ]
    Di : float
        load factor for the weight of ice [ ]
    E : float
        load factor for the eqarthquake load [ ]
    F : float
        load factor for the load due to fluids with well-defined
        pressures and maximum heights  [ ]
    Fa : float
        load factor for the flood load [ ]
    H : float
        load factor for the load due to lateral earth pressure,
        ground water pressure, or pressure of bulk materials [ ]
    L : float
        load factor for the live load [ ]
    Lr : float
        load factor for the roof live load [ ]
    R : float
        load factor for the rain load [ ]
    S : float
        load factor for the snow load [ ]
    T : float
        load factor for the self-tensing load [ ]
    W : float
        load factor for the wind load [ ]
    Wi : float
        load factor for the wind-on-ice load [ ]

    Example 1
    ---------
    The load combination :math:`1.4 D` is defiend by:
    >>> from cirsoc_402.loadclass import LoadFactors
    >>> loadcomb = LoadFactors(D=1.4)
    >>> loadcomb

    The factors for all the loads are stored, but only the non zero
    ones are displayed. To see the full set of factors:
    >>> from cirsoc_402.loadclass import LoadFactors
    >>> from dataclass import asdict
    >>> loadcomb = LoadFactors(D=1.4)
    >>> asdict(loadcomb)

    Example 2
    ---------
    The load combination :math:`1.2 D + 1.6 Lr + 0.5 W` is defiend by:
    >>> from cirsoc_402.loadclass import LoadFactors
    >>> loadcomb = LoadFactors(D=1.2, Lr=1.6, R=0.5)
    >>> loadcomb

    '''
    D: float = 0
    Di: float = 0
    E: float = 0
    F: float = 0
    Fa: float = 0
    H: float = 0
    L: float = 0
    Lr: float = 0
    R: float = 0
    S: float = 0
    T: float = 0
    W: float = 0
    Wi: float = 0

    def __repr__(self):
        txt = ''
        nload = 0
        for loadid in self.__dataclass_fields__.keys():
            loadfactor = getattr(self, loadid)
            if loadfactor == 0:
                continue
            elif loadfactor==1:
                txtadd = loadid
                nload += 1
            elif loadfactor!=0:
                txtadd = "{:.2f} ".format(loadfactor) + loadid
                nload += 1
            if nload > 1:
                txtadd = ' + ' + txtadd
            txt += txtadd
        return txt


class LoadFactorDict(dict):
    '''Dictionary with all the load combinations corresponding to either
    a ultimate or service limit states. Each load combination must be
    defined with a LoadFactors object. 
    '''

    def __init__(self):
        super(LoadFactorDict, self).__init__()
    
    def asdict(self):
        '''Formats all the load combinations into a dictionary.

        Returns
        -------
        dict
            Dictionary with the dictionaries that contain the load
            factors for each load combination.
        '''
        outdict = {}
        for key in self.keys():
            outdict[key] = asdict(self[key])
        return outdict
