


from dataclasses import dataclass, asdict, astuple
from cirsoc_402.constants import LANGUAGE, LOAD, STANDARD, DEFAULTSTANDARD

from asce import ultimate as asceultimate
from asce import service as asceservice


def load_type_validator(name, loadtype):
    '''Validates that the load type belongs to the supported types
    as defined by cirsoc_402.constants.LOAD

    Parameters
    ----------
    name : [type]
        [description]
    loadtype : str
        load type

    Raises
    ------
    ValueError
        Load type requested by the user is not supported
    '''
    if value not in LOAD:
        if LANGUAGE == 'EN':
            mesage = "Unsupported load type {}. Supported types are: "
        elif LANGUAGE == 'ES':
            mesage = ("Tipo de carga no soportado {}. Los tipos de "
                      "carga disponibles son: ")
        mesage = mesage + ', '.join(LOAD)
        mesage = mesage.format(loadtype)
        raise ValueError(mesage)


class MyAttr:
     def __init__(self, type, validators=()):
          self.type = type
          self.validators = validators

     def __set_name__(self, owner, name):
          self.name = name

     def __get__(self, instance, owner):
          if not instance: return self
          return instance.__dict__[self.name]

     def __delete__(self, instance):
          del instance.__dict__[self.name]

     def __set__(self, instance, value):
          if not isinstance(value, self.type):
                raise TypeError(f"{self.name!r} values must be of type {self.type!r}")
          for validator in self.validators:
               validator(self.name, value)
          instance.__dict__[self.name] = value


def _rotation_matrix(theta):
    '''2D rotation matrix

    Parameters
    ----------
    theta : float, int
        rotation [deg]

    Returns
    -------
    np.ndarray
        rotation matrix
    '''
    t = np.radians(theta)
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])

@dataclass
class _LoadTypeBase:
    '''Base class for loads with the loadtype atrribute that is
    protected to only accept values included in
    cirsoc_402.constants.LOAD

    Attributes
    ----------
    loadtype : str
        load type accoding to cirsoc_402.constants.LOAD

    '''
    loadtype: str = MyAttr((str), [load_type_validator,])


@dataclass
class _LoadBase:
    '''Base class for the acting forces and moments in a load. The
    components of the force and moments are defined accoridng to the
    following convention. Arrows indicate positive directions. Right
    rule is used for computing moments. 


     ▲ My                 ▲ Mz                  ▲ Mz
     ▲ Qy                 ▲ Qz                  ▲ Qz                     
     ▲ y                  ▲ z                   ▲ z
     |                    |                     |
     |                    | y                   |
     o------► -► -►       o------► -► -►        o------► -► -►
    z       x Qx Mx              x Qx Mx       x       y  Qy My
    
    The axial loads and moments are expressed relative to a reference
    point of coordinates (xcoord, ycoord, zcoord) with the axis
    rotated (xtheta, ytheta, ztheta). The reference coordinate system
    can be moved and rotated by the user, updating the force and moment
    values.

    Attributes
    ----------
        xforce : float
            force in the x direction [kN]
        yforce : float
            force in the y direction [kN]
        zforce : float
            force in the z direction [kN]
        xmoment : float
            bending moment in the x direction [kN]
        ymoment : float
            bending moment in the y direction [kN]
        zmoment : float
            bending moment in the z direction [kN]
        xcoord : float
            x coordinate of the reference point for moment calculation
            [m]
        ycoord : float
            y coordinate of the reference point for moment calculation
            [m]
        zcoord : float
            z coordinate of the reference point for moment calculation
            [m]
    '''

    xforce: float = 0
    yforce: float = 0
    zforce: float = 0
    xmoment: float = 0
    ymoment: float = 0
    zmoment: float = 0
    xcoord: float = 0
    ycoord: float = 0
    zcoord: float = 0
    xtheta: float = 0
    ytheta: float = 0
    ztheta: float = 0

    def zshift(self, shift):
        '''Updates moments for a movement of the reference point in the
        z direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        self.xmoment = self.xmoment + self.xforce * shift
        self.ymoment = self.ymoment - self.yforce * shift
        self.zcoord = self.zcoord + shift
    
    def xshift(self, shift):
        '''Updates moments for a movement of the reference point in the
        x direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        self.zmoment = self.zmoment - self.yforce * shift
        self.ymoment = self.ymoment + self.zforce * shift
        self.xcoord = self.xcoord + shift

    def yshift(self, shift):
        '''Updates moments for a movement of the reference point in the
        y direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        self.zmoment = self.zmoment + self.xforce * shift
        self.xmoment = self.xmoment - self.zforce * shift
        self.ycoord = self.ycoord + shift
    
    def shift(self, xshift, yshift, zshift):
        '''Updates moments for a movement of the reference point in the
        x, y and z directions.

        Parameters
        ----------
        xshift : float
            movement of the reference point in the x direction [m]
        yshift : float
            movement of the reference point in the y direction [m]
        zshift : float
            movement of the reference point in the yz direction [m]
        '''
        self.xshift(xshift)
        self.yshift(yshift)
        self.zshift(zshift)
    
    def xrotate(self, theta):
        '''Updates forces and movements for a rotation of the reference
        axis arrond the x axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the x axis [deg]
        '''
        force = _rotation_matrix(theta).dot(np.array([self.yforce, self.zforce]))
        self.yforce = force[0]
        self.zforce = force[1]

        moment = _rotation_matrix(theta).dot(np.array([self.ymoment, self.zmoment]))
        self.ymoment = moment[0]
        self.zmoment = moment[1]
        self.xtheta = self.xtheta + theta

    def yrotate(self, theta):
        '''Updates forces and movements for a rotation of the reference
        axis arrond the y axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the y axis [deg]
        '''
        force = _rotation_matrix(theta).dot(np.array([self.xforce, self.zforce]))
        self.xforce = force[0]
        self.zforce = force[1]

        moment = _rotation_matrix(theta).dot(np.array([self.xmoment, self.zmoment]))
        self.xmoment = moment[0]
        self.zmoment = moment[1]
        self.ytheta = self.ytheta + theta
    
    def zrotate(self, theta):
        '''Updates forces and movements for a rotation of the reference
        axis arrond the z axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the z axis [deg]
        '''
        force = _rotation_matrix(theta).dot(np.array([self.xforce, self.yforce]))
        self.xforce = force[0]
        self.yforce = force[1]

        moment = _rotation_matrix(theta).dot(np.array([self.xmoment, self.ymoment]))
        self.xmoment = moment[0]
        self.ymoment = moment[1]
        self.ztheta = self.ztheta + theta

    def rotate(self, xtheta, ytheta, ztheta):
        '''Updates forces and movements for a rotation of the reference
        axis

        Parameters
        ----------
        xtheta : float, int
            rotation arround the x axis [deg]
        ytheta : float, int
            rotation arround the y axis [deg]
        ztheta : float, int
            rotation arround the z axis [deg]
        '''
        self.xrotate(xtheta)
        self.yrotate(ytheta)
        self.zrotate(ztheta)

    def toorigin(self):
        '''Revertns the reference point to the origin, updating the
        forces and moments.
        '''
        self.xshift(-self.xcoord)
        self.yshift(-self.ycoord)
        self.zshift(-self.zcoord)
        self.xrotate(-self.xtheta)
        self.yrotate(-self.ytheta)
        self.zrotate(-self.ztheta)

    def resetorigin(self):
        '''Sets the current reference point as the origin, seting the
        coordinates and angles to zero.
        '''
        self.xcoord = 0
        self.ycoord = 0
        self.zcoord = 0
        self.xtheta = 0
        self.ytheta = 0
        self.ztheta = 0

    def __add__(self, other):
        '''Adds two Load objects, preserving the reference point of the
        first of them.

        Parameters
        ----------
        other : Load
            Load object to be added

        Raises
        ------
        TypeError
            Input is not a Load object.
        '''
        if not isinstance(other, Load):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        other.toorigin()
        
        other.shift(self.xcoord, self.ycoord, self.zcoord)
        other.rotate(self.xtheta, self.ytheta, self.ztheta)
        
        selfcopy = copy.deepcopy(self)
        selfcopy.zforce += other.zforce
        selfcopy.xforce += other.xforce
        selfcopy.yforce += other.yforce
        selfcopy.xmoment += other.xmoment
        selfcopy.ymoment += other.ymoment
        selfcopy.zmoment += other.zmoment
        
        return(selfcopy)

    def __sub__(self, other):
        '''Substracts two Load objects, preserving the reference point
        of the first of them.

        Parameters
        ----------
        other : Load
            Load object to be substracted

        Raises
        ------
        TypeError
            Input is not a Load object.
        '''
        if not isinstance(other, Load):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        other.toorigin()
        
        other.shift(self.xcoord, self.ycoord, self.zcoord)
        other.rotate(self.xtheta, self.ytheta, self.ztheta)
        
        selfcopy = copy.deepcopy(self)
        selfcopy.zforce -= other.zforce
        selfcopy.xforce -= other.xforce
        selfcopy.yforce -= other.yforce
        selfcopy.xmoment -= other.xmoment
        selfcopy.ymoment -= other.ymoment
        selfcopy.zmoment -= other.zmoment
        
        return(selfcopy)
    
    def __mul__(self, mulby):
        '''Multiplies the forces and moments by a float or integer

        Parameters
        ----------
        mulby : float, int
            Float or integer to be multiplied to the forces and moments
            components

        Raises
        ------
        TypeError
            The value by which the forces are to be multiplied is not
            a float or int.
        '''

        if not isinstance(mulby, (int, float)):
            raise TypeError('Forces can only be mumplitied by a float or int.')
            
        selfcopy = copy.deepcopy(self)
        selfcopy.xforce = mulby * self.xforce
        selfcopy.yforce = mulby * self.yforce
        selfcopy.zforce = mulby * self.zforce
        selfcopy.xmoment = mulby * self.xmoment
        selfcopy.ymoment = mulby * self.ymoment
        selfcopy.zmoment = mulby * self.zmoment
        
        return(selfcopy)
    
    def __rmul__(self, mulby):
        '''Right multiplies the forces and moments by a float or integer

        Parameters
        ----------
        mulby : float, int
            Float or integer to be multiplied to the forces and moments
            components

        Raises
        ------
        TypeError
            The value by which the forces are to be multiplied is not
            a float or int.
        '''
        return self.__mul__(mulby)

    def __truediv__(self, divby):
        '''Dividez the forces and moments by a float or integer

        Parameters
        ----------
        divby : float, int
            Float or integer use to divide the forces and moments
            components

        Raises
        ------
        TypeError
            The value by which the forces are to be divided is not
            a float or int.
        ValueError
            Division by zero.
        '''
        if not isinstance(divby, (int, float)):
            raise TypeError('Forces can only be divided by a float or int.')
        
        if divby == 0:
            raise ValueError('Division by zero.')
            
        selfcopy = copy.deepcopy(self)
        selfcopy.xforce = self.xforce / divby
        selfcopy.yforce = self.yforce / divby
        selfcopy.zforce = self.zforce / divby
        selfcopy.xmoment = self.xmoment / divby
        selfcopy.ymoment = self.ymoment / divby
        selfcopy.zmoment = self.zmoment / divby
        
        return(selfcopy)


@dataclass
class Load(_LoadBase, _LoadTypeBase):
    '''Class for the acting forces and moments in a load. The component
    of the force and moments are defined accoridng to the following
    convention. Arrows indicate positive directions. Right rule is used
    for computing moments. 


     ▲ My                 ▲ Mz                  ▲ Mz
     ▲ Qy                 ▲ Qz                  ▲ Qz                     
     ▲ y                  ▲ z                   ▲ z
     |                    |                     |
     |                    | y                   |
     o------► -► -►       o------► -► -►        o------► -► -►
    z       x Qx Mx              x Qx Mx       x       y  Qy My
    
    The axial loads and moments are expressed relative to a reference
    point of coordinates (xcoord, ycoord, zcoord) with the axis
    rotated (xtheta, ytheta, ztheta). The reference coordinate system
    can be moved and rotated by the user, updating the force and moment
    values.

    Attributes
    ----------
        loadtype : str
            load type accoding to cirsoc_402.constants.LOAD
        xforce : float
            force in the x direction [kN]
        yforce : float
            force in the y direction [kN]
        zforce : float
            force in the z direction [kN]
        xmoment : float
            bending moment in the x direction [kN]
        ymoment : float
            bending moment in the y direction [kN]
        zmoment : float
            bending moment in the z direction [kN]
        xcoord : float
            x coordinate of the reference point for moment calculation
            [m]
        ycoord : float
            y coordinate of the reference point for moment calculation
            [m]
        zcoord : float
            z coordinate of the reference point for moment calculation
            [m]
        name : str
            name of the load
    
    Example 1
    ---------
    '''

    name: str = ''


@dataclass
class LoadGroup():
    '''Set of loads (D, L, Lr, ...) acting simultaneosly on a same
    place.

    Attributes
    -------
    D : _LoadBase
        dead load
    Di : _LoadBase
        weight of ice
    E : _LoadBase
        eqarthquake load
    F : _LoadBase
        load due to fluids with well-defined pressures and maximum
        heights 
    Fa : _LoadBase
        flood load
    H : _LoadBase
        load due to lateral earth pressure, ground water pressure, or
        pressure of bulk materials
    L : _LoadBase
        live load
    Lr : _LoadBase
        roof live load
    R : _LoadBase
        rain load
    S : _LoadBase
        snow load
    T : _LoadBase
        self-tensing load
    W : _LoadBase
        wind load
    Wi : _LoadBase
        wind-on-ice load

    Example 1
    ---------
    '''

    D: Load = _LoadBase()
    Di: Load = _LoadBase()
    E: Load = _LoadBase()
    F: Load = _LoadBase()
    Fa: Load = _LoadBase()
    H: Load = _LoadBase()
    L: Load = _LoadBase()
    Lr: Load = _LoadBase()
    R: Load = _LoadBase()
    S: Load = _LoadBase()
    T: Load = _LoadBase()
    W: Load = _LoadBase()
    Wi: Load = _LoadBase()
    
    def xshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the x direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        for loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.xshift(shift)
            setattr(self, loadid, load)
    
    def yshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the y direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        for loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.yshift(shift)
            setattr(self, loadid, load)
    
    def zshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the z direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        for loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.zshift(shift)
            setattr(self, loadid, load)
    
    def shift(self, xshift, yshift, zshift):
        def zshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the x, y and z directions.

        Parameters
        ----------
        xshift : float
            movement of the reference point in the x direction [m]
        yshift : float
            movement of the reference point in the y direction [m]
        zshift : float
            movement of the reference point in the z direction [m]
        '''
        for loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.zshift(shift)
            setattr(self, loadid, load)
        self.xshift(xshift)
        self.yshift(yshift)
        self.zshift(zshift)

    def xrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the x axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the x axis [deg]
        '''
        or loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.xrotate(theta)
            setattr(self, loadid, load)
    
    def yrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the y axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the y axis [deg]
        '''
        or loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.yrotate(theta)
            setattr(self, loadid, load)
    
    def zrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the z axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the z axis [deg]
        '''
        or loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.zrotate(theta)
            setattr(self, loadid, load)

    def rotate(self, xtheta, ytheta, ztheta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the x, y and z axes.

        Parameters
        ----------
        xtheta : float, int
            rotation arround the x axis [deg]
        ytheta : float, int
            rotation arround the y axis [deg]
        ztheta : float, int
            rotation arround the z axis [deg]
        '''
        or loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.rotate(xtheta, ytheta, ztheta)
            setattr(self, loadid, load)
       
    def toorigin(self):
        '''Revertns the reference point to the origin, updating the
        forces and moments of all loads.
        '''
        self.xshift(-self.D.xcoord)
        self.yshift(-self.D.ycoord)
        self.zshift(-self.D.zcoord)
        self.xrotate(-self.D.xtheta)
        self.yrotate(-self.D.ytheta)
        self.zrotate(-self.D.ztheta)
    
    def resetorigin(self):
        '''Sets the current reference point as the origin, seting the
        coordinates and angles to zero.
        '''
        for loadid in self.__dataclass_fields__.keys():
            load = getattr(self, loadid)
            load.xcoord = 0
            load.ycoord = 0
            load.zcoord = 0
            load.xtheta = 0
            load.ytheta = 0
            load.ztheta = 0
            setattr(self, loadid, load)

    def __add__(self, other):           
        if isinstance(other, Load):
            return self.__add_load__(other)
        elif isinstance(other, LoadGroup):
            return self.__add_loadgroup__(other)
        else:
            raise TypeError()
    
    def __add_load__(self, other):
        selfcopy = copy.deepcopy(self)
        setattr(selfcopy, other.loadtype, getattr(selfcopy, other.loadtype) + other)
        return selfcopy
    
    def __add_loadgroup__(self, other):
        selfcopy = copy.deepcopy(self)
        for loadid in selfcopy.__dataclass_fields__.keys():
            loadtype = getattr(selfcopy, loadid)
            otherload = getattr(other, loadid) 
            setattr(selfcopy, loadid, loadtype + otherload)
        return selfcopy
    
    def __mul__(self, multby):
        selfcopy = copy.deepcopy(self)
        for loadid in selfcopy.__dataclass_fields__.keys():
            setattr(selfcopy, loadid, getattr(selfcopy, loadid) * multby)
        return selfcopy
    
    def __rmul__(self, multby):
        self.__mult__(multby)
    
    def __truediv__(self, divby):
        selfcopy = copy.deepcopy(self)
        for loadid in selfcopy.__dataclass_fields__.keys():
            setattr(selfcopy, loadid, getattr(selfcopy, loadid) / divby)
        return selfcopy

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
            elif loadfactor>1:
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


@dataclass
class LoadCombination(dict):
    
    def __init__(self, LoadGroup, standard=DEFAULTSTANDARD, combiation='ultiamte'):
        self._set_standard(standard, combiation)

        for lc in self._loadcombinations.keys():
            self[lc] = _LoadBase()
            for loadid in self._loadcombinations[lc].__dataclass_fields__.keys():
                self[lc] = self[lc] + self._loadcombinations[lc].loadid * LoadGroup.loadid
    
    def _set_standard(self, standard, combiation):
        combiation = combiation.lower()
        if combiation not in ['combiation', 'service']
            mesage = ("Unsupported oad combination '{}', suported types are "
                      "'ultimate' and 'service'")
            mesage.format(combiation)
            raise ValueError(mesage)
        self._combiation = combiation
        standard = standard.lower()
        self._standard = standard
        if standard == 'asce':
            if self._combiation == 'ultimate':
                self._loadcombinations = asceultimate
            else: 
                self._loadcombinations = asceservice
        elif standard == 'cirsoc':
            if self._combiation == 'ultimate':
                self._loadcombinations = cirsocultimate
            else: 
                self._loadcombinations = cirsocservice
        else:
            raise StandardError(standard)
        
        

