
import copy
from dataclasses import dataclass, asdict, astuple
import numpy as np


from cirsoc_402.constants import LANGUAGE, LOAD, STANDARD, DEFAULTSTANDARD
from cirsoc_402.exceptions import StandardError
from cirsoc_402.load.quaternion import Quaternion
from cirsoc_402.load.asce import ultimate as asceultimate
from cirsoc_402.load.asce import service as asceservice

from cirsoc_402.load.cirsoc import ultimate as cirsocultimate
from cirsoc_402.load.cirsoc import service as cirsocservice


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
    if loadtype not in LOAD:
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
    
    def xshift(self, shift):
        '''Updates moments for a movement of the frane of reference in
        the x direction of the origin coordinate system.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        self.zmoment = self.zmoment - self.yforce * shift
        self.ymoment = self.ymoment + self.zforce * shift
        self.xcoord = self.xcoord + shift

    def yshift(self, shift):
        '''Updates moments for a movement of the frane of reference in
        the y direction of the origin coordinate system.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        self.zmoment = self.zmoment + self.xforce * shift
        self.xmoment = self.xmoment - self.zforce * shift
        self.ycoord = self.ycoord + shift
    
    def zshift(self, shift):
        '''Updates moments for a movement of the frane of reference in
        the z direction of the origin coordinate system.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        self.xmoment = self.xmoment + self.xforce * shift
        self.ymoment = self.ymoment - self.yforce * shift
        self.zcoord = self.zcoord + shift
    
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
        delta = self.excentricity()

        force = _rotation_matrix(theta).dot(np.array([self.yforce, self.zforce]))
        self.yforce = force[0]
        self.zforce = force[1]

        deltarot = _rotation_matrix(theta).dot(np.array([delta[1], delta[2]]))
        delta = np.array([delta[0], deltarot[0], deltarot[1]])

        moment = np.cross(delta, [self.xforce, self.yforce, self.zforce])

        self.xmoment = moment[0]
        self.ymoment = moment[1]
        self.ymoment = moment[2]
        self.ztheta = self.xtheta + theta

    def yrotate(self, theta):
        '''Updates forces and movements for a rotation of the reference
        axis arrond the y axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the y axis [deg]
        '''
        delta = self.excentricity()

        force = _rotation_matrix(theta).dot(np.array([self.zforce, self.xforce]))
        self.xforce = force[1]
        self.zforce = force[0]

        deltarot = _rotation_matrix(theta).dot(np.array([delta[2], delta[0]]))
        delta = np.array([deltarot[1], delta[1], deltarot[0]])

        moment = np.cross(delta, [self.xforce, self.yforce, self.zforce])

        self.xmoment = moment[0]
        self.ymoment = moment[1]
        self.ymoment = moment[2]
        self.ztheta = self.ytheta + theta
    
    def zrotate(self, theta):
        '''Updates forces and movements for a rotation of the reference
        axis arrond the z axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the z axis [deg]
        '''
        delta = self.excentricity()

        force = _rotation_matrix(theta).dot(np.array([self.xforce, self.yforce]))
        self.xforce = force[0]
        self.yforce = force[1]

        deltarot = _rotation_matrix(theta).dot(np.array([delta[0], delta[1]]))
        delta = np.array([deltarot[0], deltarot[1], delta[2]])

        moment = np.cross(delta, [self.xforce, self.yforce, self.zforce])

        self.xmoment = moment[0]
        self.ymoment = moment[1]
        self.ymoment = moment[2]
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

    def excentricity(self):
        if self.xforce==0 and self.yforce==0 and self.zforce==0:
            if self.xmoment!=0 or self.ymoment!=0 or self.zmoment!=0:
                raise RuntimeError("Excentricity cannot be computed with no actign force.")
        forcematrix = np.array([[0, self.zforce, -self.yforce],
                                [-self.zforce, 0, self.xforce],
                                [self.yforce, -self.xforce, 0]])
        delta = np.linalg.solve(forcematrix, [self.xmoment, self.ymoment, self.zmoment])
        return delta[0], delta[1], delta[2]

    def toorigin(self):
        '''Revertns the reference point to the origin, updating the
        forces and moments.
        '''
        self.xrotate(-self.xtheta)
        self.yrotate(-self.ytheta)
        self.zrotate(-self.ztheta)

        self.xshift(-self.xcoord)
        self.yshift(-self.ycoord)
        self.zshift(-self.zcoord)
        

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
        if not isinstance(other, (Load, _LoadBase)):
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
        if not isinstance(other, (Load, _LoadBase)):
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

    def __repr__(self):
        txt = self.name + " (" + self.loadtype + ")\n"
        txt += "F = ({:.2f}, {:.2f}, {:.2f}) \n".format(self.xforce, self.yforce, self.zforce)
        txt += "M = ({:.2f}, {:.2f}, {:.2f}) \n".format(self.xmoment, self.ymoment, self.zmoment)
        txt += "O = ({:.2f}, {:.2f}, {:.2f}) \n".format(self.xcoord, self.ycoord, self.zcoord)
        txt += "θ = ({:.2f}, {:.2f}, {:.2f})".format(self.xtheta, self.ytheta, self.ztheta)
        return txt
    
    def asdict(self):
        return asdict(self)
    
    def astuple(self):
        return astuple(self)
        


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
            load.shift(xshift, yshift, zshift)
            setattr(self, loadid, load)

    def xrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the x axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the x axis [deg]
        '''
        for loadid in self.__dataclass_fields__.keys():
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
        for loadid in self.__dataclass_fields__.keys():
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
        for loadid in self.__dataclass_fields__.keys():
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
        for loadid in self.__dataclass_fields__.keys():
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
        self.__mul__(multby)
    
    def __truediv__(self, divby):
        selfcopy = copy.deepcopy(self)
        for loadid in selfcopy.__dataclass_fields__.keys():
            setattr(selfcopy, loadid, getattr(selfcopy, loadid) / divby)
        return selfcopy
    
    def asdict(self):
        return asdict(self)
    
    def astuple(self):
        return astuple(self)

    def __repr__(self):
        row = "{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n"

        txt = "Load\tFx\t\tFy\t\tFz\t\t\tMx\t\tMy\t\tMz\n"
        txt += row.format('D', self.D.xforce, self.D.yforce, self.D.zforce, \
                          self.D.xmoment, self.D.ymoment, self.D.ymoment)
        txt += row.format('Di', self.Di.xforce, self.Di.yforce, self.Di.zforce, \
                          self.Di.xmoment, self.Di.ymoment, self.Di.ymoment)
        txt += row.format('E', self.E.xforce, self.E.yforce, self.E.zforce, \
                          self.E.xmoment, self.E.ymoment, self.E.ymoment)
        txt += row.format('F', self.F.xforce, self.F.yforce, self.F.zforce, \
                          self.F.xmoment, self.F.ymoment, self.F.ymoment)
        txt += row.format('Fa', self.Fa.xforce, self.Fa.yforce, self.Fa.zforce, \
                          self.Fa.xmoment, self.Fa.ymoment, self.Fa.ymoment)
        txt += row.format('H', self.H.xforce, self.H.yforce, self.H.zforce, \
                          self.H.xmoment, self.H.ymoment, self.H.ymoment)
        txt += row.format('L', self.L.xforce, self.L.yforce, self.L.zforce, \
                          self.L.xmoment, self.L.ymoment, self.L.ymoment)
        txt += row.format('Lr', self.Lr.xforce, self.Lr.yforce, self.Lr.zforce, \
                          self.Lr.xmoment, self.Lr.ymoment, self.Lr.ymoment)
        txt += row.format('R', self.R.xforce, self.R.yforce, self.R.zforce, \
                          self.R.xmoment, self.R.ymoment, self.R.ymoment)
        txt += row.format('S', self.S.xforce, self.S.yforce, self.S.zforce, \
                          self.S.xmoment, self.S.ymoment, self.S.ymoment)
        txt += row.format('T', self.T.xforce, self.T.yforce, self.T.zforce, \
                          self.T.xmoment, self.T.ymoment, self.T.ymoment)
        txt += row.format('W', self.W.xforce, self.W.yforce, self.W.zforce, \
                          self.W.xmoment, self.W.ymoment, self.W.ymoment)
        txt += row.format('Wi', self.Wi.xforce, self.Wi.yforce, self.Wi.zforce, \
                          self.Wi.xmoment, self.Wi.ymoment, self.Wi.ymoment)   
        txt += "\n"
        txt += "O = ({:.2f}, {:.2f}, {:.2f}) \n".format(self.D.xcoord, self.D.ycoord, self.D.zcoord)
        txt += "θ = ({:.2f}, {:.2f}, {:.2f})".format(self.D.xtheta, self.D.ytheta, self.D.ztheta)

        return txt




@dataclass
class LoadCombination(dict):
    
    def __init__(self, loadgroup, standard=DEFAULTSTANDARD, combination='ultiamte'):
        self._set_standard(standard, combination)
        self._loadgroup = loadgroup
        
        for lc in self._loadcombinations.keys():
            self[lc] = _LoadBase()
            for loadid in self._loadcombinations[lc].__dataclass_fields__.keys():
                self[lc] = self[lc] + getattr(self._loadcombinations[lc],loadid) * getattr(loadgroup, loadid)
    
    def _set_standard(self, standard, combination):
        combination = combination.lower()
        if combination not in ['ultiamte', 'service']:
            mesage = ("Unsupported load combination '{}', suported types are "
                      "'ultimate' and 'service'")
            mesage.format(combination)
            raise ValueError(mesage)
        self._combination = combination
        standard = standard.lower()
        self._standard = standard
        if standard == 'asce':
            if self._combination == 'ultimate':
                self._loadcombinations = asceultimate
            else: 
                self._loadcombinations = asceservice
        elif standard == 'cirsoc':
            if self._combination == 'ultimate':
                self._loadcombinations = cirsocultimate
            else: 
                self._loadcombinations = cirsocservice
        else:
            raise StandardError(standard)
    
    def __repr__(self):
        
        row = "{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n"
        txt = "{} {} load combination \n \n".format(self._standard, self._combination)
        txt += "LC\tFx\t\tFy\t\tFz\t\t\tMx\t\tMy\t\tMz\n"
        for lc in self.keys():
            txt += row.format(lc, self[lc].xforce, self[lc].yforce, self[lc].zforce, \
                              self[lc].xmoment, self[lc].ymoment, self[lc].zmoment)
        txt += "\n"
        txt += "O = ({:.2f}, {:.2f}, {:.2f}) \n".format(self[lc].xcoord, self[lc].ycoord, self[lc].zcoord)
        txt += "θ = ({:.2f}, {:.2f}, {:.2f})".format(self[lc].xtheta, self[lc].ytheta, self[lc].ztheta)
        return txt
    
    def asdict(self):
        outdict = {}
        for lc in self.keys():
            outdict[lc] = asdict(self[lc])
        return outdict
    
    def xshift(self, shift):
        '''Updates moments of the load group and the load combinations
        for a movement of the reference point in the x direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        self._loadgroup.xshift(shift)
        for lc in self.keys():
            self[lc].xshift(shift)
    
    def yshift(self, shift):
        '''Updates moments of the load group and the load combinations
        for a movement of the reference point in the y direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        self._loadgroup.yshift(shift)
        for lc in self.keys():
            self[lc].yshift(shift)
    
    def zshift(self, shift):
        '''Updates moments of the load group and the load combinations
        for a movement of the reference point in the z direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        self._loadgroup.zshift(shift)
        for lc in self.keys():
            self[lc].zshift(shift)
    
    def shift(self, xshift, yshift, zshift):
        '''Updates moments of the load group and the load combinations
        for a movement of the reference point in the x, y and z
        directions.

        Parameters
        ----------
        xshift : float
            movement of the reference point in the x direction [m]
        yshift : float
            movement of the reference point in the y direction [m]
        zshift : float
            movement of the reference point in the z direction [m]
        '''
        self._loadgroup.shift(xshift, yshift, zshift)
        for lc in self.keys():
            self[lc].shift(xshift, yshift, zshift)

    def xrotate(self, theta):
        '''Updates forces and moments of the load group and the load
        combinations for a rotation of the reference axis arrond the x
        axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the x axis [deg]
        '''
        self._loadgroup.xrotate(theta)
        for lc in self.keys():
            self[lc].xrotate(theta)
    
    def yrotate(self, theta):
        '''Updates forces and moments of the load group and the load
        combinations for a rotation of the reference axis arrond the y
        axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the y axis [deg]
        '''
        self._loadgroup.yrotate(theta)
        for lc in self.keys():
            self[lc].yrotate(theta)
    
    def zrotate(self, theta):
        '''Updates forces and moments of the load group and the load
        combinations for a rotation of the reference axis arrond the z
        axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the z axis [deg]
        '''
        self._loadgroup.zrotate(theta)
        for lc in self.keys():
            self[lc].zrotate(theta)

    def rotate(self, xtheta, ytheta, ztheta):
        '''Updates forces and moments of the load group and the load
        combinations for a rotation of the reference axis arrond the x,
        y and z axes.

        Parameters
        ----------
        xtheta : float, int
            rotation arround the x axis [deg]
        ytheta : float, int
            rotation arround the y axis [deg]
        ztheta : float, int
            rotation arround the z axis [deg]
        '''
        self._loadgroup.rotation(xtheta, ytheta, ztheta)
        for lc in self.keys():
            self[lc].rotation(xtheta, ytheta, ztheta)
       
    def toorigin(self):
        '''Revertns the reference point of the load group and load
        combinations to the origin, updating the forces and moments of
        all loads.
        '''
        self._loadgroup.toorigin()
        for lc in self.keys():
            self[lc].toorigin()
    
    def resetorigin(self):
        '''Sets the current reference point as the origin of the load
        group and load combinations, seting the coordinates and angles
        to zero.
        '''
        self._loadgroup.resetorigin()
        for lc in self.keys():
            self[lc].resetorigin()



