import copy
import numbers

from cirsoc_402.constants import LOAD
from cirsoc_402.load.loadclass import Load
from cirsoc_402.load.loadclass import _LoadBase
from cirsoc_402.load.referenceframe import ReferenceFrame


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

    def __init__(self):
        self.D = Load('D')
        self.Di = Load('Di')
        self.E = Load('E')
        self.F = Load('F')
        self.Fa = Load('Fa')
        self.H = Load('H')
        self.L = Load('L')
        self.Lr = Load('Lr')
        self.R = Load('R')
        self.S = Load('S')
        self.T = Load('T')
        self.W = Load('W')
        self.Wi = Load('Wi')
    
    def __repr__(self):
        row = "{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n"

        txt = "Load\tFx\t\tFy\t\tFz\t\t\tMx\t\tMy\t\tMz\n"
        for loadid in LOAD:
            load = getattr(self, loadid)
            force = load.force
            moment = load.moment
            txt += row.format(loadid, force[0], force[1], force[2],
                              moment[0], moment[1], moment[2])
        txt += "\n"
        refframe = self.D.reference
        txt += "At reference frame:\n"
        txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.origin[0], refframe.origin[1], refframe.origin[2])
        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.xversor[0], refframe.xversor[1], refframe.xversor[2])
        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.yversor[0], refframe.yversor[1], refframe.yversor[2])
        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(refframe.zversor[0], refframe.zversor[1], refframe.zversor[2])
        return txt

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
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.shift(xshift, yshift, zshift)
            setattr(self, loadid, load)
        
    def xshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the x direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        self.shift(shift, 0, 0)
    
    def yshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the y direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the z direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        self.shift(0, 0, shift)
    
    def shift_ref(self, xshift, yshift, zshift):
        '''Updates moments of all loads for a movement of the reference
        point in the x, y and z directions.

        Parameters
        ----------
        xshift : float
            movement of the reference point in the x direction [m]
        yshift : float
            movement of the reference point in the y direction [m]
        zshift : float
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.shift_ref(xshift, yshift, zshift)
            setattr(self, loadid, load)
        
    def xshift_ref(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the x direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the x direction [m]
        '''
        self.shift_ref(shift, 0, 0)
    
    def yshift_ref(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the y direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the y direction [m]
        '''
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        '''Updates moments of all loads for a movement of the reference
        point in the z direction.

        Parameters
        ----------
        shift : float
            movement of the reference point in the z direction [m]
        '''
        self.shift_ref(0, 0, shift)
    
    def rotate_along(self, direction, theta):
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.rotate_along(direction, theta)
            setattr(self, loadid, load)

    def xrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the x axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the x axis [deg]
        '''
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the y axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the y axis [deg]
        '''
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        '''Updates forces and movements of all loads for a rotation of
        the reference axis arrond the z axis.

        Parameters
        ----------
        theta : float, int
            rotation arround the z axis [deg]
        '''
        self.rotate_along([0, 0, 1], theta)

    def rotate_along_ref(self, direction, theta):
        self.rotate_along(self.D.reference.r2o(direction), theta)

    def xrotate_ref(self, theta):
        self.rotate_along_ref([1, 0, 0], theta)
    
    def yrotate_ref(self, theta):
        self.rotate_along_ref([0, 1, 0], theta)
    
    def zrotate_ref(self, theta):
        self.rotate_along_ref([0, 0, 1], theta)
    
    def to_reference(self, other_reference):
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.to_reference(other_reference)
            setattr(self, loadid, load)

    def to_origin(self):
        self.to_reference(ReferenceFrame())

    def resetorigin(self):
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.reference = ReferenceFrame()
            setattr(self, loadid, load)

    def __add__(self, other):
        if isinstance(other, Load):
            return self.__add_load__(other)
        elif isinstance(other, LoadGroup):
            return self.__add_loadgroup__(other)
        else:
            raise TypeError('Only loads or loadgroups can be added to a loadgroup.')
    
    def __add_load__(self, other):
        selfcopy = copy.deepcopy(self)
        setattr(selfcopy, other.loadtype, getattr(selfcopy, other.loadtype) + other)
        return selfcopy
    
    def __add_loadgroup__(self, other):
        selfcopy = copy.deepcopy(self)
        for loadid in LOAD:
            load = getattr(selfcopy, loadid)
            otherload = getattr(other, loadid) 
            setattr(selfcopy, loadid, load + otherload)
        return selfcopy
    
    def __mul__(self, mulby):
        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')
        selfcopy = copy.deepcopy(self)
        for loadid in LOAD:
            setattr(selfcopy, loadid, getattr(selfcopy, loadid) * mulby)
        return selfcopy
    
    def __rmul__(self, mulby):
        return self.__mul__(mulby)
    
    def __truediv__(self, divby):
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a float or int.')
        if divby == 0:
            raise ValueError('Division by zero.')
        selfcopy = copy.deepcopy(self)
        for loadid in LOAD:
            setattr(selfcopy, loadid, getattr(selfcopy, loadid) / divby)
        return selfcopy

    def __iadd__(self, other):
        if isinstance(other, Load):
            self.__iadd_load__(other)
        elif isinstance(other, LoadGroup):
            self.__iadd_loadgroup__(other)
        else:
            raise TypeError('Only loads or loadgroups can be added to a loadgroup.')
    
    def __iadd_load__(self, other):
        setattr(self, other.loadtype, getattr(self, other.loadtype) + other)
  
    def __iadd_loadgroup__(self, other):
        for loadid in LOAD:
            load = getattr(self, loadid)
            otherload = getattr(other, loadid) 
            setattr(self, loadid, load + otherload)
    
    def __imul__(self, mulby):
        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')
        for loadid in LOAD:
            setattr(self, loadid, getattr(self, loadid) * mulby)

    def __irmul__(self, mulby):
        self.__imul__(mulby)
    
    def __itruediv__(self, divby):
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a float or int.')
        if divby == 0:
            raise ValueError('Division by zero.')
        for loadid in LOAD:
            setattr(self, loadid, getattr(self, loadid) / divby)