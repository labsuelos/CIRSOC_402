import copy
import numbers

from cirsoc_402.constants import LOAD
from cirsoc_402.load.loadclass import Load
from cirsoc_402.load.loadclass import GenericLoad
from cirsoc_402.load.referenceframe import ReferenceFrame


class LoadGroup():
    '''Set of loads (D, L, Lr, ...) acting simultaneosly on a same
    place.

    Attributes
    -------
    D : GenericLoad
        dead load
    Di : GenericLoad
        weight of ice
    E : GenericLoad
        eqarthquake load
    F : GenericLoad
        load due to fluids with well-defined pressures and maximum
        heights 
    Fa : GenericLoad
        flood load
    H : GenericLoad
        load due to lateral earth pressure, ground water pressure, or
        pressure of bulk materials
    L : GenericLoad
        live load
    Lr : GenericLoad
        roof live load
    R : GenericLoad
        rain load
    S : GenericLoad
        snow load
    T : GenericLoad
        self-tensing load
    W : GenericLoad
        wind load
    Wi : GenericLoad
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
    
    # makes sure that numpy methods don't override the class methods
    __array_priority__ = 10000

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
        '''Moves the load group reference frame with the movement
        specified in the origin system coordinates updating the force
        and moment components.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the origin system
        yshift : float, int
            displacement along the y-axis of the origin system
        zshift : float, int
            displacement along the z-axis of the origin system
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.shift(xshift, yshift, zshift)
            setattr(self, loadid, load)
        
    def xshift(self, shift):
        '''Moves the load group reference frame in the x direction of
        the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the origin system
        '''
        self.shift(shift, 0, 0)
    
    def yshift(self, shift):
        '''Moves the load group reference frame in the y direction of
        the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the origin system
        '''
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        '''Moves the load group reference frame in the z direction of
        the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the origin system
        '''
        self.shift(0, 0, shift)
    
    def shift_ref(self, xshift, yshift, zshift):
        '''Moves the load group reference frame with the movement
        specified in the reference frame coordinates updating the force
        and moment components.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the reference frame
        yshift : float, int
            displacement along the y-axis of the reference frame
        zshift : float, int
            displacement along the z-axis of the reference frame
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.shift_ref(xshift, yshift, zshift)
            setattr(self, loadid, load)
        
    def xshift_ref(self, shift):
        '''Moves the load group reference frame in the x direction to
        the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the reference frame
        '''
        self.shift_ref(shift, 0, 0)
    
    def yshift_ref(self, shift):
        '''Moves the load group reference frame in the y direction to
        the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the reference frame
        '''
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        '''Moves the load group reference frame in the z direction to
        the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the reference frame
        '''
        self.shift_ref(0, 0, shift)
    
    def rotate_along(self, direction, theta):
        '''Rotates the load group reference frame relative to its own
        origin along a direction specified in the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector relative to the origin system along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.rotate_along(direction, theta)
            setattr(self, loadid, load)

    def xrotate(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the x-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the y-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the z-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 0, 1], theta)

    def rotate_along_ref(self, direction, theta):
        '''Rotates the load group reference frame relative to its own
        origin along a direction specified in the reference frame
        updating the force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector expressed in the refrence frame along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.D.reference.r2o(direction), theta)

    def xrotate_ref(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the x-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([1, 0, 0], theta)
    
    def yrotate_ref(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the y-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 1, 0], theta)
    
    def zrotate_ref(self, theta):
        '''Rotates the load group reference frame relative to its own
        origin along the z-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 0, 1], theta)
    
    def to_reference(self, other_reference):
        '''Moves and rotates the load group reference frame to match a
        target reference frame updating the force and moment components.

        Parameters
        ----------
        other_reference : ReferenceFrame
            target frame of reference

        Raises
        ------
        TypeError
            The target frame of refernce wasn't specified with a
            ReferenceFrame object
        '''
        if not isinstance(other_reference, ReferenceFrame):
            raise TypeError('The frame of reference must be specified by a ReferenceFrame object.')

        for loadid in LOAD:
            load = getattr(self, loadid)
            load.to_reference(other_reference)
            setattr(self, loadid, load)

    def to_origin(self):
        '''Moves and rotates the load group reference frame to match the 
        absolut origin updating the force and moment components.
        '''
        self.to_reference(ReferenceFrame())

    def resetorigin(self):
        '''Sets the current reference frame as te origin, keeping the
        current values of the force and moment components.
        '''
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.reference = ReferenceFrame()
            setattr(self, loadid, load)

    def __add__(self, other):
        if isinstance(other, Load):
            return self.__add_load__(other)
        elif isinstance(other, LoadGroup):
            return self.__add_loadgroup__(other)
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
    
    def __sub__(self, other):
        if isinstance(other, Load):
            return self.__sub_load__(other)
        elif isinstance(other, LoadGroup):
            return self.__sub_loadgroup__(other)
        raise TypeError('Only loads or loadgroups can be added to a loadgroup.')
    
    def __sub_load__(self, other):
        selfcopy = copy.deepcopy(self)
        setattr(selfcopy, other.loadtype, getattr(selfcopy, other.loadtype) - other)
        return selfcopy
    
    def __sub_loadgroup__(self, other):
        selfcopy = copy.deepcopy(self)
        for loadid in LOAD:
            load = getattr(selfcopy, loadid)
            otherload = getattr(other, loadid) 
            setattr(selfcopy, loadid, load - otherload)
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
        if isinstance(other, Load) or isinstance(other, LoadGroup):
            return self + other
        raise TypeError('Only loads or loadgroups can be added to a loadgroup.')

    def __isub__(self, other):
        if isinstance(other, Load) or isinstance(other, LoadGroup):
            return self - other
        raise TypeError('Only loads or loadgroups can be added to a loadgroup.')

    def __imul__(self, mulby):
        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')
        for loadid in LOAD:
            setattr(self, loadid, getattr(self, loadid) * mulby)
        return self

    def __irmul__(self, mulby):
        self.__imul__(mulby)
    
    def __itruediv__(self, divby):
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a float or int.')
        if divby == 0:
            raise ValueError('Division by zero.')
        for loadid in LOAD:
            setattr(self, loadid, getattr(self, loadid) / divby)
        return self