
import copy
import numbers
import numpy as np

from cirsoc_402.constants import LOAD
from cirsoc_402.load.quaternion import Quaternion
from cirsoc_402.load.referenceframe import ReferenceFrame
from cirsoc_402.load.referenceframe import Arrow3D

class GenericLoad:
    '''Class that contains the components of the force and moment
    expressed in a given reference system, with the methods neccesary
    to update the force and moment components under changes in the
    reference frame.

    Attributes
    ----------
        force : np.ndarray
            components of the force in the reference frame
        moment : np.ndarray
            components of the moment in the reference frame
        reference : RefereceFrame
            reference frame in which the load and moment components are
            expressed
    '''
    
    def __init__(self, xforce=0.0, yforce=0.0, zforce=0.0,
                 xmoment=0.0, ymoment=0.0, zmoment=0.0,
                 reference=ReferenceFrame()):
        
        if not isinstance(xforce, numbers.Number) \
           or not isinstance(yforce, numbers.Number) \
           or not isinstance(zforce, numbers.Number):
            raise TypeError('Force components must be specified by a numeric value.')
        if not isinstance(xmoment, numbers.Number) \
           or not isinstance(ymoment, numbers.Number) \
           or not isinstance(zmoment, numbers.Number):
            raise TypeError('Moment components must be specified by a numeric value.')
        if not isinstance(reference, ReferenceFrame):
            raise TypeError('The frame of reference must be specified by a ReferenceFrame object.')
            
        self.force = np.array([xforce, yforce, zforce])
        self.moment = np.array([xmoment, ymoment, zmoment])
        self.reference = copy.deepcopy(reference)
    
    # makes sure that numpy methods don't override the class methods
    __array_priority__ = 10000

    def __repr__(self):
        refframe = self.reference
        txt = "F = ({:.2f}, {:.2f}, {:.2f})\n".format(self.force[0], self.force[1], self.force[2])
        txt += "M = ({:.2f}, {:.2f}, {:.2f})\n".format(self.moment[0], self.moment[1], self.moment[2])
        txt += "At reference frame:\n"
        txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.origin[0], refframe.origin[1], refframe.origin[2])
        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.xversor[0], refframe.xversor[1], refframe.xversor[2])
        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.yversor[0], refframe.yversor[1], refframe.yversor[2])
        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(refframe.zversor[0], refframe.zversor[1], refframe.zversor[2])
        return txt

    def __eq__(self, other):
        if  not isinstance(other, GenericLoad):
            return False
        if not all(self.force == other.force):
            return False
        if not all(self.moment == other.moment):
            return False
        if not self.reference == other.reference:
            return False
        return True

    @classmethod
    def fromarray(cls, force=[0,0,0], moment=[0,0,0], reference=ReferenceFrame()):
        '''Instanciates a GenericLoad object from array-like inputs

        Parameters
        ----------
        force : array-like, optional
            array with the forces componets, by default [0,0,0]
        moment : array-like, optional
            array with the moment components, by default [0,0,0]
        reference : RefereceFrame, optional
            reference frame of the load, by default ReferenceFrame()

        Returns
        -------
        GenericLoad
            Generic load object

        Raises
        ------
        TypeError
            Force components weren't specified as an array-like
        ValueError
            Wrong number of components in force array
        TypeError
            Moment components weren't specified as an array-like
        ValueError
            Wrong number of components in moment array
        '''
        if not isinstance(force, (list, tuple, np.ndarray)):
            raise TypeError('Force components must be specified by a 3-element array.')
        if len(force)!=3:
            raise ValueError('Force components must be specified by a 3-element array.')
        
        if not isinstance(moment, (list, tuple, np.ndarray)):
            raise TypeError('Moment components must be specified by a 3-element array.')
        if len(moment)!=3:
            raise ValueError('Moment components must be specified by a 3-element array.')

        return cls(xforce=force[0], yforce=force[1], zforce=force[2],
                   xmoment=moment[0], ymoment=moment[1], zmoment=moment[2],
                   reference=reference)

    def shift(self, xshift, yshift, zshift):
        '''Moves the load's reference frame with the movement specified in
        the origin system coordinates updating the force and moment
        components.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the origin system
        yshift : float, int
            displacement along the y-axis of the origin system
        zshift : float, int
            displacement along the z-axis of the origin system
        '''
        # displacement of the reference frame expressed in the origin coordinate system 
        refdisp = np.array([xshift, yshift, zshift])
        # displacement of the reference frame expressed in the reference coordinate system
        refdisp = self.reference.o2r(refdisp)
        # new moment due to displacement expressed in the reference coordinate system 
        momentadd = np.cross(-refdisp, self.force)
        self.moment = self.moment + momentadd
        # update reference frame
        self.reference.shift(xshift, yshift, zshift)
        
    def xshift(self, shift):
        '''Moves the load's reference frame in the x direction of the
        origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the origin system
        '''
        self.shift(shift, 0, 0)
        
    def yshift(self, shift):
        '''Moves the load's reference frame in the y direction of the
        origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the origin system
        '''
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        '''Moves the load's reference frame in the z direction of the
        origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the origin system
        '''
        self.shift(0, 0, shift)
        
    def shift_ref(self, xshift, yshift, zshift):
        '''Moves the load's reference frame with the movement specified in
        the reference frame coordinates updating the force and moment
        components.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the reference frame
        yshift : float, int
            displacement along the y-axis of the reference frame
        zshift : float, int
            displacement along the z-axis of the reference frame
        '''
        # new moment due to displacement expressed in the reference coordinate system 
        momentadd = np.cross(-np.array([xshift, yshift, zshift]), self.force)
        self.moment = self.moment + momentadd
        # update reference frame
        self.reference.shift_ref(xshift, yshift, zshift)
    
    def xshift_ref(self, shift):
        '''Moves the load's reference frame in the x direction of the
        reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the reference frame
        '''
        self.shift_ref(shift, 0, 0)
        
    def yshift_ref(self, shift):
        '''Moves the load's reference frame in the y direction of the
        reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the reference frame
        '''
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        '''Moves the load's reference frame in the z direction of the
        reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the reference frame
        '''
        self.shift_ref(0, 0, shift)
        
    def rotate_along(self, direction, theta):
        '''Rotates the load's reference frame relative to its own origin
        along a direction specified in the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector relative to the origin system along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        # components of the force and moment expressed in the origin coordinate system
        force_origin = self.reference.r2o(self.force)
        moment_origin = self.reference.r2o(self.moment)
        
        # rotation of the reference system
        self.reference.rotate_along(direction, theta)
        
        # components of the force and moment expressed in the rotated system
        self.force = self.reference.o2r(force_origin)
        self.moment = self.reference.o2r(moment_origin)
    
    def xrotate(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the x-direction of the absolute origin system updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the y-direction of the absolute origin system updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the z-direction of the absolute origin system updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 0, 1], theta)
        
    def rotate_along_ref(self, direction, theta):
        '''Rotates the load's reference frame relative to its own origin
        along a direction specified in the reference frame updating the
        force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector expressed in the refrence frame along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.reference.r2o(direction), theta)
    
    def xrotate_ref(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the x-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([1, 0, 0], theta)
    
    def yrotate_ref(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the y-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 1, 0], theta)
    
    def zrotate_ref(self, theta):
        '''Rotates the load's reference frame relative to its own origin
        along the z-direction of the reference frame updating the
        force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 0, 1], theta)
    
    def to_reference(self, other_reference):
        '''Moves and rotates the load's reference frame to match a
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
        
        # move to match position in space
        
        disp = other_reference.origin - self.reference.origin
        force_o = self.reference.r2o(self.force)
        moment_o = self.reference.r2o(self.moment)

        self.force = other_reference.o2r(force_o)
        self.moment = other_reference.o2r(moment_o - np.cross(disp, force_o))
        self.reference = copy.deepcopy(other_reference)
        #self.shift(disp[0], disp[1], disp[2])
        
        '''
        # rotate to match orientation of xversors
        unitvec = self.reference.r2o([1, 0, 0])
        otherunitvec = other_reference.r2o([1, 0, 0])
        #rotation direction
        rotdir = np.cross(unitvec, otherunitvec)
        rotdirnorm = np.dot(rotdir, rotdir)**(1/2)
        if rotdirnorm>0:
            rotdir = rotdir/rotdirnorm
            theta = np.arccos(np.dot(unitvec, otherunitvec) \
                             / np.dot(unitvec, unitvec)**(1/2) \
                             / np.dot(otherunitvec, otherunitvec)**(1/2))
            self.rotate_along(rotdir, -np.rad2deg(theta))
            
        # rotate to match orientation of yversors
        unitvec = self.reference.r2o([0, 1, 0])
        otherunitvec = other_reference.r2o([0, 1, 0])
        #rotation direction
        rotdir = np.cross(unitvec, otherunitvec)
        rotdirnorm = np.dot(rotdir, rotdir)**(1/2)
        if rotdirnorm>0:
            rotdir = rotdir/rotdirnorm
            theta = np.arccos(np.dot(unitvec, otherunitvec) \
                             / np.dot(unitvec, unitvec)**(1/2) \
                             / np.dot(otherunitvec, otherunitvec)**(1/2))
            self.rotate_along(rotdir, -np.rad2deg(theta))
        
        # rotate to match orientation of zversors
        unitvec = self.reference.r2o([0, 0, 1])
        otherunitvec = other_reference.r2o([0, 0, 1])
        #rotation direction
        rotdir = np.cross(unitvec, otherunitvec)
        rotdirnorm = np.dot(rotdir, rotdir)**(1/2)
        if rotdirnorm>0:
            rotdir = rotdir/rotdirnorm
            theta = np.arccos(np.dot(unitvec, otherunitvec) \
                             / np.dot(unitvec, unitvec)**(1/2) \
                             / np.dot(otherunitvec, otherunitvec)**(1/2))
            self.rotate_along(rotdir, -np.rad2deg(theta))
        '''
        
    def to_origin(self):
        '''Moves and rotates the load's reference frame to match the 
        absolut origin updating the force and moment components.
        '''
        self.to_reference(ReferenceFrame())
        
    def resetorigin(self):
        '''Sets the current reference point as the origin, seting the
        coordinates and angles to zero.
        '''
        self.reference = ReferenceFrame()
    
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
        if not isinstance(other, (Load, GenericLoad)):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        if any(np.isnan(self.force)) or any(np.isnan(self.moment)) or \
           any(np.isnan(other.force)) or any(np.isnan(other.moment)):
            selfcopy = copy.deepcopy(self)
            selfcopy.force = np.array([np.nan, np.nan, np.nan])
            selfcopy.moment = np.array([np.nan, np.nan, np.nan])
            return selfcopy
        selfcopy = copy.deepcopy(self)
        othercopy = copy.deepcopy(other)
        othercopy.to_reference(self.reference)
        selfcopy.force = selfcopy.force +  othercopy.force
        selfcopy.moment = selfcopy.moment + othercopy.moment
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
        if not isinstance(other, (Load, GenericLoad)):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        if any(np.isnan(self.force)) or any(np.isnan(self.moment)) or \
           any(np.isnan(other.force)) or any(np.isnan(other.moment)):
            selfcopy = copy.deepcopy(self)
            selfcopy.force = np.array([np.nan, np.nan, np.nan])
            selfcopy.moment = np.array([np.nan, np.nan, np.nan])
            return selfcopy
        selfcopy = copy.deepcopy(self)
        othercopy = copy.deepcopy(other)
        othercopy.to_reference(self.reference)
        selfcopy.force = selfcopy.force - othercopy.force
        selfcopy.moment = selfcopy.moment - othercopy.moment
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

        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')
            
        selfcopy = copy.deepcopy(self)
        selfcopy.force = mulby * selfcopy.force
        selfcopy.moment = mulby * selfcopy.moment
        
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
        '''Divide the forces and moments by a float or integer

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
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a number.')
        
        if divby == 0:
            raise ValueError('Division by zero.')
        
        selfcopy = copy.deepcopy(self)
        selfcopy.force = self.force / divby
        selfcopy.moment = self.moment / divby
        return(selfcopy)

    def __iadd__(self, other):
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
        return self + other
        
    def __isub__(self, other):
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
        return self - other
    
    def __imul__(self, mulby):
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
        return mulby * self
    
    def __irmul__(self, mulby):
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
        return mulby * self

    def __itruediv__(self, divby):
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
        return self / divby

    def plot(self, scale=1, margin=0.2, figsize=(8,8), elev=30, azimut=45):
        fig = self.reference.plot(scale=scale, margin=margin, figsize=figsize,
                                  elev=elev, azimut=azimut)
        ax = fig.get_axes()[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        xversor = np.array(self.reference.xversor)
        yversor = np.array(self.reference.yversor)
        zversor = np.array(self.reference.zversor)
        reforigin = [self.reference.origin[0], self.reference.origin[1], self.reference.origin[2]]
        
        force = xversor * self.force[0] + yversor * self.force[1] + zversor * self.force[2]
        forcenorm = np.dot(force, force)**(1/2)
        if forcenorm>0:
            force = force / forcenorm * scale
            arrow = Arrow3D([reforigin[0], reforigin[0] + force[0]],
                            [reforigin[1], reforigin[1] + force[1]],
                            [reforigin[2], reforigin[2] + force[2]],
                            mutation_scale=20, 
                            lw=3, arrowstyle="-|>", color="g", zorder=2)
            ax.add_artist(arrow)

        moment = xversor * self.moment[0] + yversor * self.moment[1] + zversor * self.moment[2]
        momentorm =  np.dot(moment, moment)**(1/2)
        if momentorm>0:
            moment = moment / momentorm * scale
            arrow = Arrow3D([reforigin[0], reforigin[0] + moment[0]],
                            [reforigin[1], reforigin[1] + moment[1]],
                            [reforigin[2], reforigin[2] + moment[2]],
                            mutation_scale=20, 
                            lw=3, arrowstyle="-|>", color="orange", zorder=2)
            ax.add_artist(arrow)
            arrow = Arrow3D([reforigin[0], reforigin[0] + 0.9 * moment[0]],
                            [reforigin[1], reforigin[1] + 0.9 * moment[1]],
                            [reforigin[2], reforigin[2] + 0.9 * moment[2]],
                            mutation_scale=20, 
                            lw=3, arrowstyle="-|>", color="orange", zorder=2)
            ax.add_artist(arrow) 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        return fig      


class Load(GenericLoad):
    '''Class that contains the components of the force and moment
    expressed in a given reference system, with the methods neccesary
    to update the force and moment components under changes in the
    reference frame. The class requires the specification of the
    load type.

    Attributes
    ----------
        loadtype : str
            load type accoding to cirsoc_402.constants.LOAD
        name : str
            name of the load
        force : np.ndarray
            components of the force in the reference frame
        moment : np.ndarray
            components of the moment in the reference frame
        reference : Referece
            reference frame in which the load and moment components are
            expressed
    '''

    def __init__(self, loadtype, name='', **kwargs):
        
        if not isinstance(loadtype, str):
            raise TypeError('Load type must be of type string.')
        if loadtype not in LOAD:
            mesage = "Unsupported load type {}. Supported types are: "
            mesage = mesage + ', '.join(LOAD)
            mesage = mesage.format(loadtype)
            raise ValueError(mesage)
        self.loadtype = loadtype
        if not isinstance(name, str):
            raise TypeError('Load name must be of type string.')
        self.name = name

        super(Load, self).__init__(**kwargs)


    def __repr__(self):
        refframe = self.reference

        txt = self.name + " (" + self.loadtype + ")\n"
        txt += "F = ({:.2f}, {:.2f}, {:.2f})\n".format(self.force[0], self.force[1], self.force[2])
        txt += "M = ({:.2f}, {:.2f}, {:.2f})\n".format(self.moment[0], self.moment[1], self.moment[2])
        txt += "At reference frame:\n"
        txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.origin[0], refframe.origin[1], refframe.origin[2])
        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.xversor[0], refframe.xversor[1], refframe.xversor[2])
        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.yversor[0], refframe.yversor[1], refframe.yversor[2])
        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(refframe.zversor[0], refframe.zversor[1], refframe.zversor[2])
        return txt
    
    def __eq__(self, other):
        if  not isinstance(other, Load):
            return False
        if self.name != other.name:
            return False
        if self.loadtype != other.loadtype:
            return False
        if not all(self.force == other.force):
            return False
        if not all(self.moment == other.moment):
            return False
        if not self.reference == other.reference:
            return False
        return True
    
    @classmethod
    def fromarray(cls, loadtype, name='', force=[0,0,0], moment=[0,0,0],
                  reference=ReferenceFrame()):
        '''Instanciates a Load object from array-like inputs

        Parameters
        ----------
        loadtype : str
            load type accoding to cirsoc_402.constants.LOAD
        name : str
            name of the load
        force : array-like, optional
            array with the forces componets, by default [0,0,0]
        moment : array-like, optional
            array with the moment components, by default [0,0,0]
        reference : RefereceFrame, optional
            reference frame of the load, by default ReferenceFrame()

        Returns
        -------
        GenericLoad
            Generic load object

        Raises
        ------
        TypeError
            Force components weren't specified as an array-like
        ValueError
            Wrong number of components in force array
        TypeError
            Moment components weren't specified as an array-like
        ValueError
            Wrong number of components in moment array
        '''
        if not isinstance(force, (list, tuple, np.ndarray)):
            raise TypeError('Force components must be specified by a 3-element array.')
        if len(force)!=3:
            raise ValueError('Force components must be specified by a 3-element array.')
        
        if not isinstance(moment, (list, tuple, np.ndarray)):
            raise TypeError('Moment components must be specified by a 3-element array.')
        if len(moment)!=3:
            raise ValueError('Moment components must be specified by a 3-element array.')

        return cls(loadtype, name=name,
                   xforce=force[0], yforce=force[1], zforce=force[2],
                   xmoment=moment[0], ymoment=moment[1], zmoment=moment[2],
                   reference=reference)
