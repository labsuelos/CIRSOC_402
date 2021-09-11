
import copy
import numbers
import numpy as np

from cirsoc_402.constants import LOAD
from cirsoc_402.load.quaternion import Quaternion
from cirsoc_402.load.referenceframe import ReferenceFrame
from cirsoc_402.load.referenceframe import Arrow3D

class _LoadBase:
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
        force : np.ndarray
            components of the force in the reference frame
        moment : np.ndarray
            components of the moment in the reference frame
        reference : Referece
            reference frame in which the load and moment components are
            expressed
    
    Example 1
    ---------
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
        self.reference = reference
    
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
        
    def shift(self, xshift, yshift, zshift):
        # displacement of the reference frame expressed in the origin coordinate system 
        refdisp = np.array([xshift, yshift, zshift])
        # displacement of the reference frame expressed in the reference coordinate system
        refdisp = self.reference.o2r(refdisp)
        # new moment due to displacement expressed in the reference coordinate system 
        momentadd = np.cross(-refdisp, self.force)
        self.moment = self.moment + momentadd
        
    def xshift(self, shift):
        self.shift(shift, 0, 0)
        
    def yshift(self, shift):
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        self.shift(0, 0, shift)
        
    def shift_ref(self, xshift, yshift, zshift):
        # new moment due to displacement expressed in the reference coordinate system 
        momentadd = np.cross(-np.array([xshift, yshift, zshift]), self.force)
        self.moment = self.moment + momentadd
    
    def xshift_ref(self, shift):
        self.shift_ref(shift, 0, 0)
        
    def yshift_ref(self, shift):
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        self.shift_ref(0, 0, shift)
        
    def rotate_along(self, direction, theta):
        # components of the force and moment expressed in the origin coordinate system
        force_origin = self.reference.r2o(self.force)
        moment_origin = self.reference.r2o(self.moment)
        
        # rotation of the reference system
        self.reference.rotate_along(direction, theta)
        
        # components of the force and moment expressed in the rotated system
        self.force = self.reference.o2r(force_origin)
        self.moment = self.reference.o2r(moment_origin)
    
    def xrotate(self, theta):
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        self.rotate_along([0, 0, 1], theta)
        
    def rotate_along_ref(self, direction, theta):
        self.rotate_along(self.reference.r2o(direction), theta)
    
    def xrotate_ref(self, theta):
        self.rotate_along_ref([1, 0, 0], theta)
    
    def yrotate_ref(self, theta):
        self.rotate_along_ref([0, 1, 0], theta)
    
    def zrotate_ref(self, theta):
        self.rotate_along_ref([0, 0, 1], theta)
    
    def to_reference(self, other_reference):
        if not isinstance(other_reference, ReferenceFrame):
            raise TypeError('The frame of reference must be specified by a ReferenceFrame object.')
        
        # move to match position in space
        disp = other_reference.origin - self.reference.origin
        self.shift(disp[0], disp[1], disp[2])
        
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
        
    def to_origin(self):
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
        if not isinstance(other, (Load, _LoadBase)):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        selfcopy = copy.deepcopy(self)
        other.to_reference(self.reference)
        selfcopy.force += other.force
        selfcopy.moment += other.moment
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
        
        selfcopy = copy.deepcopy(self)
        other.to_reference(self.reference)
        selfcopy.force -= other.force
        selfcopy.moment -= other.moment
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
        selfcopy.moment = mulby * self.moment
        
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
        if not isinstance(other, (Load, _LoadBase)):
            raise TypeError("The forces to add must be specified in a Load objcet.")
        
        other.to_reference(self.reference)
        self.force += other.force
        self.moment += other.moment
        
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
        if not isinstance(other, (Load, _LoadBase)):
            raise TypeError("The forces to add must be specified in a Load objcet.")

        other.to_reference(self.reference)
        self.force -= other.force
        self.moment -= other.moment
    
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

        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')

        self.force = mulby * self.force
        self.moment = mulby * self.moment
    
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
        self.__imul__(mulby)

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
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a number.')
        
        if divby == 0:
            raise ValueError('Division by zero.')
        
        self.force = self.force / divby
        self.moment = self.moment / divby

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


class Load(_LoadBase):
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
        name : str
            name of the load
        force : np.ndarray
            components of the force in the reference frame
        moment : np.ndarray
            components of the moment in the reference frame
        reference : Referece
            reference frame in which the load and moment components are
            expressed
    
    Example 1
    ---------
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