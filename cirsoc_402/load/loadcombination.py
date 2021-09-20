
'''Module with the definition of the LoadCombination class.
'''

import copy
import numbers
import numpy as np

from cirsoc_402.constants import DEFAULTSTANDARD
from cirsoc_402.constants import LOAD
from cirsoc_402.exceptions import StandardError
from cirsoc_402.load.loadclass import GenericLoad
from cirsoc_402.load.loadclass import Load
from cirsoc_402.load.referenceframe import ReferenceFrame
from cirsoc_402.load.loadgroup import LoadGroup
from cirsoc_402.load.asce import ultimate as asceultimate
from cirsoc_402.load.asce import service as asceservice
from cirsoc_402.load.cirsoc import ultimate as cirsocultimate
from cirsoc_402.load.cirsoc import service as cirsocservice

class LoadCombination(dict):
    '''[summary]

    Attributes
    ----------
    dict : [type]
        [description]
    '''
    def __init__(self, standard=DEFAULTSTANDARD, combination='ultiamte'):
        self._set_standard(standard, combination)
        self._loadgroup = LoadGroup()
        
        for lc in self._loadcombinations.keys():
            self[lc] = GenericLoad()
    
    # makes sure that numpy methods don't override the class methods
    __array_priority__ = 10000

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
            txt += row.format(lc, self[lc].force[0], self[lc].force[1], self[lc].force[2], \
                              self[lc].moment[0], self[lc].moment[1], self[lc].moment[2])
        txt += "\n"
        refframe = self[lc].reference
        txt += "At reference frame:\n"
        txt += "R = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.origin[0], refframe.origin[1], refframe.origin[2])
        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.xversor[0], refframe.xversor[1], refframe.xversor[2])
        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(refframe.yversor[0], refframe.yversor[1], refframe.yversor[2])
        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(refframe.zversor[0], refframe.zversor[1], refframe.zversor[2])
        return txt
    
    def shift(self, xshift, yshift, zshift):
        '''Moves the load combination reference frame with the movement
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
        self._loadgroup.shift(xshift, yshift, zshift)
        for loadid in LOAD:
            self[loadid].shift(xshift, yshift, zshift)
    
    def xshift(self, shift):
        '''Moves the load combination reference frame in the x direction
        of the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the origin system
        '''
        self.shift(shift, 0, 0)
    
    def yshift(self, shift):
        '''Moves the load combination reference frame in the y direction
        of the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the origin system
        '''
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        '''Moves the load combination reference frame in the z direction
        of the origin system updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the origin system
        '''
        self.shift(0, 0, shift)
    
    def shift_ref(self, xshift, yshift, zshift):
        '''Moves the load combination reference frame with the movement
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
        self._loadgroup.shift_ref(xshift, yshift, zshift)
        for loadid in LOAD:
            self[loadid].shift_ref(xshift, yshift, zshift)

    def xshift_ref(self, shift):
        '''Moves the load combination reference frame in the x direction
        to the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the reference frame
        '''
        self.shift_ref(shift, 0, 0)
    
    def yshift_ref(self, shift):
        '''Moves the load combination reference frame in the y direction
        to the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the reference frame
        '''
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        '''Moves the load combination reference frame in the z direction
        to the reference frame updating the force and moment components.

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the reference frame
        '''
        self.shift_ref(0, 0, shift)

    def rotate_along(self, direction, theta):
        '''Rotates the load combination reference frame relative to its
        own origin along a direction specified in the absolute origin
        system updating the force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector relative to the origin system along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        self._loadgroup.rotate_along(direction, theta)
        for loadid in LOAD:
            self[loadid].rotate_along(direction, theta)
    
    def xrotate(self, theta):
        '''Rotates the load combiantion reference frame relative to its
        own origin along the x-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        '''Rotates the load combiantion reference frame relative to its
        own origin along the y-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        '''Rotates the load combiantion reference frame relative to its
        own origin along the z-direction of the absolute origin system
        updating the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 0, 1], theta)

    def rotate_along_ref(self, direction, theta):
        '''Rotates the load combination reference frame relative to its
        own origin along a direction specified in the reference frame
        updating the force and moment components.

        Parameters
        ----------
        direction : array-like
            direction vector expressed in the refrence frame along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        self._loadgroup.rotate_along_ref(direction, theta)
        for loadid in LOAD:
            self[loadid].rotate_along_ref(direction, theta)
    
    def xrotate_ref(self, theta):
        '''Rotates the load combination reference frame relative to its
        own origin along the x-direction of the reference frame updating
        the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([1, 0, 0], theta)
    
    def yrotate_ref(self, theta):
        '''Rotates the load combination reference frame relative to its
        own origin along the y-direction of the reference frame updating
        the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 1, 0], theta)
    
    def zrotate_ref(self, theta):
        '''Rotates the load combination reference frame relative to its
        own origin along the z-direction of the reference frame updating
        the force and moment components.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along_ref([0, 0, 1], theta)

    def to_reference(self, other_reference):
        '''Moves and rotates the load combination reference frame to
        match a target reference frame updating the force and moment
        components.

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

        self._loadgroup.to_reference(other_reference)
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.to_reference(other_reference)
            setattr(self, loadid, load)

    def toorigin(self):
        '''Moves and rotates the load combination reference frame to  
        match the absolut origin updating the force and moment
        components.
        '''
        self.to_reference(ReferenceFrame())
    
    def resetorigin(self):
        '''Sets the current reference point as the origin, seting the
        coordinates and angles to zero.
        '''
        self._loadgroup.resetorigin()
        for loadid in LOAD:
            load = getattr(self, loadid)
            load.resetorigin()
            setattr(self, loadid, load)
    
    def __add__(self, other):
        if isinstance(other, LoadGroup):
            return self.__add__load(other)
        elif isinstance(other, Load):
            return self.__add__loadgroup(other)
        raise TypeError('')

    def __add__load(self, other):
        selfcopy = copy.deepcopy(self)
        selfcopy._loadgroup += other
        loadid = other.loadtype
        for lc in self._loadcombinations.keys():
            selfcopy[lc] = selfcopy[lc] + getattr(selfcopy._loadcombinations[lc],loadid) * other
        return selfcopy

    def __add__loadgroup(self, other):
        selfcopy = copy.deepcopy(self)
        selfcopy._loadgroup += other
        for lc in self._loadcombinations.keys():
            for loadid in LOAD:
                selfcopy[lc] = selfcopy[lc] + getattr(selfcopy._loadcombinations[lc], loadid) * getattr(other, loadid)
        return selfcopy

    def __sub__(self, other):
        if isinstance(other, [LoadGroup, Load]):
            othercopy = copy.deepcopy(other)
            othercopy *= -1
            return self + othercopy
        raise TypeError('')
    
    def __mul__(self, mulby):
        if not isinstance(mulby, numbers.Number):
            raise TypeError('Forces can only be mumplitied by a number.')
        
        selfcopy = copy.deepcopy(self)
        selfcopy._loadgroup *= mulby
        for loadid in LOAD:
            selfcopy[loadid] *= mulby
        return selfcopy
    
    def __rmul__(self, mulby):
        return self.__mul__(mulby)
    
    def __truediv__(self, divby):
        if not isinstance(divby, numbers.Number):
            raise TypeError('Forces can only be divided by a number.')
        
        if divby == 0:
            raise ValueError('Division by zero.')
        
        selfcopy = copy.deepcopy(self)
        selfcopy._loadgroup /= divby
        for loadid in LOAD:
            selfcopy[loadid] /= divby
        return selfcopy
    
    def __iadd__(self, other):
        return self + other
    
    def __isub__(self, other):
        return self - other
    
    def __imul__(self, mulby):
        return self * mulby
    
    def __irmul__(self, mulby):
        return self * mulby
    
    def __itruediv__(self, divby):
        return self / divby