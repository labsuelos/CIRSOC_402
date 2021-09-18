'''Module with the definition of the Quaternion class used for rotating
forces [1]_.
'''

import numbers
import numpy as np


class Quaternion():
    '''Class that defines quaternion objects used for rotating vectors
    in 3D space. A quaternion is a generalization of the concept of
    complex numbers to 3D space. Quaternions can be expressed as a real
    component and 3 imaginary components assoicated to the unit vectors
    1i, 1j and 1k:

    q = q0 + q1 1i + q2 1j + q3 1k

    Where:
    1i**2 = 1j**2 = 1k**2 = −1
    1i * 1j = 1k, 1j * 1i = −1k
    1j * 1k = 1i, 1k * 1j = −1i
    1k * 1i = 1j, 1i * 1k = −1j 

    A vector x = (x1, x2, x3) can be rotated an angle theta along the
    direction (u1, u2, u3) by defining the quaternions:

    p = 0 + x1 1i + x2 1j + x3 1k

    r = cos(theta/2) + sin(theta/2) * (u1 1i + u2 1j + u3 1k)

    The rotated components p' are obtained as:

    p' = r * p * conj(r)

    Where the conjugate of a quaterion is defined by:

    conj(q) = q0 - q1 1i - q2 1j - q3 1k

    Compared to rotation matrices, quaternions are more compact,
    efficient, and numerically stable. Compared to Euler angles, they
    are simpler to compose and avoid the problem of gimbal lock.

    For a detailed explanation see [1]_.

    Attributes
    ----------
        _coord0 : float
            real component of the quaternion
        _coord1 : float
            imaginary component of the quaternion in the 1i axis
        _coord2 : float
            imaginary component of the quaternion in the 1j axis
        _coord3 : float
            imaginary component of the quaternion in the 1k axis
    '''

    def __init__(self, coord0, coord1, coord2, coord3):
        if not isinstance(coord0, numbers.Number) \
           or not isinstance(coord1, numbers.Number) \
           or not isinstance(coord2, numbers.Number) \
           or not isinstance(coord3, numbers.Number): 
            raise TypeError('Quaternion coordinates must be a numeric value.')
        self._coord0 = coord0
        self._coord1 = coord1
        self._coord2 = coord2
        self._coord3 = coord3
        
    def __repr__(self):
        return "({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(self._coord0, self._coord1, self._coord2, self._coord3)
    
    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        if self._coord0 != other._coord0:
            return False
        if self._coord1 != other._coord1:
            return False
        if self._coord2 != other._coord2:
            return False
        if self._coord3 != other._coord3:
            return False
        return True

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError('Only quaternions can be added to each other.')

        coord0 = self._coord0 + other._coord0
        coord1 = self._coord1 + other._coord1
        coord2 = self._coord2 + other._coord2
        coord3 = self._coord3 + other._coord3
        return Quaternion(coord0, coord1, coord2, coord3)

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError('Only quaternions can be substracted to each other.')

        coord0 = self._coord0 - other._coord0
        coord1 = self._coord1 - other._coord1
        coord2 = self._coord2 - other._coord2
        coord3 = self._coord3 - other._coord3
        return Quaternion(coord0, coord1, coord2, coord3)
    
    def __mul__(self, other):
        if not isinstance(other, (numbers.Number, Quaternion)):
            raise TypeError('Quaternions can be multiplied by numbers or other quaternions.')
        elif isinstance(other, (numbers.Number)):
            coord0 = self._coord0 * other
            coord1 = self._coord1 * other
            coord2 = self._coord2 * other
            coord3 = self._coord3 * other
            return Quaternion(coord0, coord1, coord2, coord3)
        elif isinstance(other, Quaternion):
            # (q0p0 − q1p1 − q2p2 − q3p3)
            coord0 = self._coord0 * other._coord0 - self._coord1 * other._coord1 \
                     - self._coord2 * other._coord2 - self._coord3 * other._coord3
            # (q0p1 + q1p0 + q2p3 − q3p2)
            coord1 = self._coord0 * other._coord1 + self._coord1 * other._coord0 \
                     + self._coord2 * other._coord3 - self._coord3 * other._coord2
            # (q0p2 + q2p0 − q1p3 + q3p1)
            coord2 = self._coord0 * other._coord2 + self._coord2 * other._coord0 \
                     - self._coord1 * other._coord3 + self._coord3 * other._coord1
            # (q0p3 + q3p0 + q1p2 − q2p1)
            coord3 = self._coord0 * other._coord3 + self._coord3 * other._coord0 \
                     + self._coord1 * other._coord2 - self._coord2 * other._coord1
            return Quaternion(coord0, coord1, coord2, coord3)
    
    def __rmul__(self, other):
        if not isinstance(other, (numbers.Number, Quaternion)):
            raise TypeError('Quaternions can be multiplied by numbers or other quaternions.')
        elif isinstance(other, (numbers.Number)):
            coord0 = self._coord0 * other
            coord1 = self._coord1 * other
            coord2 = self._coord2 * other
            coord3 = self._coord3 * other
            return Quaternion(coord0, coord1, coord2, coord3)
        elif isinstance(other, Quaternion):
            # (q0p0 − q1p1 − q2p2 − q3p3)
            coord0 = other._coord0 * self._coord0 - other._coord1 * self._coord1 \
                     - other._coord2 * self._coord2 - other._coord3 * self._coord3
            # (q0p1 + q1p0 + q2p3 − q3p2)
            coord1 = other._coord0 * self._coord1 + other._coord1 * self._coord0 \
                     + other._coord2 * self._coord3 - other._coord3 * self._coord2
            # (q0p2 + q2p0 − q1p3 + q3p1)
            coord2 = other._coord0 * self._coord2 + other._coord2 * self._coord0 \
                     - other._coord1 * self._coord3 + other._coord3 * self._coord1
            # (q0p3 + q3p0 + q1p2 − q2p1)
            coord3 = other._coord0 * self._coord3 + other._coord3 * self._coord0 \
                     + other._coord1 * self._coord2 - other._coord2 * self._coord1
            return Quaternion(coord0, coord1, coord2, coord3)

    def __truediv__(self, other):
        if not isinstance(other, (numbers.Number)):
            raise TypeError('Quaternions can be divided only by numbers.')
        coord0 = self._coord0 / other
        coord1 = self._coord1 / other
        coord2 = self._coord2 / other
        coord3 = self._coord3 / other
        return Quaternion(coord0, coord1, coord2, coord3)

    def __iadd__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError('Only quaternions can be added to each other.')

        self._coord0 += other._coord0
        self._coord1 += other._coord1
        self._coord2 += other._coord2
        self._coord3 += other._coord3
        return self

    def __isub__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError('Only quaternions can be substracted to each other.')

        self._coord0 -= other._coord0
        self._coord1 -= other._coord1
        self._coord2 -= other._coord2
        self._coord3 -= other._coord3
        return self

    def __imul__(self, other):
        if not isinstance(other, (numbers.Number, Quaternion)):
            raise TypeError('Quaternions can be multiplied by numbers or other quaternions.')
        elif isinstance(other, (numbers.Number)):
            self._coord0 *= other
            self._coord1 *= other
            self._coord2 *= other
            self._coord3 *= other
            return self
        elif isinstance(other, Quaternion):
            s_coord0 = self._coord0
            s_coord1 = self._coord1
            s_coord2 = self._coord2
            s_coord3 = self._coord3 
            # (q0p0 − q1p1 − q2p2 − q3p3)
            coord0 = s_coord0 * other._coord0 - s_coord1 * other._coord1 \
                     - s_coord2 * other._coord2 - s_coord3 * other._coord3
            # (q0p1 + q1p0 + q2p3 − q3p2)
            coord1 = s_coord0 * other._coord1 + s_coord1 * other._coord0 \
                     + s_coord2 * other._coord3 - s_coord3 * other._coord2
            # (q0p2 + q2p0 − q1p3 + q3p1)
            coord2 = s_coord0 * other._coord2 + s_coord2 * other._coord0 \
                     - s_coord1 * other._coord3 + s_coord3 * other._coord1
            # (q0p3 + q3p0 + q1p2 − q2p1)
            coord3 = s_coord0* other._coord3 + s_coord3 * other._coord0 \
                     + s_coord1 * other._coord2 - s_coord2 * other._coord1
            self._coord0 = coord0
            self._coord1 = coord1
            self._coord2 = coord2
            self._coord3 = coord3
            return self
    
    def __itruediv__(self, other):
        if not isinstance(other, (numbers.Number)):
            raise TypeError('Quaternions can be divided only by numbers.')
        self._coord0 /= other
        self._coord1 /= other
        self._coord2 /= other
        self._coord3 /= other
        return self

    def conj(self):
        '''Conjugate of the quaternion

        Returns
        -------
        Quaternion
            Conjugate of the quaternion
        '''
        return Quaternion(self._coord0, -self._coord1, -self._coord2, -self._coord3)
    
    def norm(self):
        '''Norm of the quaternion

        Returns
        -------
        float
            Norm of the quaternion
        '''
        return np.sqrt(self._coord0**2 + self._coord1**2 + self._coord2**2 + self._coord3**2)

    def inv(self):
        '''Inverse of the quaternion

        Returns
        -------
        Quaternion
            Inverse of the quaternion
        '''
        norm = self.norm()
        if norm == 0:
            return Quaternion(np.nan, np.nan, np.nan, np.nan)
        return self.conj() / norm**2
    
    def rotate(self, other):
        '''Rotates another quaternion based on the current quaternion.
        Being q the current quaternion and p the quaternion to be
        rotated, the function returns q * p * conj(q)

        Parameters
        ----------
        other : Quaternion
            Quaternion to be rotated

        Returns
        -------
        Quaternion
            Rotated coordinates of the quaternion.

        Raises
        ------
        TypeError
            The coordinates to be rotated must be specified as a
            quaternion.
        '''
        if not isinstance(other, Quaternion):
            raise TypeError('The coordinates to be rotated must be specified as a quaternion.')
        return self * other * self.conj()
    
    @staticmethod
    def xrotate_vector(xcoord, ycoord, zcoord, xtheta):
        '''Returns the components of a vector after it was rotated along
        the x-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before rotation
        ycoord : float, int
            y coordinate of the vector before rotation
        zcoord : float, int
            z coordinate of the vector before rotation
        xtheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the rotated coordinates of the vector
        '''
        # coordinates of the vector to be rotated expressed in a quaternion
        vecq = Quaternion(0, xcoord, ycoord, zcoord)
        # rotation quaternion as defined by the rotation axis and the rotation
        # angle
        rotq = Quaternion(np.cos(np.radians(xtheta / 2)), np.sin(np.radians(xtheta / 2)), 0, 0)
        # rotated vector
        return rotq.rotate(vecq)
    
    @staticmethod
    def yrotate_vector(xcoord, ycoord, zcoord, ytheta):
        '''Returns the components of a vector after it was rotated along
        the y-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before rotation
        ycoord : float, int
            y coordinate of the vector before rotation
        zcoord : float, int
            z coordinate of the vector before rotation
        ytheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the rotated coordinates of the vector
        '''
        # coordinates of the vector to be rotated expressed in a quaternion
        vecq = Quaternion(0, xcoord, ycoord, zcoord)
        # rotation quaternion as defined by the rotation axis and the rotation
        # angle
        rotq = Quaternion(np.cos(np.radians(ytheta / 2)), 0, np.sin(np.radians(ytheta / 2)), 0)
        # rotated vector
        return rotq.rotate(vecq)

    @staticmethod
    def zrotate_vector(xcoord, ycoord, zcoord, ztheta):
        '''Returns the components of a vector after it was rotated along
        the z-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before rotation
        ycoord : float, int
            y coordinate of the vector before rotation
        zcoord : float, int
            z coordinate of the vector before rotation
        ztheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the rotated coordinates of the vector
        '''
        # coordinates of the vector to be rotated expressed in a quaternion
        vecq = Quaternion(0, xcoord, ycoord, zcoord)
        # rotation quaternion as defined by the rotation axis and the rotation
        # angle
        rotq = Quaternion(np.cos(np.radians(ztheta / 2)), 0, 0, np.sin(np.radians(ztheta / 2)))
        # rotated vector
        return rotq.rotate(vecq)

    @staticmethod
    def rotate_vector_along(xcoord, ycoord, zcoord, xrot, yrot, zrot, theta):
        '''Rotates a vector along a specified axis

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector to be rotated
        ycoord : float, int
            y coordinate of the vector to be rotated
        zcoord : float, int
            z coordinate of the vector to be rotated
        xrot : float, int
            x coordinate of the rotation axis versor
        yrot : float, int
            y coordinate of the rotation axis versor
        zrot : float, int
            z coordinate of the rotation axis versor
        theta : float, int
            rotation angle [deg]

        Returns
        -------
        Quaternion
            Quaternion with the rotated components of the vector
        '''
        # coordinates of the vector to be rotated expressed in a quaternion
        vecq = Quaternion(0, xcoord, ycoord, zcoord)
        # rotation quaternion as defined by the rotation axis and the rotation
        # angle
        norm = np.sqrt(xrot**2 + yrot**2 + zrot**2)
        xrot = xrot / norm
        yrot = yrot / norm
        zrot = zrot / norm
        rotq = Quaternion(np.cos(np.radians(theta / 2)), xrot * np.sin(np.radians(theta / 2)),
                          yrot * np.sin(np.radians(theta / 2)), zrot * np.sin(np.radians(theta / 2)))
        return rotq.rotate(vecq)

    @staticmethod
    def xrotate_reference(xcoord, ycoord, zcoord, xtheta):
        '''Returns the components of a vector after the frame of
        reference is rotated along the x-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before reference rotation
        ycoord : float, int
            y coordinate of the vector before reference rotation
        zcoord : float, int
            z coordinate of the vector before reference rotation
        xtheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the coordinates of the vector after the
            rotation of the frame of reference
        '''
        return Quaternion.xrotate_vector(xcoord, ycoord, zcoord, -xtheta)
    
    @staticmethod
    def yrotate_reference(xcoord, ycoord, zcoord, ytheta):
        '''Returns the components of a vector after the frame of
        reference is rotated along the y-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before reference rotation
        ycoord : float, int
            y coordinate of the vector before reference rotation
        zcoord : float, int
            z coordinate of the vector before reference rotation
        ytheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the coordinates of the vector after the
            rotation of the frame of reference
        '''
        return Quaternion.yrotate_vector(xcoord, ycoord, zcoord, -ytheta)

    @staticmethod
    def zrotate_reference(xcoord, ycoord, zcoord, ztheta):
        '''Returns the components of a vector after the frame of
        reference is rotated along the z-axis.

        Parameters
        ----------
        xcoord : float, int
            x coordinate of the vector before reference rotation
        ycoord : float, int
            y coordinate of the vector before reference rotation
        zcoord : float, int
            z coordinate of the vector before reference rotation
        ztheta : float, int
            angle of rotation [deg]

        Returns
        -------
        Quaternion
            Quaternion with the coordinates of the vector after the
            rotation of the frame of reference
        '''
        return Quaternion.zrotate_vector(xcoord, ycoord, zcoord, -ztheta)