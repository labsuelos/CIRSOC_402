'''Module with the definition of the ReferenceFrame class.
'''

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numbers
import numpy as np

from cirsoc_402.load.quaternion import Quaternion

class Arrow3D(FancyArrowPatch):
    '''Class that extends matplotlib's Arrow patch to 3D plots. 

    Parameters
    ----------
    FancyArrowPatch : matplotlib.patches.FancyArrowPatch
        matplotlib.patches.FancyArrowPatch
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d,self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class ReferenceFrame:
    '''Class that define the frame of reference used for describing
    vectors, positions, forces and moments in 3D space. The
    ``ReferenceFrame`` object assumes that there is an absolute origin O
    relative to witch the location and orientation of the frame of
    reference is specified. The location of the ``ReferenceFrame`` is
    identifyed by the position vector in the absolute origin system of
    the point R with coordintes (0, 0, 0) in the reference system. The
    orientation of the reference system is indicated by the unit versors
    of the reference system axes (ex, ey and ez) expressed in the
    absolute origin system.

    Attributes
    ----------
        origin : np.ndarray
            coordinates of the reference system origin in the absolute
            origin system
        xversor : np.ndarray
            componets in the absolute origin system of the ex versor o
            the reference system
        yversor : np.ndarray
            componets in the absolute origin system of the ey versor o
            the reference system
        zversor : np.ndarray
            componets in the absolute origin system of the ez versor o
            the reference system
    '''
     
    def __init__(self, xcoord=0.0, ycoord=0.0, zcoord=0.0):
        if not isinstance(xcoord, numbers.Number) \
           or not isinstance(ycoord, numbers.Number) \
           or not isinstance(zcoord, numbers.Number):
            raise TypeError('Reference framce coordinates must be a numeric value.')
        self.origin = np.array([xcoord, ycoord, zcoord])
        self.xversor = np.array([1, 0, 0])
        self.yversor = np.array([0, 1, 0])
        self.zversor = np.array([0, 0, 1])
    
    def __repr__(self):
        txt = "R = ({:.2f}, {:.2f}, {:.2f})\n".format(self.origin[0], self.origin[1], self.origin[2])
        txt += "ex = ({:.2f}, {:.2f}, {:.2f})\n".format(self.xversor[0], self.xversor[1], self.xversor[2])
        txt += "ey = ({:.2f}, {:.2f}, {:.2f})\n".format(self.yversor[0], self.yversor[1], self.yversor[2])
        txt += "ez = ({:.2f}, {:.2f}, {:.2f})".format(self.zversor[0], self.zversor[1], self.zversor[2])
        return txt
    
    def shift(self, xshift, yshift, zshift):
        '''Moves the reference frame with the movement specified in the
        origin system coordinates.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the origin system
        yshift : float, int
            displacement along the y-axis of the origin system
        zshift : float, int
            displacement along the z-axis of the origin system
        '''
        self.origin = np.add(self.origin, [xshift, yshift, zshift],
                             casting="unsafe")

    def xshift(self, shift):
        '''Moves the reference frame in the x direction of the origin
        system. 

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the origin system
        '''
        self.shift(shift, 0, 0)
    
    def yshift(self, shift):
        '''Moves the reference frame in the y direction of the origin
        system. 

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the origin system
        '''
        self.shift(0, shift, 0)
    
    def zshift(self, shift):
        '''Moves the reference frame in the z direction of the origin
        system. 

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the origin system
        '''
        self.shift(0, 0, shift)
    
    def shift_ref(self, xshift, yshift, zshift):
        '''Moves the reference frame with the movement specified in the
        reference frame coordinates.

        Parameters
        ----------
        xshift : float, int
            displacement along the x-axis of the reference frame
        yshift : float, int
            displacement along the y-axis of the reference frame
        zshift : float, int
            displacement along the z-axis of the reference frame
        '''
        add = xshift * self.xversor + yshift * self.yversor \
              + zshift * self.zversor
        self.shift(add[0], add[1], add[2]) 

    def xshift_ref(self, shift):
        '''Moves the reference frame in the x direction of the reference
        frame. 

        Parameters
        ----------
        shift : float, int
            displacement along the x-axis of the reference frame
        '''
        self.shift_ref(shift, 0, 0)
    
    def yshift_ref(self, shift):
        '''Moves the reference frame in the y direction of the reference
        frame. 

        Parameters
        ----------
        shift : float, int
            displacement along the y-axis of the reference frame
        '''
        self.shift_ref(0, shift, 0)
    
    def zshift_ref(self, shift):
        '''Moves the reference frame in the z direction of the reference
        frame. 

        Parameters
        ----------
        shift : float, int
            displacement along the z-axis of the reference frame
        '''
        self.shift_ref(0, 0, shift)
    
    def rotate_along(self, direction, theta):
        '''Rotates a reference frame relative to its own origin along a
        direction specified in the absolute origin system.

        Parameters
        ----------
        direction : array-like
            direction vector relative to the origin system along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        direction = np.array(direction)
        direction = direction / np.dot(direction, direction) ** (1/2)
        xversor = Quaternion.rotate_vector_along(self.xversor[0], self.xversor[1], self.xversor[2],
                                                 direction[0], direction[1], direction[2], theta)
        yversor = Quaternion.rotate_vector_along(self.yversor[0], self.yversor[1], self.yversor[2],
                                                 direction[0], direction[1], direction[2], theta)
        zversor = Quaternion.rotate_vector_along(self.zversor[0], self.zversor[1], self.zversor[2],
                                                 direction[0], direction[1], direction[2], theta)
        self.xversor = [xversor._coord1, xversor._coord2, xversor._coord3]
        self.yversor = [yversor._coord1, yversor._coord2, yversor._coord3]
        self.zversor = [zversor._coord1, zversor._coord2, zversor._coord3]
        self._normalize_versors()
    
    def xrotate(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the x-direction of the absolute origin system.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the y-direction of the absolute origin system.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the z-direction of the absolute origin system.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along([0, 0, 1], theta)
    
    def rotate_along_ref(self, direction, theta):
        '''Rotates a reference frame relative to its own origin along a
        direction specified in the reference frame.

        Parameters
        ----------
        direction : array-like
            direction vector expressed in the refrence frame along which
            the reference frame will be rotated.
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.r2o(direction), theta)

    def xrotate_ref(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the x-direction of the reference frame.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.xversor, theta)
    
    def yrotate_ref(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the y-direction of the reference frame.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.yversor, theta)
    
    def zrotate_ref(self, theta):
        '''Rotates a reference frame relative to its own origin along
        the z-direction of the reference frame.

        Parameters
        ----------
        theta : float, int
            rotation [deg]
        '''
        self.rotate_along(self.zversor, theta)
    
    def move_to_origin(self):
        '''Translates and rotates the reference frame to match the 
        absolute origin frame.
        '''
        self.xshift(-self.origin[0])
        self.yshift(-self.origin[1])
        self.zshift(-self.origin[2])

        rotdir = np.cross([1, 0, 0], self.xversor)
        rotdir = rotdir / np.dot(rotdir, rotdir)**(1/2)
        if not all(np.round(rotdir, 6) == np.array([0, 0, 0])):
            theta = np.arccos(np.dot([1, 0, 0], self.xversor))
            self.rotate_along(rotdir, -np.rad2deg(theta))
        
        rotdir = np.cross([0, 1, 0], self.yversor)
        rotdir = rotdir / np.dot(rotdir, rotdir)**(1/2)
        if not all(np.round(rotdir, 6) == np.array([0, 0, 0])):
            theta = np.arccos(np.dot([0, 1, 0], self.yversor))
            self.rotate_along(rotdir, -np.rad2deg(theta))
            
        rotdir = np.cross([0, 0, 1], self.zversor)
        rotdir = rotdir / np.dot(rotdir, rotdir)**(1/2)
        if not all(np.round(rotdir, 6) == np.array([0, 0, 0])):
            theta = np.arccos(np.dot([0, 0, 1], self.zversor))
            self.rotate_along(rotdir, -np.rad2deg(theta))
    
    def o2r(self, vector):
        '''Coordinate transformation from the origin to the reference
        system.

        Parameters
        ----------
        vector : array-like
            3-element array like with the vector coordinates in the
            origin system

        Returns
        -------
        np.ndarray
            3-element array like with the vector coordinates in the
            reference system
        '''
        xcomp = np.dot(vector, [np.dot([1, 0, 0], self.xversor), np.dot([0, 1, 0], self.xversor), np.dot([0, 0, 1], self.xversor)])
        ycomp = np.dot(vector, [np.dot([1, 0, 0], self.yversor), np.dot([0, 1, 0], self.yversor), np.dot([0, 0, 1], self.yversor)])
        zcomp = np.dot(vector, [np.dot([1, 0, 0], self.zversor), np.dot([0, 1, 0], self.zversor), np.dot([0, 0, 1], self.zversor)])
        return np.array([xcomp, ycomp, zcomp])

    def r2o(self, vector):
        '''Coordinate transformation from the reference to the origin 
        system.

        Parameters
        ----------
        vector : array-like
            3-element array like with the vector coordinates in the
            reference system

        Returns
        -------
        np.ndarray
            3-element array like with the vector coordinates in the
            origin system
        '''
        xcomp = np.dot(vector, [np.dot([1, 0, 0], self.xversor), np.dot([1, 0, 0], self.yversor), np.dot([1, 0, 0], self.zversor)])
        ycomp = np.dot(vector, [np.dot([0, 1, 0], self.xversor), np.dot([0, 1, 0], self.yversor), np.dot([0, 1, 0], self.zversor)])
        zcomp = np.dot(vector, [np.dot([0, 0, 1], self.xversor), np.dot([0, 0, 1], self.yversor), np.dot([0, 0, 1], self.zversor)])
        return np.array([xcomp, ycomp, zcomp])

    def pos_o2r(self, vector):
        '''Finds the coordinates of a position vector in the reference
        system

        Parameters
        ----------
        vector : array-like
            3-element array like with the position vector coordinates in
            the origin system

        Returns
        -------
        np.ndarray
            3-element array like with the position vector coordinates in
            the reference system
        '''
        # position vector relative to the reference frame with its
        # coordinates expressed in the origin system
        vector = np.array(vector) - self.origin
        return self.o2r(vector)

    def pos_r2o(self, vector):
        '''Finds the coordinates of a position vector in the origin
        system

        Parameters
        ----------
        vector : array-like
            3-element array like with the position vector coordinates in
            the reference system

        Returns
        -------
        np.ndarray
            3-element array like with the position vector coordinates in
            the reference system
        '''
        # position of the origin relative to the reference frame with
        # its components expressed the reference frame 
        origin = self.o2r(-self.origin)
        # position vector relative to the origin frame with its
        # coordinates expressed in the reference system
        vector = vector - np.array(origin)
        return self.r2o(vector)

    def _normalize_versors(self):
        '''Makes sure that the direction versors of the reference frame
        have unit length.
        '''
        self.xversor =  self.xversor / np.dot(self.xversor, self.xversor)**(1/2)
        self.yversor =  self.yversor / np.dot(self.yversor, self.yversor)**(1/2)
        self.zversor =  self.zversor / np.dot(self.zversor, self.zversor)**(1/2)
        
    def plot(self, scale=1, margin=0.2, figsize=(8,8), elev=30, azimut=45):
        '''Creates a plot of the origin and reference systems. The plot
        convention is blue lines represents the x-axes, red lines for
        tye y-axes and black lines for z-axes. The origin system is
        shown with dotted lines and the reference frame is shown as
        full lines. The plot box is aligned with the origin system. 

        Parameters
        ----------
        scale : int, optional
            length of the line segments that ideintify each of the axes
            in the origin and reference systmes, by default 1
        margin : float, optional
            sapace between the plotted axes and the boundaries of the
            plot box expressed as a % (0-1) of the distance between the
            reference systems, by default 0.2
        figsize : tuple, optional
            figute size, by default (8,8)
        elev : int, optional
            viewpoint elevation, by default 30
        azimut : int, optional
            viewpoint angle, by default 45

        Returns
        -------
        matplotlib.pyplot.figure
            figure object with the plot
        

        Example
        -------

        Creates a reference frame coincident with the origin, rotates it
        45° relative to the x-axis and translates it relative to the
        origin x-axis. 

        >>> from cirsoc_402.load import ReferenceFrame
        >>> frame = ReferenceFrame()
        >>> frame.xrotate(45)
        >>> frame.xshift(3)
        >>> frame.plot() 

        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # plot origin directions
        ax.plot([0, scale], [0, 0], [0, 0], '--b', lw=2)
        ax.plot([0, 0], [0, scale], [0, 0], '--r', lw=2)
        ax.plot([0, 0], [0, 0], [0, scale], '--k', lw=2)
        # plot origin
        ax.plot([0],[0], [0], 'or', ms=9)
        
        ax.plot([self.origin[0]],[self.origin[1]], [self.origin[2]], 'ok', ms=9)
        
        ax.plot([self.origin[0], self.origin[0] + scale],
                [self.origin[1], self.origin[1]],
                [self.origin[2], self.origin[2]],
                '-k', alpha=0.5, lw=1)
        ax.plot([self.origin[0], self.origin[0] ],
                [self.origin[1], self.origin[1] + scale],
                [self.origin[2], self.origin[2]],
                '-k', alpha=0.5, lw=1)
        ax.plot([self.origin[0], self.origin[0]],
                [self.origin[1], self.origin[1]],
                [self.origin[2], self.origin[2] +  scale],
                '-k', alpha=0.5, lw=1)
        
        ax.plot([self.origin[0], self.origin[0] + self.xversor[0] * scale],
                [self.origin[1], self.origin[1] + self.xversor[1] * scale],
                [self.origin[2], self.origin[2] + self.xversor[2] * scale],
                '-b', lw=2)
        ax.plot([self.origin[0], self.origin[0] + self.yversor[0] * scale],
                [self.origin[1], self.origin[1] + self.yversor[1] * scale],
                [self.origin[2], self.origin[2] + self.yversor[2] * scale],
                '-r', lw=2)
        ax.plot([self.origin[0], self.origin[0] + self.zversor[0] * scale],
                [self.origin[1], self.origin[1] + self.zversor[1] * scale],
                [self.origin[2], self.origin[2] + self.zversor[2] * scale],
                '-k', lw=2)
        
        xcoords = [0, scale, self.origin[0], self.origin[0] + self.xversor[0] * scale,
                   self.origin[0] + self.yversor[0] * scale,
                   self.origin[0] + self.zversor[0] * scale]
        ycoords = [0, scale, self.origin[1], self.origin[1] + self.xversor[1] * scale,
                   self.origin[1] + self.yversor[1] * scale,
                   self.origin[1] + self.zversor[1] * scale]
        zcoords = [0, scale, self.origin[2], self.origin[2] + self.xversor[2] * scale,
                   self.origin[2] + self.yversor[2] * scale,
                   self.origin[2] + self.zversor[2] * scale]
        
        deltax = np.max(xcoords) - np.min(xcoords)
        deltay = np.max(ycoords) - np.min(ycoords)
        deltaz = np.max(zcoords) - np.min(zcoords)
        
        xlims =[np.min(xcoords) - deltax * margin, np.min(xcoords) + (1 + margin) * deltax]
        ylims = [np.min(ycoords) - deltay * margin, np.min(ycoords) + (1 + margin) * deltay]
        zlims = [np.min(zcoords) - deltaz * margin, np.min(zcoords) + (1 + margin) * deltaz]
        
        ax.plot([self.origin[1]], [self.origin[2]], 'ok', zs=xlims[0], zdir='x')
        ax.plot([self.origin[0]], [self.origin[2]], 'ok', zs=ylims[0], zdir='y')
        ax.plot([self.origin[0]], [self.origin[1]], 'ok', zs=zlims[0], zdir='z')
        
        ax.plot([0], [0], 'or', zs=xlims[0], zdir='x')
        ax.plot([0], [0], 'or', zs=ylims[0], zdir='y')
        ax.plot([0], [0], 'or', zs=zlims[0], zdir='z')
        
        ax.plot([0, 0], zlims, '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot([self.origin[1], self.origin[1]], zlims, '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot(ylims, [0, 0], '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot(ylims, [self.origin[2], self.origin[2]], '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        
        ax.plot([0, 0], zlims, '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot([self.origin[0], self.origin[0]], zlims, '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot(xlims, [0, 0], '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot(xlims, [self.origin[2], self.origin[2]], '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        
        ax.plot([0, 0], ylims, '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot(xlims, [0, 0],'-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot([self.origin[0], self.origin[0]], ylims, '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot(xlims, [self.origin[1], self.origin[1]], '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
  
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('$x_o$', fontsize=16)
        ax.set_ylabel('$y_o$', fontsize=16)
        ax.set_zlabel('$z_o$', fontsize=16)
        
        ax.grid(None)
        ax.set_xticks([0, self.origin[0]])
        ax.set_yticks([0, self.origin[1]])
        ax.set_zticks([0, self.origin[2]])
        ax.view_init(elev, azimut)

        deltax = xlims[1] - xlims[0]
        deltay = ylims[1] - ylims[0]
        deltaz = zlims[1] - zlims[0]

        delta = np.max([deltax, deltay, deltaz])
        ax.set_box_aspect(aspect = (deltax/delta, deltay/delta, deltaz/delta))
        plt.close(fig)
        return fig

    def plot_position(self, position, system, scale=1, margin=0.2,
                      figsize=(8,8), elev=30, azimut=45):
        '''Creates a plot of the origin and reference systems and the
        position vectors for both systems to a point in space. The plot
        convention is blue lines represents the x-axes, red lines for
        tye y-axes and black lines for z-axes. The origin system is
        shown with dotted lines and the reference frame is shown as
        full lines. The plot box is aligned with the origin system. 

        Parameters
        ----------
        position : array-like
            3-element array with the point coordiantes
        system : str
            system of reference in which the position is specified.
            Either 'origin' or 'reference'
        scale : int, optional
            length of the line segments that ideintify each of the axes
            in the origin and reference systmes, by default 1
        margin : float, optional
            sapace between the plotted axes and the boundaries of the
            plot box expressed as a % (0-1) of the distance between the
            reference systems, by default 0.2
        figsize : tuple, optional
            figute size, by default (8,8)
        elev : int, optional
            viewpoint elevation, by default 30
        azimut : int, optional
            viewpoint angle, by default 45

        Returns
        -------
        matplotlib.pyplot.figure
            figure object with the plot
        

        Example 1
        ---------

        Creates a reference frame coincident with the origin, rotates it
        45° relative to the x-axis and translates it relative to the
        origin x and z axes of the reference frame. Then plots the
        position vector with coordinates (1, 1, 1) in the origin system. 

        >>> from cirsoc_402 import ReferenceFrame
        >>> frame = ReferenceFrame()
        >>> frame.xrotate(45)
        >>> frame.xshift_rel(7)
        >>> frame.zshift_rel(-4)
        >>> frame.plot_position([1, 1, 1], 'origin') 

        Example 2
        ---------

        Creates a reference frame coincident with the origin, rotates it
        45° relative to the x-axis and translates it relative to the
        origin x and z axes of the reference frame. Then plots the
        position vector with coordinates (1, 1, 1) in the reference
        system. 

        >>> from cirsoc_402 import ReferenceFrame
        >>> frame = ReferenceFrame()
        >>> frame.xrotate(45)
        >>> frame.xshift_rel(7)
        >>> frame.zshift_rel(-4)
        >>> frame.plot_position([1, 1, 1], 'reference') 
        '''

        fig = self.plot(scale=scale, margin=margin, figsize=figsize, 
                        elev=elev, azimut=azimut)
        ax = fig.get_axes()[0]

        if not isinstance(system, str):
            raise TypeError("Reference system must be specified as 'origin' or 'reference'.")
        if system.lower() == 'origin':
            poso = position
            posr = self.pos_o2r(position)
        elif system.lower() == 'reference':
            posr = position
            poso = self.pos_r2o(position)
        else:
            raise ValueError("Reference system must be specified as 'origin' or 'reference'.")

        # position arrow from origin
        arrow = Arrow3D([0, poso[0]],
                        [0, poso[1]],
                        [0, poso[2]],
                        mutation_scale=20, 
                        lw=3, arrowstyle="-|>", color="g", zorder=2)
        ax.add_artist(arrow)

        # position arrow from reference system
        posr_in_o = self.r2o(posr)
        arrow = Arrow3D([self.origin[0], self.origin[0] + posr_in_o[0]],
                        [self.origin[1], self.origin[1] + posr_in_o[1]],
                        [self.origin[2], self.origin[2] + posr_in_o[2]],
                        mutation_scale=20, 
                        lw=3, arrowstyle="-|>", color="g", zorder=2)
        ax.add_artist(arrow)

        # remove gridlines
        for _ in range(12):
            ax.lines.pop(-1)

        xcoords = [0, scale, self.origin[0], self.origin[0] + self.xversor[0] * scale,
                   self.origin[0] + self.yversor[0] * scale,
                   self.origin[0] + self.zversor[0] * scale, poso[0]]
        ycoords = [0, scale, self.origin[1], self.origin[1] + self.xversor[1] * scale,
                   self.origin[1] + self.yversor[1] * scale,
                   self.origin[1] + self.zversor[1] * scale, poso[1]]
        zcoords = [0, scale, self.origin[2], self.origin[2] + self.xversor[2] * scale,
                   self.origin[2] + self.yversor[2] * scale,
                   self.origin[2] + self.zversor[2] * scale, poso[2]]
        
        deltax = np.max(xcoords) - np.min(xcoords)
        deltay = np.max(ycoords) - np.min(ycoords)
        deltaz = np.max(zcoords) - np.min(zcoords)
        
        xlims =[np.min(xcoords) - deltax * margin, np.min(xcoords) + (1 + margin) * deltax]
        ylims = [np.min(ycoords) - deltay * margin, np.min(ycoords) + (1 + margin) * deltay]
        zlims = [np.min(zcoords) - deltaz * margin, np.min(zcoords) + (1 + margin) * deltaz]
        
        ax.plot([poso[1]], [poso[2]], 'og', zs=xlims[0], zdir='x')
        ax.plot([poso[0]], [poso[2]], 'og', zs=ylims[0], zdir='y')
        ax.plot([poso[0]], [poso[1]], 'og', zs=zlims[0], zdir='z')
        
        
        ax.plot([0, 0], zlims, '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot([self.origin[1], self.origin[1]], zlims, '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot([poso[1], poso[1]], zlims, '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot(ylims, [0, 0], '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot(ylims, [self.origin[2], self.origin[2]], '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')
        ax.plot(ylims, [poso[2], poso[2]], '-k', lw=0.5, alpha=0.4, zs=xlims[0], zdir='x')

        ax.plot([0, 0], zlims, '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot([self.origin[0], self.origin[0]], zlims, '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot([poso[0], poso[0]], zlims, '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot(xlims, [0, 0], '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot(xlims, [self.origin[2], self.origin[2]], '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        ax.plot(xlims, [poso[2], poso[2]], '-k', lw=0.5, alpha=0.4, zs=ylims[0], zdir='y')
        
        ax.plot([0, 0], ylims, '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot([self.origin[0], self.origin[0]], ylims, '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot([poso[0], poso[0]], ylims, '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot(xlims, [0, 0],'-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot(xlims, [self.origin[1], self.origin[1]], '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')
        ax.plot(xlims, [poso[1], poso[1]], '-k', lw=0.5, alpha=0.4, zs=zlims[0], zdir='z')

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)

        ax.grid(None)
        ax.set_xticks([0, self.origin[0], poso[0]])
        ax.set_yticks([0, self.origin[1], poso[1]])
        ax.set_zticks([0, self.origin[2], poso[2]])

        deltax = xlims[1] - xlims[0]
        deltay = ylims[1] - ylims[0]
        deltaz = zlims[1] - zlims[0]
        delta = np.max([deltax, deltay, deltaz])
        ax.set_box_aspect(aspect = (deltax/delta, deltay/delta, deltaz/delta))

        plt.close(fig)
        return fig
