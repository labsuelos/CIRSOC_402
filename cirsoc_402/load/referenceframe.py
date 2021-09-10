import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numbers
import numpy as np

from cirsoc_402.load.quaternion import Quaternion

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d,self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class ReferenceFrame:
     
    def __init__(self, xcoord=0, ycoord=0, zcoord=0):
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
    
    def xshift(self, shift):
        self.origin[0] += shift
    
    def yshift(self, shift):
        self.origin[1] += shift
    
    def zshift(self, shift):
        self.origin[2] += shift
        
    def xshift_ref(self, shift):
        shift = self.xversor * shift
        self.origin[0] += shift[0]
        self.origin[1] += shift[1]
        self.origin[2] += shift[2]
    
    def yshift_ref(self, shift):
        shift = self.yversor * shift
        self.origin[0] += shift[0]
        self.origin[1] += shift[1]
        self.origin[2] += shift[2]
    
    def zshift_ref(self, shift):
        shift = self.zversor * shift
        self.origin[0] += shift[0]
        self.origin[1] += shift[1]
        self.origin[2] += shift[2]
    
    def rotate_along(self, direction, theta):
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
        self.rotate_along([1, 0, 0], theta)
    
    def yrotate(self, theta):
        self.rotate_along([0, 1, 0], theta)
    
    def zrotate(self, theta):
        self.rotate_along([0, 0, 1], theta)
    
    def rotate_along_ref(self, direction, theta):
        self.rotate_along(self.r2o(direction), theta)

    def xrotate_ref(self, theta):
        self.rotate_along(self.xversor, theta)
    
    def yrotate_ref(self, theta):
        self.rotate_along(self.yversor, theta)
    
    def zrotate_ref(self, theta):
        self.rotate_along(self.zversor, theta)
    
    def move_to_origin(self):
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
        xcomp = np.dot(vector, [np.dot([1, 0, 0], self.xversor), np.dot([0, 1, 0], self.xversor), np.dot([0, 0, 1], self.xversor)])
        ycomp = np.dot(vector, [np.dot([1, 0, 0], self.yversor), np.dot([0, 1, 0], self.yversor), np.dot([0, 0, 1], self.yversor)])
        zcomp = np.dot(vector, [np.dot([1, 0, 0], self.zversor), np.dot([0, 1, 0], self.zversor), np.dot([0, 0, 1], self.zversor)])
        return np.array([xcomp, ycomp, zcomp])

    def r2o(self, vector):
        xcomp = np.dot(vector, [np.dot([1, 0, 0], self.xversor), np.dot([1, 0, 0], self.yversor), np.dot([1, 0, 0], self.zversor)])
        ycomp = np.dot(vector, [np.dot([0, 1, 0], self.xversor), np.dot([0, 1, 0], self.yversor), np.dot([0, 1, 0], self.zversor)])
        zcomp = np.dot(vector, [np.dot([0, 0, 1], self.xversor), np.dot([0, 0, 1], self.yversor), np.dot([0, 0, 1], self.zversor)])
        return np.array([xcomp, ycomp, zcomp])

    def pos_o2r(self, vector):
        # position vector relative to the reference frame with its
        # coordinates expressed in the origin system
        vector = np.array(vector) - self.origin
        return self.o2r(vector)

    def pos_r2o(self, vector):
        # position of the origin relative to the reference frame with
        # its components expressed the reference frame 
        origin = self.o2r(-self.origin)
        # position vector relative to the origin frame with its
        # coordinates expressed in the reference system
        vector = vector - np.array(origin)
        return self.r2o(vector)

    def _normalize_versors(self):
        self.xversor =  self.xversor / np.dot(self.xversor, self.xversor)**(1/2)
        self.yversor =  self.yversor / np.dot(self.yversor, self.yversor)**(1/2)
        self.zversor =  self.zversor / np.dot(self.zversor, self.zversor)**(1/2)
        
    def plot(self, scale=1, margin=0.2, figsize=(8,8), elev=30, azimut=45):
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
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        
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
