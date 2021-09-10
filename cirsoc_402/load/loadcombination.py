

from cirsoc_402.constants import DEFAULTSTANDARD
from cirsoc_402.exceptions import StandardError
from cirsoc_402.load.loadclass import _LoadBase
from cirsoc_402.load.asce import ultimate as asceultimate
from cirsoc_402.load.asce import service as asceservice
from cirsoc_402.load.cirsoc import ultimate as cirsocultimate
from cirsoc_402.load.cirsoc import service as cirsocservice

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


