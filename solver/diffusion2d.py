import numpy as np
from . import Conduction2D

class Diffusion2D(Conduction2D):
    """
    Extends the Conduction2D object with explicit timestepping
    """

    def __init__(self, minCoords, maxCoords, res):
        
        super(Diffusion2D, self).__init__(minCoords, maxCoords, res)
        
        self.maxdx = np.diff(self.Xcoords).max()
        self.maxdy = np.diff(self.Ycoords).max()
        
    def refine(self, x_fn=None, y_fn=None):
        
        # Build on the existing method
        super(Diffusion2D, self).refine(x_fn, y_fn)
        
        self.maxdx = np.diff(self.Xcoords).max()
        self.maxdy = np.diff(self.Ycoords).max()
        
    def _update_bcs(self, vec, flux, *args):
        for wall in args:
            bc = self.bc[wall]
            if flux == bc['flux']:
                mask = bc['mask'].reshape(self.ny, self.nx)
                vec[mask] = bc['val']
        return vec
    
    def solve_timestep(self, nsteps=1):
        
        nx, ny = self.nx, self.ny
        T = self.temperature.reshape(ny, nx)
        H = self.heat_sources.reshape(ny, nx)
        k = self.diffusivity.reshape(ny, nx)

        maxdx = self.maxdx
        maxdy = self.maxdy

        dt = (0.5*maxdx**2*maxdy**2)/(k.max()*(maxdx**2 + maxdy**2))
        
        bckeys = self.bc.keys()
        
        ttime = 0.0
        for i in xrange(0, nsteps):
            # Enforce Dirichlet BCs
            self._update_bcs(T, False, *bckeys)

            D_c = k[1:-1,1:-1]
            D_e = 0.5*(k[1:-1,0:-2] + D_c)
            D_w = 0.5*(k[1:-1,2:]   + D_c)
            D_n = 0.5*(k[0:-2,1:-1] + D_c)
            D_s = 0.5*(k[2:,1:-1]   + D_c)

            U_c = T[1:-1,1:-1]
            U_e = T[1:-1,0:-2]
            U_w = T[1:-1,2:]  
            U_n = T[0:-2,1:-1]
            U_s = T[2:,1:-1]  

            h_x = (D_e*(U_e - U_c)/maxdx - D_w*(U_c - U_w)/maxdx)/maxdx
            h_y = (D_s*(U_s - U_c)/maxdy - D_n*(U_c - U_n)/maxdy)/maxdy

            T[1:-1,1:-1] += dt*(h_x + h_y + H[1:-1,1:-1])
            ttime += dt

        return T, ttime