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
        
        bckeys = self.bc.keys()
        
        ttime = 0.0
        for i in xrange(0, nsteps):
            # Enforce Dirichlet BCs
            self._update_bcs(T, False, *bckeys)
            
            # Evaluate derivatives
            d1x, d1y = self.gradient(T)
            flux_x = k*d1x
            flux_y = k*d1y
            
            # Enforce Neumann BCs
            flux_x = self._update_bcs(flux_x, True, 'minX', 'maxX')
            flux_y = self._update_bcs(flux_y, True, 'minY', 'maxY')
            
            # Second derivative
            flux_xx, flux_xy = self.gradient(flux_x)
            flux_yx, flux_yy = self.gradient(flux_y)
            
            d2 = flux_xx + flux_yy
            
            # Update timestep size
            dt = 0.5*maxdx**2*maxdy**2/k.max()*(maxdx**2 + maxdy**2)
            
            T += dt*(d2 + H)
            # sync
            ttime += dt
            
        return T, ttime