"""
Copyright 2017 Ben Mather

This file is part of Conduction <https://git.dias.ie/itherc/conduction/>

Conduction is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Conduction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Conduction.  If not, see <http://www.gnu.org/licenses/>.
"""

try: range = xrange
except: pass

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD

from . import ConductionND

class DiffusionND(ConductionND):
    """
    Implicit N-dimensional solver for the time-dependent heat equation
    over a structured grid using PETSc data structures (inherits ConductionND).

    Parameters
    ----------
     theta  : float, parameter that controls temporal discretisation [0, 1]
        0.0 = backward Euler
        0.5 = Crank-Nicholson (default, most accurate)
        1.0 = forward Euler
    """
    def __init__(self, minCoord, maxCoord, res, theta=0.5, **kwargs):

        super(DiffusionND, self).__init__(minCoord, maxCoord, res, **kwargs)

        self.temperature_new = self.create_meshVariable("temperature_new")

        # theta rule in time, centred differences in space
        if theta < 0 or theta > 1:
            raise ValueError("theta must be in the range [0, 1]")
        self.theta = theta

        # get maximum spacing
        delta = 0.0
        for i in range(self.dim):
            dx = np.diff(self.grid_coords[i])
            delta = max(delta, dx.max())

        delta = np.array(delta)
        all_delta = np.array(0.0)
        comm.Allreduce([delta, MPI.DOUBLE], [all_delta, MPI.DOUBLE], op=MPI.MAX)
        self.delta = all_delta


    def calculate_dt(self):
        """
        Calculate optimal timestep size
        """
        kappa = self.diffusivity
        delta = self.delta
        return dt


    def construct_rhs(self):

        vec = -1.0*self.heat_sources[:]

        for wall in self.bc:
            val  = self.bc[wall]['val']
            flux = self.bc[wall]['flux']
            mask = self.bc[wall]['mask']
            if flux:
                vec[mask] += val
            else:
                vec[mask] = val

        rhs[:] = vec

        ## new bit


        return rhs


    def timestep(self, steps=1, dt=None):
        """
        Solve a timestep
        """
        if type(dt) == type(None):
            dt = self.calculate_dt()

        theta = self.theta
        Lscale = dt*theta
        Rscale = dt*(1.0 - theta)

        rhs = self.construct_rhs()
        mat = self.construct_matrix()
        rhs.scale(Rscale)
        mat.scale(Lscale)


        for step in range(steps):
            T = self.solve(mat, rhs)

        

        return T