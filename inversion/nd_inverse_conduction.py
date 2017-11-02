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
from ..interpolation import RegularGridInterpolator
from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD

from ..mesh import MeshVariable
from .objective_variables import InvPrior, InvObservation
from .grad_ad import gradient_ad as ad_grad

class InversionND(object):

    def __init__(self, lithology, mesh):
        self.mesh = mesh
        self.lithology = np.array(lithology).ravel()

        # communicate lithology index
        lithology_index = np.unique(lithology)
        lith_min = np.array(lithology_index.min(), dtype=np.int32)
        lith_max = np.array(lithology_index.max(), dtype=np.int32)
        all_lith_min = np.array(0, dtype=np.int32)
        all_lith_max = np.array(0, dtype=np.int32)
        comm.Allreduce([lith_min, MPI.INT], [all_lith_min, MPI.INT], op=MPI.MIN)
        comm.Allreduce([lith_max, MPI.INT], [all_lith_max, MPI.INT], op=MPI.MAX)

        self.lithology_index = np.arange(all_lith_min, all_lith_max+1)
        self.lithology_mask = list(range(len(self.lithology_index)))

        for i, index in enumerate(self.lithology_index):
            self.lithology_mask[i] = np.nonzero(self.lithology == index)[0]


        # Custom linear / nearest-neighbour interpolator
        self.interp = RegularGridInterpolator(mesh.grid_coords[::-1],
                                              np.zeros(mesh.n),
                                              bounds_error=False, fill_value=np.nan)

        # Get weights for ghost points
        mesh.lvec.set(1.0)
        mesh.gvec.set(0.0)
        mesh.dm.localToGlobal(mesh.lvec, mesh.gvec, addv=True)
        mesh.dm.globalToLocal(mesh.gvec, mesh.lvec)
        self.ghost_weights = np.rint(mesh.lvec.array)


        # We assume uniform grid spacing for now
        delta = []
        for i in range(0, mesh.dim):
            dx = np.diff(mesh.grid_coords[i])
            delta.append(dx.mean())
        self.grid_delta = delta


        # Cost function variables
        self.observation = {}
        self.prior = {}


        # Initialise linear solver
        self.ksp = self._initialise_ksp()
        self.ksp_T = self._initialise_ksp() # <- need to pass transposed matrix

        # these should be depreciated soon
        self.temperature = self.mesh.gvec.duplicate()
        self._temperature = self.mesh.gvec.duplicate()


    def _initialise_ksp(self, matrix=None, solver='gmres', atol=1e-10, rtol=1e-50):
        """
        Initialise linear solver object
        """
        if matrix is None:
            matrix = self.mesh.mat

        ksp = PETSc.KSP().create(comm)
        ksp.setType(solver)
        ksp.setOperators(matrix)
        ksp.setTolerances(atol, rtol)
        ksp.setFromOptions()
        return ksp

    def get_boundary_conditions(self):
        """
        Retrieve the boundary conditions so they can be restored.
        This is only useful in the adjoint linear solve where we must assert
        Dirichlet BCs (I think)

        order is [minX, maxX, minY, maxY, minZ, maxZ]

        Returns
        -------
         bc_vals : values at the boundary conditions
         bc_flux : whether it is a Neumann boundary condition

        """
        dim = self.mesh.dim
        bc_vals = np.empty(dim*2)
        bc_flux = np.empty(dim*2, dtype=bool)

        wall = [("minX", "maxX"), ("minY", "maxY"), ("minZ", "maxZ")]

        for i in range(0, dim):
            w0, w1 = wall[i]
            i0, i1 = i*2, i*2+1

            bc_vals[i0] = self.mesh.bc[w0]["val"]
            bc_flux[i0] = self.mesh.bc[w0]["flux"]

            bc_vals[i1] = self.mesh.bc[w1]["val"]
            bc_flux[i1] = self.mesh.bc[w1]["flux"]

        return bc_vals, bc_flux


    def set_boundary_conditions(self, bc_vals, bc_flux):
        """
        Set the boundary conditions easily using two vectors
        order is [minX, maxX, minY, maxY, minZ, maxZ]

        Parameters
        -------
         bc_vals : values at the boundary conditions
         bc_flux : whether it is a Neumann boundary condition

        """
        dim = self.mesh.dim
        if len(bc_vals) != len(bc_flux) or len(bc_vals) != dim*2:
            raise ValueError("Input vectors should be of size {}".format(dim*2))

        wall = [("minX", "maxX"), ("minY", "maxY"), ("minZ", "maxZ")]

        for i in range(0, dim):
            w0, w1 = wall[i]
            i0, i1 = i*2, i*2+1

            self.mesh.bc[w0]["val"]  = bc_vals[i0]
            self.mesh.bc[w0]["flux"] = bc_flux[i0]

            self.mesh.bc[w1]["val"]  = bc_vals[i1]
            self.mesh.bc[w1]["flux"] = bc_flux[i1]


    def add_observation(self, **kwargs):
        """
        Add an observation to the Inversion routine.
        These will automatically be called when the objective function is called
        and will handle interpolation.

        """
        interp = self.interp
        interp.values = self.ghost_weights.reshape(self.mesh.n)

        for arg in kwargs:
            obs = kwargs[arg]
            if type(obs) is not InvObservation:
                raise TypeError("Need to pass {} instead of {}".format(InvObservation,type(obs)))

            # add interpolation information to obs
            w = interp(obs.coords)
            w = 1.0/np.floor(w + 1e-12)
            offproc = np.isnan(w)
            w[offproc] = 0.0 # these are weighted with zeros
            obs.w = w

            # store in dictionary
            self.observation[arg] = obs


    def add_prior(self, **kwargs):
        """
        Add a prior to the Inversion routine
        """

        for arg in kwargs:
            prior = kwargs[arg]
            if type(prior) is not InvPrior:
                raise TypeError("Need to pass {} instead of {}".format(InvPrior, type(prior)))

            prior.w = 1.0

            # store in dictionary
            self.prior[arg] = prior


    def objective_routine(self, **kwargs):
        """
        This always comes at the end of the forward model (beginning of the adjoint)
        so we can safely roll interpolation, cost into one method.

        Argument is a field if it is an observation - so that we can interpolate it.
        """

        # ensure an objective function is provided
        # if self.objective_function is None:
            # raise ValueError("Pass an objective function")
        c = 0.0

        for arg in kwargs:
            val = kwargs[arg]
            if arg in self.prior:
                prior = self.prior[arg]
                c += self.objective_function(val, prior.v, prior.dv)
            elif arg in self.observation:
                obs = self.observation[arg]

                # interpolation
                ival = self.interpolate(val, obs.coords)

                # weighting
                c += self.objective_function(ival*obs.w, obs.v*obs.w, obs.dv)

        return c

    def objective_routine_ad(self, **kwargs):

        dcdv = 0.0

        for arg in kwargs:
            val = kwargs[arg]
            if arg in self.prior:
                prior = self.prior[arg]
                dcdv = self.objective_function_ad(val, prior.v, prior.dv)
            elif arg in self.observation:
                obs = self.observation[arg]

                ival = self.interpolate(val, obs.coords)

                # weighting
                dcdinterp = self.objective_function_ad(ival*obs.w, obs.v*obs.w, obs.dv)

                # interpolation
                dcdv = self.interpolate_ad(dcdinterp, val, obs.coords)

                print arg, np.shape(val), np.shape(ival), np.shape(dcdv)
            else:
                dcdv = np.zeros_like(val)

        return dcdv





    def interpolate(self, field, xi, method="nearest"):
        self.interp.values = field.reshape(self.mesh.n)
        return self.interp(xi, method=method)

    def interpolate_ad(self, dxi, field, xi, method="nearest"):
        self.interp.values = field.reshape(self.mesh.n)
        return self.interp.adjoint(xi, dxi, method=method).ravel()



    def objective_function(self, x, x0, sigma_x0):
        return np.sum((x - x0)**2/sigma_x0**2)

    def objective_function_ad(self, x, x0, sigma_x0):
        return (2.0*x - 2.0*x0)/sigma_x0**2


    def map(self, *args):
        """
        Requires a tuple of vectors corresponding to an inversion variable
        these are mapped to the mesh.

        tuple(vec1, vec2, vecN) --> tuple(field1, field2, fieldN)
        """

        nf = len(args)
        nl = len(self.lithology_index)

        # preallocate memory
        mesh_variables = np.zeros((nf, self.lithology.size))

        # unpack vector to field
        for i in range(0, nl):
            idx = self.lithology_mask[i]
            for f in range(nf):
                mesh_variables[f,idx] = args[f][i]

        return list(mesh_variables)

    def map_ad(self, *args):
        """
        Map mesh variables back to the list
        """
        
        nf = len(args)
        nl = len(self.lithology_index)

        lith_variables = np.zeros((nf, self.lithology_index.size))

        for i in range(0, nl):
            idx = self.lithology_mask[i]
            for f in range(nf):
                lith_variables[f,i] += args[f][idx].sum()

        return list(lith_variables)


    def linear_solve(self, matrix=None, rhs=None):

        if matrix == None:
            matrix = self.mesh.construct_matrix()
        if rhs == None:
            rhs = self.mesh.construct_rhs()

        gvec = self.mesh.gvec
        lvec = self.mesh.lvec

        res = self.mesh.temperature

        self.ksp.setOperators(matrix)
        self.ksp.solve(rhs._gdata, res._gdata)
        return res[:].copy()

    def linear_solve_ad(self, T, dT, matrix=None, rhs=None):
        """
        If dT  = 0, adjoint=False : no need for this routine
        If dT != 0 and inside lithology, lith_size > 0
        """
        adjoint = np.array(False)
        lith_size = np.array(0.0)

        idxT = np.nonzero(dT != 0.0)[0]
        nT = idxT.any()
        comm.Allreduce([nT, MPI.BOOL], [adjoint, MPI.BOOL], op=MPI.LOR)
        if adjoint:
            if matrix == None:
                matrix = self.mesh.construct_matrix(in_place=False)
            if rhs == None:
                rhs = self.mesh.construct_rhs(in_place=False)
            rhs[:] = dT

            gvec = self.mesh.gvec
            lvec = self.mesh.lvec

            res = self.mesh.temperature
            res[:] = T

            # adjoint b vec
            db_ad = lvec.duplicate()

            matrix_T = self.mesh._initialise_matrix()
            matrix.transpose(matrix_T)
            self.ksp_T.setOperators(matrix_T)
            self.ksp_T.solve(rhs._gdata, gvec)
            self.mesh.dm.globalToLocal(gvec, db_ad)

            # adjoint A mat
            dk_ad = np.zeros_like(T)

            matrix.scale(-1.0)
            self.ksp.setOperators(matrix)
            # self.mesh.boundary_condition('maxZ', 0.0, flux=False) # not ND!!
            dT_ad = dT[:]
            kappa = np.zeros_like(T)
            
            nl = len(self.lithology_index)
            for i in range(0, nl):
                # find if there are nonzero dT that intersect a lithology
                idxM  = self.lithology_mask[i]
                idx_n = np.intersect1d(idxT, idxM)
                gnodes = self.ghost_weights[idx_n]
                local_size = np.array(float(idx_n.size)) - np.sum(1.0 - 1.0/gnodes) # ghost nodes
                comm.Allreduce([local_size, MPI.DOUBLE], [lith_size, MPI.DOUBLE], op=MPI.SUM)

                if lith_size > 0:
                    kappa.fill(0.0)
                    kappa[idxM] = 1.0
                    self.mesh.diffusivity[:] = kappa
                    dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
                    dAdklT = dAdkl * res._gdata
                    self.ksp.solve(dAdklT, gvec)
                    self.mesh.dm.globalToLocal(gvec, lvec)
                    if local_size > 0:
                        dk_ad[idx_n] += dT_ad.dot(lvec.array)/lith_size

            # return BCs to original
            # self.set_boundary_conditions(bc_vals, bc_flux)

            return dk_ad, db_ad.array
        else:
            return np.zeros_like(T), np.zeros_like(T)


    def gradient(self, f):
        """
        Calculate the derivatives of f in each dimension.

        Parameters
        ----------
         f  : ndarray shape(n,)

        Returns
        -------
         grad_f : ndarray shape(3,n)

        """
        grad = np.gradient(f.reshape(self.mesh.n), *self.grid_delta[::-1])
        return np.array(grad).reshape(self.mesh.dim, -1)

    def gradient_ad(self, df, f):
        inshape = [self.mesh.dim] + list(self.mesh.n)
        grad_ad = ad_grad(df.reshape(inshape), *self.grid_delta[::-1])
        return grad_ad.ravel()


    def heatflux(self, T, k):
        """
        Calculate heat flux.

        Arguments
        ---------
         T  : ndarray shape(n,) temperature
         k  : ndarray shape(n,) conductivity

        Returns
        -------
         q  : ndarray shape(3,n), heatflux vectors
        """
        return -k*self.gradient(T)

    def heatflux_ad(self, dq, q, T, k):

        dqddelT = -k
        dqdk = -self.gradient(T)

        ddelT = dqddelT*dq
        dk = dqdk*dq

        inshape = [self.mesh.dim] + list(self.mesh.n)
        print ddelT.shape, inshape

        dT = self.gradient_ad(ddelT, T)

        return dT.ravel(), dk.sum(axis=0)

        # dqn = dq
        # if np.shape(dq):
        #     dqn = dq.reshape(self.mesh.n)
        # gradT = self.gradient(T)
        # kn = -k.reshape(self.mesh.n)

        # # careful of the negative!
        # dqdk = -np.array(gradT).sum(axis=0)
        # dk = dqdk*dqn

        # dqdgradT = kn
        # dT = np.zeros_like(kn)
        # for i in range(0, self.mesh.dim):
        #     delta = np.mean(np.diff(self.mesh.grid_coords[::-1][i]))
        #     dqdT = kn/(self.mesh.n[i]*delta)
        #     dT += dqdT*dqn

        # return dT.ravel(), dk.ravel()

