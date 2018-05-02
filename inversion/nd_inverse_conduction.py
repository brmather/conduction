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
from ..interpolation import RegularGridInterpolator, KDTreeInterpolator
from ..mesh import MeshVariable
from ..tools import sum_duplicates
from .objective_variables import InvPrior, InvObservation
from .grad_ad import gradient_ad as ad_grad

from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD


class InversionND(object):

    def __init__(self, lithology, mesh, **kwargs):
        self.mesh = mesh
        lithology = np.array(lithology).ravel()

        # communicate lithology index
        lithology_index = np.unique(lithology)
        lith_min = np.array(lithology_index.min(), dtype=np.int32)
        lith_max = np.array(lithology_index.max(), dtype=np.int32)
        all_lith_min = np.array(0, dtype=np.int32)
        all_lith_max = np.array(0, dtype=np.int32)
        comm.Allreduce([lith_min, MPI.INT], [all_lith_min, MPI.INT], op=MPI.MIN)
        comm.Allreduce([lith_max, MPI.INT], [all_lith_max, MPI.INT], op=MPI.MAX)


        # update internal mask structures
        self.lithology_index = np.arange(all_lith_min, all_lith_max+1)
        self.update_lithology(lithology)


        # Custom linear / nearest-neighbour interpolator
        self.interp = RegularGridInterpolator(mesh.grid_coords[::-1],\
                                              np.zeros(mesh.n),\
                                              bounds_error=False, fill_value=np.nan)

        self.ndinterp = KDTreeInterpolator(mesh.coords, np.zeros(mesh.nn),\
                                           bounds_error=False, fill_value=0.0)

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
        self.ksp = self._initialise_ksp(**kwargs)
        self.ksp_ad  = self._initialise_ksp(**kwargs)
        self.ksp_adT = self._initialise_ksp(**kwargs) # <- need to pass transposed matrix

        # these should be depreciated soon
        self.temperature = self.mesh.gvec.duplicate()
        self._temperature = self.mesh.gvec.duplicate()
        self.iii = 0


    def update_lithology(self, new_lithology):
        """
        Update the configuration of lithologies

        Internal mask structures are updated to reflect the change in
        lithology configuration
        """

        new_lithology = np.array(new_lithology).ravel()

        nl = len(self.lithology_index)
        lithology_mask = [i for i in range(nl)]

        for i, index in enumerate(self.lithology_index):
            lithology_mask[i] = np.nonzero(new_lithology == index)[0]

        self.lithology_mask = lithology_mask
        self._lithology = new_lithology

        return

    @property
    def lithology(self):
        return self._lithology
    @lithology.setter
    def lithology(self, new_lithology):
        self.update_lithology(new_lithology)


    def _initialise_ksp(self, matrix=None, atol=1e-10, rtol=1e-50, **kwargs):
        """
        Initialise linear solver object
        """
        if matrix is None:
            matrix = self.mesh.mat

        solver = kwargs.pop('solver', 'gmres')
        precon = kwargs.pop('pc', None)

        ksp = PETSc.KSP().create(comm)
        ksp.setType(solver)
        ksp.setOperators(matrix)
        ksp.setTolerances(atol, rtol)
        if precon is not None:
            pc = ksp.getPC()
            pc.setType(precon)
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
            w = interp(obs.coords[:,::-1])
            w = 1.0/np.floor(w + 1e-12)
            offproc = np.isnan(w)
            w[offproc] = 0.0 # these are weighted with zeros
            obs.w = w # careful with 2x ghost nodes+

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

        c = np.array(0.0) # local prior values same as global
        c_obs = np.array(0.0) # obs have to be summed over all procs
        c_all = np.array(0.0) # sum of all obs

        for arg in kwargs:
            val = kwargs[arg]
            if arg in self.prior:
                prior = self.prior[arg]
                c += self.objective_function(val, prior.v, prior.dv)
            elif arg in self.observation:
                obs = self.observation[arg]

                # interpolation
                ival = self.interpolate(val, obs.coords)

                # weighting for ghost nodes
                c_obs += self.objective_function(ival*obs.w, obs.v*obs.w, obs.dv)


        comm.Allreduce([c_obs, MPI.DOUBLE], [c_all, MPI.DOUBLE], op=MPI.SUM)
        c += c_all

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
                dcdinterp = self.objective_function_ad(ival, obs.v, obs.dv)

                # interpolation
                dcdv = self.interpolate_ad(dcdinterp, val, obs.coords)
                # print arg, np.shape(val), np.shape(ival), np.shape(dcdv)

                # sync
                dcdv = self.mesh.sync(dcdv)
            else:
                dcdv = np.zeros_like(val)

        return dcdv


    def create_covariance_matrix(self, sigma_x0, width=1, fn=None, *args):
        """
        Create a covariance matrix assuming some function for variables on the mesh
        By default this is Gaussian.

        Arguments
        ---------
         sigma_x0 : uncertainty values to insert into matrix
         width    : width of stencil for matrix (int)
            i.e. extended number of neighbours for each node
         fn     : function to apply (default is Gaussian)
         *args  : input arguments to pass to fn

        Returns
        -------
            mat : covariance matrix
        """

        def gaussian_fn(sigma_x0, dist, *args):
            L = max(1e-12, dist.max() - dist.min()) # length scale
            return sigma_x0**2 * np.exp(-dist**2/(2*L**2))

        if type(fn) == type(None):
            fn = gaussian_fn

        nodes = self.mesh.nodes
        nn = self.mesh.nn
        n = self.mesh.n
        dim = self.mesh.dim

        coords = self.mesh.coords

        # setup new stencil
        stencil_width = 2*self.mesh.dim*width + 1
        rows = np.empty((stencil_width, self.mesh.nn), dtype=PETSc.IntType)
        cols = np.empty((stencil_width, self.mesh.nn), dtype=PETSc.IntType)
        vals = np.empty((stencil_width, self.mesh.nn))
        index = np.pad(nodes.reshape(n), width, 'constant', constant_values=-1)
        sigma = np.pad(sigma_x0.reshape(n), width, 'constant', constant_values=0)

        closure = []
        for w in range(width, 0, -1):
            closure_array = self.mesh._get_closure_array(dim, w, width)
            closure.extend(closure_array[:-1])
        closure.append(closure_array[-1]) # centre node at last

        # create closure object
        closure = self.mesh._create_closure_object(closure, width)


        for i in range(0, stencil_width):
            obj = closure[i]

            rows[i] = nodes
            cols[i] = index[obj].ravel()

            distance = np.linalg.norm(coords[cols[i]] - coords, axis=1)
            vals[i] = fn(sigma[obj].ravel(), distance, *args)

        vals[cols < 0] = 0.0
        vals[-1] = 0.0

        row = rows.ravel()
        col = cols.ravel()
        val = vals.ravel()

        # mask off-grid entries and sum duplicates
        mask = col >= 0
        row, col, val = sum_duplicates(row[mask], col[mask], val[mask])

        nnz = np.bincount(row)
        indptr = np.insert(np.cumsum(nnz),0,0)

        mat = self.mesh._initialise_matrix()
        mat.assemblyBegin()
        mat.setValuesLocalCSR(indptr.astype(PETSc.IntType), col, val)
        mat.assemblyEnd()

        # set diagonal vector
        lvec = self.mesh.lvec
        gvec = self.mesh.gvec
        lvec.setArray(sigma_x0**2)
        self.mesh.dm.localToGlobal(lvec, gvec)
        mat.setDiagonal(gvec)
        return mat



    def interpolate(self, field, xi, method="nearest"):
        self.ndinterp.values = field #.reshape(self.mesh.n)
        return self.ndinterp(xi, method=method)

    def interpolate_ad(self, dxi, field, xi, method="nearest"):
        self.ndinterp.values = field #.reshape(self.mesh.n)
        return self.ndinterp.adjoint(xi, dxi, method=method) #.ravel()



    def objective_function(self, x, x0, sigma_x0):
        return np.sum((x - x0)**2/sigma_x0**2)

    def objective_function_ad(self, x, x0, sigma_x0):
        return (2.0*x - 2.0*x0)/sigma_x0**2


    def objective_function_lstsq(self, x, x0, cov):
        """
        Nonlinear least squares objective function
        """
        ksp = PETSc.KSP().create(comm)
        ksp.setPC('lu')
        ksp.setOperators(cov)
        ksp.setFromOptions()

        misfit = np.array(x - x0)
        lhs, rhs = cov.createVecs()
        rhs.setArray(misfit)
        ksp.solve(rhs, lhs)
        sol = rhs*lhs
        sol.scale(0.5)

        ksp.destroy()
        lhs.destroy()
        rhs.destroy()

        return sol.sum()

    def objective_function_lstsq_ad(self, x, x0, cov):
        """
        Adjoint of the nonlinear least squares objective function
        """
        ksp = PETSc.KSP().create(comm)
        ksp.setPC('lu')
        ksp.setOperators(cov)
        ksp.setFromOptions()

        misfit = np.array(x - x0)
        lhs, rhs = cov.createVecs()
        rhs.set(1.0)
        ksp.solveTranspose(rhs, lhs)
        sol = rhs
        sol.setArray(misfit)
        sol *= lhs
        sol.scale(0.5)

        ksp.destroy()
        lhs.destroy()

        return sol.array


    def map(self, *args):
        """
        Requires a tuple of vectors corresponding to an inversion variable
        these are mapped to the mesh.

        tuple(vec1, vec2, vecN) --> tuple(field1, field2, fieldN)
        """

        nf = len(args)
        nl = len(self.lithology_index)

        # preallocate memory
        mesh_variables = np.zeros((nf, self.mesh.nn))

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
        all_lith_variables = np.zeros_like(lith_variables)

        for i in range(0, nl):
            idx = self.lithology_mask[i]
            for f in range(nf):
                lith_variables[f,i] += (args[f]/self.ghost_weights)[idx].sum()

        comm.Allreduce([lith_variables, MPI.DOUBLE], [all_lith_variables, MPI.DOUBLE], op=MPI.SUM)

        return list(all_lith_variables)


    def linear_solve(self, matrix=None, rhs=None):

        if matrix == None:
            matrix = self.mesh.construct_matrix()
        if rhs == None:
            rhs = self.mesh.construct_rhs()

        gvec = self.mesh.gvec
        lvec = self.mesh.lvec

        res = self.mesh.temperature
        # res._gdata.setArray(rhs._gdata)

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
            self.ksp_adT.setOperators(matrix_T)
            gvec.setArray(rhs._gdata)
            self.ksp_adT.solve(rhs._gdata, gvec)
            self.mesh.dm.globalToLocal(gvec, db_ad)

            # adjoint A mat
            dk_ad = np.zeros_like(T)

            matrix.scale(-1.0)
            self.ksp_ad.setOperators(matrix)
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
                # print comm.rank, i, lith_size, idx_n.size

                if lith_size > 0:
                    kappa.fill(0.0)
                    kappa[idxM] = 1.0
                    self.mesh.diffusivity[:] = kappa
                    dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
                    dAdklT = dAdkl * res._gdata
                    gvec.setArray(dAdklT) # try make the solution somewhat close
                    self.ksp_ad.solve(dAdklT, gvec)
                    self.mesh.dm.globalToLocal(gvec, lvec)

                    # need to call sum on the global vec
                    dk_local = (dT_ad*lvec.array)/lith_size
                    lvec.setArray(dk_local)
                    self.mesh.dm.localToGlobal(lvec, gvec)
                    gdot = gvec.sum()

                    if local_size > 0:
                        # splatter inside vector
                        dk_ad[idx_n] += gdot

            dk_ad = self.mesh.sync(dk_ad)


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

        dT = self.gradient_ad(ddelT, T)

        return dT.ravel(), dk.sum(axis=0)


    def add_perplex_table(self, TPtable):
        """
        Add Perplex table.

        Performs checks to make sure all lithologies exist inside
        the table.
        """
        from ..tools import PerplexTable
        if type(TPtable) is PerplexTable:
            for idx in self.lithology_index:
                if idx not in TPtable.table:
                    raise ValueError('{} not in TPtable'.format(idx))
            self.TPtable = TPtable
        else:
            raise ValueError('TPtable is incorrect type')

    def lookup_velocity(self, T=None, P=None):
        """
        Lookup velocity from VelocityTable object (vtable)

        Parameters
        ----------
         T  : temperature (optional)
           taken from active mesh variable if not given
         P  : pressure (optional)
           calculated from depth assuming a constant density of 2700 kg/m^3
        
        Returns
        -------
         table  : everything in the table for given nodes

        """
        if T is None:
            T = self.mesh.temperature[:]
        if P is None:
            z = np.absolute(self.mesh.coords[:,-1])

            rho = 2700.0
            r = 6.38e6 # radius of the Earth
            M = 5.98e24 # mass of the Earth
            G = 6.673e-11 # gravitational constant
            g = G*M/(r-z)**2
            P = rho*g*z*1e-5

        nl = len(self.lithology_index)
        nf = self.TPtable.ncol

        # preallocate memory
        V = np.zeros((nf, self.mesh.nn))

        for i in range(0, nl):
            idx = self.lithology_mask[i]
            lith_idx = self.lithology_index[i]
            V[:,idx] = self.TPtable(T[idx], P[idx], lith_idx).T

        return V
