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

    def __init__(self, lithology, mesh, lithology_index=None, **kwargs):
        self.mesh = mesh

        # update internal mask structures
        self.update_lithology(lithology, lithology_index)


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


        # objective function dictionaries
        self.observation = {}
        self.prior = {}


        # Initialise linear solver
        self.ksp = self._initialise_ksp(**kwargs)

        # these should be depreciated soon
        self.temperature = self.mesh.gvec.duplicate()
        self._temperature = self.mesh.gvec.duplicate()
        self.iii = 0


    def update_lithology(self, new_lithology, lithology_index=None):
        """
        Update the configuration of lithologies.

        Internal mask structures are updated to reflect the change in
        lithology configuration.

        Arguments
        ---------
         new_lithology   : field on the mesh with integers indicating
            : the position of particular lithologies
         lithology_index : array corresponding to the total number of
            : integers in new_lithology
        
        Notes
        -----
         lithology_index is determined from the min/max of elements
         in new_lithology if lithology_index=None
        """

        new_lithology = np.array(new_lithology).ravel()

        # sync across processors
        new_lithology = self.mesh.sync(new_lithology)
        new_lithology = new_lithology.astype(np.int)

        if type(lithology_index) == type(None):
            # query global vector for minx/max
            iloc, lith_min = self.mesh.gvec.min()
            iloc, lith_max = self.mesh.gvec.max()

            # create lithology index
            lithology_index = np.arange(int(lith_min), int(lith_max)+1)


        nl = len(lithology_index)
        lithology_mask = [i for i in range(nl)]

        # create lithology mask
        for i, index in enumerate(lithology_index):
            lithology_mask[i] = np.nonzero(new_lithology == index)[0]

        self.lithology_mask = lithology_mask
        self.lithology_index = lithology_index
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

                if prior.cov is None:
                    c += self.objective_function(val, prior.v, prior.dv)
                else:
                    c += self.objective_function_lstsq(val, prior.v, prior.cov)
            elif arg in self.observation:
                obs = self.observation[arg]

                # interpolation
                ival = self.interpolate(val, obs.coords)

                # weighting for ghost nodes
                if obs.cov is None:
                    c_obs += self.objective_function(ival*obs.w, obs.v*obs.w, obs.dv)
                else:
                    c_obs += self.objective_function_lstsq(ival*obs.w, obs.v*obs.w, obs.cov)



        comm.Allreduce([c_obs, MPI.DOUBLE], [c_all, MPI.DOUBLE], op=MPI.SUM)
        c += c_all
        return c

    def objective_routine_ad(self, **kwargs):

        dcdv = 0.0

        for arg in kwargs:
            val = kwargs[arg]
            if arg in self.prior:
                prior = self.prior[arg]
                if prior.cov is None:
                    dcdv = self.objective_function_ad(val, prior.v, prior.dv)
                else:
                    dcdv = self.objective_function_lstsq_ad(val, prior.v, prior.cov)

            elif arg in self.observation:
                obs = self.observation[arg]

                ival = self.interpolate(val, obs.coords)


                if obs.cov is None:
                    dcdinterp = self.objective_function_ad(ival, obs.v, obs.dv)
                else:
                    dcdinterp = self.objective_function_lstsq_ad(ival*obs.w, obs.v*obs.w, obs.cov)

                # interpolation
                dcdv = self.interpolate_ad(dcdinterp, val, obs.coords)
                # print arg, np.shape(val), np.shape(ival), np.shape(dcdv)

                # sync
                dcdv = self.mesh.sync(dcdv)
            else:
                dcdv = np.zeros_like(val)

        return dcdv


    def create_covariance_matrix(self, sigma_x0, width=1, indexing='xy', fn=None, *args):
        """
        Create a covariance matrix assuming some function for variables on the mesh
        By default this is Gaussian.

        Arguments
        ---------
         sigma_x0 : uncertainty values to insert into matrix
         width    : width of stencil for matrix (int)
            i.e. extended number of neighbours for each node
         indexing : use the xy coordinates of the mesh nodes or indices
            set to 'xy' or 'ij'
         fn       : function to apply (default is Gaussian)
         *args    : input arguments to pass to fn

        Returns
        -------
            mat : covariance matrix
        """

        def gaussian_fn(sigma_x0, dist, length_scale):
            return sigma_x0**2 * np.exp(-dist**2/(2*length_scale**2))

        if type(fn) == type(None):
            fn = gaussian_fn

        nodes = self.mesh.nodes
        nn = self.mesh.nn
        n = self.mesh.n
        dim = self.mesh.dim

        if indexing == "xy":
            coords = self.mesh.coords
        elif indexing == "ij":
            ic = []
            for i in range(dim):
                ic.append( np.arange(n[i]) )
            cij = np.meshgrid(*ic, indexing="ij")

            for i in range(dim):
                cij[i] = cij[i].ravel()
            coords = np.column_stack(cij)

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

        nnz = (stencil_width, dim*2)
        mat = self.mesh._initialise_matrix(nnz=nnz)
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


    def create_covariance_matrix_kdtree(self, sigma_x0, width=1, indexing='xy', fn=None, *args):
        """
        Create a covariance matrix assuming some function for variables on the mesh.
        By default this is Gaussian.

        This uses a KDTree to determine distance between nodes, rather than the
        matrix stencil indexing used by create_covariance_matrix
        

        Arguments
        ---------
         sigma_x0 : uncertainty values to insert into matrix
         width    : width of stencil for matrix (int)
            i.e. extended number of neighbours for each node
         indexing : use the xy coordinates of the mesh nodes or indices
            set to 'xy' or 'ij'
         fn       : function to apply (default is Gaussian)
         *args    : input arguments to pass to fn

        Returns
        -------
            mat : covariance matrix
        """

        def gaussian_fn(sigma_x0, dist, length_scale):
            return sigma_x0**2 * np.exp(-dist**2/(2*length_scale**2))

        if type(fn) == type(None):
            fn = gaussian_fn

        nodes = self.mesh.nodes
        nn = self.mesh.nn
        n = self.mesh.n
        dim = self.mesh.dim

        if indexing == "xy":
            coords = self.mesh.coords
            tree = self.ndinterp.tree
        elif indexing == "ij":
            from scipy.spatial import cKDTree

            ic = []
            for i in range(dim):
                ic.append( np.arange(n[i]) )
            cij = np.meshgrid(*ic, indexing="ij")

            for i in range(dim):
                cij[i] = cij[i].ravel()
            coords = np.column_stack(cij)
            tree = cKDTree(coords)


        # find distance between coords and centroid
        dist = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
        nnz = int(1.5*(dist <= max_dist).sum())

        mat = self.mesh._initialise_matrix(nnz=(nnz,1))
        mat.assemblyBegin()

        for i in range(0, nn):
            idx = tree.query_ball_point(coords[i], max_dist)
            dist = np.linalg.norm(coords[i] - coords[idx], axis=1)
            
            row = i
            col = idx
            val = fn(sigma[idx], dist, *args, **kwargs)
            
            mat.setValues(row, col, val)

        mat.assemblyEnd()
        return mat


    def interpolate(self, field, xi, method="nearest"):
        self.ndinterp.values = field #.reshape(self.mesh.n)
        return self.ndinterp(xi, method=method)

    def interpolate_ad(self, dxi, field, xi, method="nearest"):
        self.ndinterp.values = field #.reshape(self.mesh.n)
        return self.ndinterp.adjoint(xi, dxi, method=method) #.ravel()

    def in_bounds(self, xi):
        """
        Find if coordinates are inside the local processor bounds
        """
        idx, d, bounds = self.ndinterp._find_indices(xi)
        return bounds


    def objective_function(self, x, x0, sigma_x0):
        return np.sum(0.5*(x - x0)**2/sigma_x0**2)

    def objective_function_ad(self, x, x0, sigma_x0):
        return 0.5*(2.0*x - 2.0*x0)/sigma_x0**2


    def objective_function_lstsq(self, x, x0, cov):
        """
        Nonlinear least squares objective function
        """
        ksp = self._initialise_ksp(cov, pc='lu')
        misfit = np.array(x - x0)
        lhs, rhs = cov.createVecs()
        rhs.set(0.0)
        lindices = np.arange(0, misfit.size, dtype=PETSc.IntType)
        rhs.setValues(lindices, misfit, PETSc.InsertMode.ADD_VALUES)
        rhs.assemble()
        ksp.solve(rhs, lhs)
        sol = rhs*lhs
        sol.scale(0.5)

        ksp.destroy()
        lhs.destroy()
        rhs.destroy()

        return sol.sum()/comm.size

    def objective_function_lstsq_ad(self, x, x0, cov):
        """
        Adjoint of the nonlinear least squares objective function
        """
        ksp = self._initialise_ksp(cov, pc='lu')

        misfit = np.array(x - x0)
        lhs, rhs = cov.createVecs()
        rhs.set(0.0)
        lindices = np.arange(0, misfit.size, dtype=PETSc.IntType)
        rhs.setValues(lindices, misfit, PETSc.InsertMode.ADD_VALUES)
        rhs.assemble()
        ksp.solve(rhs, lhs)

        toall, allvec = PETSc.Scatter.toAll(lhs)
        toall.scatter(lhs, allvec, PETSc.InsertMode.INSERT)

        ksp.destroy()
        lhs.destroy()
        return allvec.array


    def map(self, *args):
        """
        Requires a tuple of vectors corresponding to an inversion variable
        these are mapped to the mesh.

        tuple(vec1, vec2, vecN) --> tuple(field1, field2, fieldN)
        """

        nf = len(args)
        nl = len(self.lithology_index)

        # preallocate local memory
        mesh_variables = np.zeros((nf, self.mesh.nn))

        # unpack vector to field
        for i in range(0, nl):
            idx = self.lithology_mask[i]
            for f in range(nf):
                mesh_variables[f,idx] = args[f][i]

        # sync fields across processors
        for f in range(nf):
            mesh_variables[f] = self.mesh.sync(mesh_variables[f])

        return list(mesh_variables)

    def map_ad(self, *args):
        """
        Map mesh variables back to the list
        """
        
        nf = len(args)
        nl = len(self.lithology_index)

        # sync fields across processors
        mesh_variables = np.zeros((nf, self.mesh.nn))
        for f in range(nf):
            mesh_variables[f] = self.mesh.sync(args[f])

        lith_variables = np.zeros((nf, self.lithology_index.size))
        all_lith_variables = np.zeros_like(lith_variables)

        for i in range(0, nl):
            idx = self.lithology_mask[i]
            for f in range(nf):
                lith_variables[f,i] += (mesh_variables[f]/self.ghost_weights)[idx].sum()

        comm.Allreduce([lith_variables, MPI.DOUBLE], [all_lith_variables, MPI.DOUBLE], op=MPI.SUM)

        return list(all_lith_variables)


    def create_wall_map(self, wall, *args):
        coords = self.mesh.coords
        dim = self.mesh.dim

        bbox = self.mesh.dm.getBoundingBox()
        sizes = self.mesh.dm.getSizes()

        # Setup boundary dictionary
        bc = dict()

        wall = [("minX", "maxX"), ("minY", "maxY"), ("minZ", "maxZ")]

        for i in range(0, dim):
            w0, w1 = wall[i]
            c0, c1 = bbox[i]
            m0, m1 = coords[:,i] == c0, coords[:,i] == c1

        
        self.boundary_index = boundary_index


    def map_wall(self, wall, *args):
        """
        Map lists of arguments to a boundary wall


        """

        if wall not in self.mesh.bc:
            raise ValueError("wall must be one of {}".format(self.mesh.bc.keys()))
        if len(args) +1 != self.mesh.dim:
            # +1 because it's a plane
            raise ValueError("dimensions of lists must equal number of dimensions")

        axis = None

        sizes = list(self.mesh.dm.getSizes())
        sizes.pop(axis)
        extent = np.reshape(self.mesh.extent, (-1,2))
        extent_bc = np.delete(extent, axis, axis=0)

        gcoords = []
        sizes = []
        for i in self.mesh.dim:
            if i != axis:
                coords = self.mesh.grid_coords[i]
                gcoords.append(coords)
                sizes.append(coords.size)

        ix = np.meshgrid(gcoords)
        
        # 0. divide wall into chunks based on the length of args
        # 1. find if proc contains bc mask
        # 2. map chunk to the bc



        bc_mask = self.mesh.bc[bc]['mask']
        if bc_mask.any():
            pass



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
                matrix = self.mesh.construct_matrix(in_place=True)
            if rhs == None:
                rhs = self.mesh.rhs
            rhs[:] = dT

            gvec = self.mesh.gvec
            lvec = self.mesh.lvec

            res = self.mesh.temperature
            res[:] = T

            # adjoint b vec
            db_ad = lvec.duplicate()

            gvec.setArray(rhs._gdata)
            self.ksp.solveTranspose(rhs._gdata, gvec)
            self.mesh.dm.globalToLocal(gvec, db_ad)

            # adjoint A mat
            dk_ad = np.zeros_like(T)

            matrix.scale(-1.0)
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
                    self.ksp.solve(dAdklT, gvec)
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
