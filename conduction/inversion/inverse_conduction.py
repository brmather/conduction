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

class Inversion(object):

    def __init__(self, lithology, mesh):
        self.mesh = mesh
        self.lithology = np.array(lithology).ravel()

        # communicate lithology index
        lithology_index = np.unique(lithology)
        lith_min, lith_max = np.array(lithology_index.min(), dtype=float), np.array(lithology_index.max(), dtype=float)
        all_lith_min, all_lith_max = np.array(0.0), np.array(0.0)
        comm.Allreduce([lith_min, MPI.DOUBLE], [all_lith_min, MPI.DOUBLE], op=MPI.MIN)
        comm.Allreduce([lith_max, MPI.DOUBLE], [all_lith_max, MPI.DOUBLE], op=MPI.MAX)

        # self.lithology_index = np.unique(lithology)
        # self.lithology_index.sort()
        self.lithology_index = np.arange(int(all_lith_min), int(all_lith_max)+1)

        self.lithology_mask = np.zeros((len(self.lithology_index), mesh.nn), dtype=bool)

        for i, index in enumerate(self.lithology_index):
            self.lithology_mask[i] = self.lithology == index


        minX, minY, minZ = mesh.coords.min(axis=0)
        maxX, maxY, maxZ = mesh.coords.max(axis=0)


        Xcoords = np.unique(mesh.coords[:,0])
        Ycoords = np.unique(mesh.coords[:,1])
        Zcoords = np.unique(mesh.coords[:,2])

        nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size

        self.dx = (maxX - minX)/nx
        self.dy = (maxY - minY)/ny
        self.dz = (maxZ - minZ)/nz

        self.nx = nx
        self.ny = ny
        self.nz = nz


        self.interp = RegularGridInterpolator(mesh.grid_coords[::-1],
                                              np.zeros(mesh.n),
                                              bounds_error=False, fill_value=np.nan)


        mesh.lvec.set(1.0)
        mesh.gvec.set(0.0)
        mesh.dm.localToGlobal(mesh.lvec, mesh.gvec, addv=True)
        mesh.dm.globalToLocal(mesh.gvec, mesh.lvec)
        self.ghost_weights = np.rint(mesh.lvec.array)

        # print comm.rank, mesh.lvec.getSizes(), mesh.gvec.getSizes()
        # print comm.rank, "global", mesh.gvec.array
        # print comm.rank, "local", mesh.lvec.array.reshape(mesh.nz, mesh.ny, mesh.nx)


        # Cost function variables
        self.observation = {}
        self.prior = {}


        # Initialise linear solver
        self.ksp = self._initialise_ksp()
        self.ksp_T = self._initialise_ksp() # <- need to pass transposed mat

        self.temperature = self.mesh.gvec.duplicate()
        self._temperature = self.mesh.gvec.duplicate()


    def _initialise_ksp(self, matrix=None, solver='bcgs', atol=1e-10, rtol=1e-50):
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

    
    def add_prior(self, **kwargs):
        """
        All priors will be inversion variables,
        but not all inversion variables will have priors.

            ARGS
                prior  : tuple(prior, uncertainty)

        """
        for arg in kwargs:
            p = list(kwargs[arg])
            p[0] = np.ma.array(p[0], mask=p[1]==0.0)
            self.prior[arg] = p


    def add_observation(self, **kwargs):
        """
            ARGS
                obs    : tuple(obs, uncertainty, coords)


        Similar to add_prior() but interpolates onto the mesh
        """
        fill_value = self.interp.fill_value
        nx, ny, nz = self.nx, self.ny, self.nz

        for arg in kwargs:
            o = list(kwargs[arg])
            o[0] = np.ma.array(o[0], mask=o[1]==0.0)
            if len(o) == 2 or o[-1] is None:
                # constant across the whole field
                ghost_weight = 1.0/self.ghost_weights
            else:
                xi = o[2]
                self.interp.fill_value = -1.
                self.interp.values = self.ghost_weights.reshape(self.mesh.n)
                w = self.interp(xi)
                ghost_weight = 1.0/np.floor(w+1e-12) # eliminate round-off error
                ghost_weight[ghost_weight==-1.] = 0.0

            o.append(ghost_weight)
            self.observation[arg] = o

        self.interp.fill_value = fill_value


    def add_observation_new(self, **kwargs):

        interp = self.interp
        interp.values = self.ghost_weights.reshape(self.mesh.n)

        for arg in kwargs:
            obs = InvObservation(interp, *kwargs[arg])
            self.observation[arg] = obs

    def add_prior_new(self, **kwargs):

        for arg in kwargs:
            prior = InvPrior(*kwargs[arg])
            self.prior[arg] = prior


    def interpolate(field, xi):
        self.interp.values = field.reshape(self.mesh.n)
        return self.interp(xi)


    def cost(self, x, inv_obj):
        x[np.isnan(x)] = 0.0
        c = (x - inv_obj.v)**2/inv_obj.dv**2
        c *= inv_obj.gweight
        return c.sum()

    def cost_ad(self, x, inv_obj):
        x[np.isnan(x)] = 0.0
        dc = (2.0*x - 2.0*inv_obj.v)/inv_obj.dv**2
        dc *= inv_obj.gweight
        return dc


    def objective_function(self, x, x0, sigma_x0, ghost_weight=1.0):
        x = np.ma.array(x, mask=np.isnan(x))
        C = (x - x0)**2/sigma_x0**2
        C *= ghost_weight
        return C.sum()

    def objective_function_ad(self, x, x0, sigma_x0, ghost_weight=1.0):
        x = np.array(x)
        C_ad = (2.0*x - 2.0*x0)/sigma_x0**2
        C_ad *= ghost_weight
        return C_ad


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

        if matrix == None:
            matrix = self.mesh.construct_matrix(in_place=False)
        if rhs == None:
            rhs = self.mesh.construct_rhs(in_place=False)

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
        solve_lith = np.array(True)

        matrix.scale(-1.0)
        self.mesh.boundary_condition('maxZ', 0.0, flux=False)
        dT_ad = dT[:]
        
        nl = len(self.lithology_index)
        for i in range(0, nl):
            idx = self.lithology_mask[i]
            idx_n = np.logical_and(idx, dT_ad != 0.0)
            ng = idx_n.any()
            comm.Allreduce([ng, MPI.BOOL], [solve_lith, MPI.BOOL], op=MPI.LOR)
            if solve_lith:
                self.mesh.diffusivity[:] = idx
                dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
                dAdklT = dAdkl * res._gdata
                self.ksp.solve(dAdklT, gvec)
                self.mesh.dm.globalToLocal(gvec, lvec)
                dk_ad[idx] += dT_ad.dot(lvec.array)/idx_n.sum()

        return dk_ad, db_ad.array

    def gradient(self, T):
        gradT = np.gradient(T.reshape(self.mesh.n), *self.mesh.grid_coords[::-1])
        return gradT

    def gradient_ad(self, dT, gradT, T):
        # gradT = np.gradient(T.reshape(self.mesh.n), *self.mesh.grid_coords[::-1])
        for i in range(0, self.mesh.dim):
            delta = np.mean(np.diff(self.mesh.grid_coords[::-1][i]))
            dT += gradT[i]/(self.mesh.n[i]*delta)
        return dT


    def heatflux(self, T, k):
        gradT = self.gradient(T)
        kn = -k.reshape(self.mesh.n)
        # self.mesh.create_meshVariable('heatflux')
        q = kn*np.array(gradT)
        return q.sum(axis=0)

    def heatflux_ad(dq, q, T, k):
        gradT = self.gradient(T)
        kn = -k.reshape(self.mesh.n)

        dqdk = np.array(gradT).sum(axis=0)
        dk = dqdk*dq

        dqdgradT = kn
        dT = np.zeros_like(kn)
        for i in range(0, self.mesh.dim):
            delta = np.mean(np.diff(self.mesh.grid_coords[::-1][i]))
            dqdT = kn/(self.mesh.n[i]*delta)
            dT += dqdT*dq
        
        return dT, dk


    def forward_model(self, x):
        """
        x : inversion variables vector
        need to map x to map() and user defined functions
        
        k and H are compulsory (should they be specified in the first part of x?)
        
        """
        dx, dy, dz = self.dx, self.dy, self.dz
        nx, ny, nz = self.nx, self.ny, self.nz
        (minX, maxX), (minY, maxY), (minZ, maxZ) = self.mesh.dm.getLocalBoundingBox()
        minBounds = (minX, minY, minZ)
        maxBounds = (maxX, maxY, maxY)

        k_list, H_list, a_list = np.array_split(x.array[:-1], 3)
        q0 = x.array[-1]

        # Unpack vectors onto mesh
        k0, H, a = self.map(k_list, H_list, a_list)
        self.mesh.update_properties(k0, H)
        self.mesh.boundary_condition('maxZ', 298.0, flux=False)
        self.mesh.boundary_condition('minZ', q0, flux=True)
        b = self.mesh.construct_rhs()

        k = k0.copy()
        error_local  = np.array(True)
        error_global = np.ones(comm.size, dtype=bool)
        i = 0
        while error_global.any():
            k_last = k.copy()

            self.mesh.update_properties(k, H)
            A = self.mesh.construct_matrix()
            self.ksp.setOperators(A)
            self.ksp.solve(b._gdata, self.temperature)
            self.mesh.dm.globalToLocal(self.temperature, self.mesh.lvec)
            T = self.mesh.lvec.array.copy()

            k = k0*(298.0/T)**a

            error_local = np.absolute(k - k_last).max() > 1e-6
            comm.Allgather([error_local, MPI.BOOL], [error_global, MPI.BOOL])
            i += 1

        idx_lowerBC = self.mesh.bc['minZ']['mask']
        idx_upperBC = self.mesh.bc['maxZ']['mask']

        # print gradT[nz//2,0,:]
        # print T.reshape(nz,ny,nx)[nz//2, -1, :]
        # print T[idx_lowerBC]

        cost = np.array(0.0)
        sum_cost = np.array(0.0)

        # Cost observations
        if 'q' in self.observation:
            obs = self.observation['q']

            # Compute heat flux
            gradTz, gradTy, gradTx = np.gradient(T.reshape(self.mesh.n), *self.mesh.grid_coords[::-1])
            heatflux = -k.reshape(self.mesh.n)*(gradTz + gradTy + gradTx)
            q_interp = heatflux.ravel()
            if obs[2] is not None:
                self.interp.values = heatflux
                q_interp = self.interp(obs[2], method='nearest')

            cost += self.objective_function(q_interp, obs[0], obs[1], obs[-1])


        if 'T' in self.observation:
            obs = self.observation['T']
            T_interp = T
            if obs[2] is not None:
                self.interp.values = T.reshape(self.mesh.n)
                T_interp = self.interp(obs[2], method='nearest')
            # out_of_bounds = np.zeros(obs[2].shape[0], dtype=bool)
            # for i, xi in enumerate(obs[2]):
            #     out_of_bounds[i] += (xi < minBounds).any()
            #     out_of_bounds[i] += (xi > maxBounds).any()
            # out_of_bounds[np.isnan(T_interp)] = False
            # T_interp[out_of_bounds] = 0.0
            
            cost += self.objective_function(T_interp, obs[0], obs[1], obs[-1])
            # C = (T_interp - obs[0])**2/obs[1]**2
            # C /= self.ghost_weights
            # cost += C.sum()

        comm.Allreduce([cost, MPI.DOUBLE], [sum_cost, MPI.DOUBLE], op=MPI.SUM)


        # Cost priors
        for key, array in [('T',T), ('k',k_list), ('H',H_list), ('a',a_list), ('q0',q0)]:
            if key in self.prior:
                prior = self.prior[key]
                sum_cost += self.objective_function(array, prior[0], prior[1])
        

        return sum_cost



    def tangent_linear(self, x, dx):
        hx, hy, hz = self.dx, self.dy, self.dz
        nx, ny, nz = self.nx, self.ny, self.nz

        k_list, H_list, a_list = np.array_split(x.array[:-1], 3)
        q0 = x.array[-1]
        dk_list, dH_list, da_list = np.array_split(dx.array[:-1], 3)
        dq0 = dx.array[-1]

        # Unpack vectors onto mesh
        k0, H, a = self.map(k_list, H_list, a_list)
        dk0, dH, da = self.map(dk_list, dH_list, da_list)

        

        dAdklT = self.mesh.gvec.duplicate()
        k = k0.copy()
        dk = dk0.copy()

        error_local  = np.array([True])
        error_global = np.ones(comm.size, dtype=bool)
        while error_global.any():
            k_last = k.copy()

            self.mesh.update_properties(k, H)
            self.mesh.boundary_condition('maxZ', 298.0, flux=False)
            self.mesh.boundary_condition('minZ', q0, flux=True)
            A = self.mesh.construct_matrix()
            b = self.mesh.construct_rhs()

            self.ksp.solve(b._gdata, self.temperature)

            self.mesh.update_properties(dk, dH)
            self.mesh.boundary_condition('maxZ', 0.0, flux=False)
            self.mesh.boundary_condition('minZ', dq0, flux=True)
            dA = self.mesh.construct_matrix(in_place=False, derivative=True)
            db = self.mesh.construct_rhs(in_place=False)
            
            # dT = A-1*db - A-1*dA*A-1*b
            self.ksp.solve(db._gdata, self._temperature)


            A.scale(-1.0)

            x1 = dA*self.temperature
            self.ksp.solve(x1, self.mesh.gvec)

            self._temperature += self.mesh.gvec


            # dA.mult(self.temperature, self.mesh.gvec)
            # self.ksp.solve(self.mesh.gvec, dT_2)

            # for lith in self.lithology_index:
            #     idx = self.lithology == lith
            #     self.mesh.diffusivity.fill(0.0)
            #     self.mesh.diffusivity[idx] = 1.0
            #     dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
            #     dAdkl.mult(self.temperature, dAdklT)
            #     self.ksp.solve(dAdklT, self.mesh.gvec)
            #     dT_2.array[idx] = self.mesh.gvec.array[idx]



            self.mesh.dm.globalToLocal(self.temperature, self.mesh.lvec)
            T = self.mesh.lvec.array.copy()
            self.mesh.dm.globalToLocal(self._temperature, self.mesh.lvec)
            dT = self.mesh.lvec.array.copy()

            dkda = np.log(298.0/T)*k0*(298.0/T)**a
            dkdk0 = (298.0/T)**a
            dkdT = -a*k0/T*(298.0/T)**a

            k = k0*(298.0/T)**a
            dk = dkda*da + dkdk0*dk0 + dkdT*dT

            error_local[0] = np.absolute(k - k_last).max() > 1e-6
            comm.Allgather([error_local, MPI.BOOL], [error_global, MPI.BOOL])



        cost = np.array(0.0)
        sum_cost = np.array(0.0)
        dc = np.array(0.0)
        sum_dc = np.array(0.0)

        # Cost observations
        if 'q' in self.observation:
            obs = self.observation['q']

            # Compute heat flux
            gradTz, gradTy, gradTx = np.gradient(T.reshape(self.mesh.n), *self.mesh.grid_coords[::-1])
            heatflux = -k.reshape(self.mesh.n)*(gradTz + gradTy + gradTx)
            q_interp = heatflux.ravel()
            
            if obs[2] is not None:
                self.interp.values = heatflux
                q_interp = self.interp(obs[2], method='nearest')
            cost += self.objective_function(q_interp, obs[0], obs[1], obs[-1])

            # dqdT = k
            dqdTz = -k/(nz*hz)
            dqdTy = -k/(ny*hy)
            dqdTx = -k/(nx*hx)
            dqdk = -(gradTz + gradTy + gradTx)

            # dq = dqdT*dT + dqdk.ravel()*dk
            dq = dqdTz*dT + dqdTy*dT + dqdTx*dT + dqdk.ravel()*dk
            dq_interp = dq
            if obs[2] is not None:
                self.interp.values = dq.reshape(self.mesh.n)
                dq_interp = self.interp(obs[2], method='nearest')
            # print obs[-1], "\n", q_interp, "\n", dq_interp
            dcdq = self.objective_function_ad(q_interp, obs[0], obs[1], obs[-1])
            # print "tl", dcdq
            dc += np.sum(dcdq*dq_interp)


        if 'T' in self.observation:
            obs = self.observation['T']
            T_interp = T
            if obs[2] is not None:
                self.interp.values = T.reshape(self.mesh.n)
                T_interp = self.interp(obs[2], method='nearest')
            cost += self.objective_function(T_interp, obs[0], obs[1], obs[-1])

            dT_interp = dT
            if obs[2] is not None:
                self.interp.values = dT.reshape(self.mesh.n)
                dT_interp = self.interp(obs[2], method='nearest')
            dcdT = self.objective_function_ad(T_interp, obs[0], obs[1], obs[-1])
            dc += np.sum(dcdT*dT_interp)

        comm.Allreduce([cost, MPI.DOUBLE], [sum_cost, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([dc, MPI.DOUBLE], [sum_dc, MPI.DOUBLE], op=MPI.SUM)


        # Cost priors
        for key, array, darray in [('k',k_list, dk_list),
                                   ('H',H_list, dH_list),
                                   ('a',a_list, da_list),
                                   ('q0',q0,dq0)]:

            if key in self.prior:
                prior = self.prior[key]
                sum_cost += self.objective_function(array, prior[0], prior[1])

                dcdp = self.objective_function_ad(array, prior[0], prior[1])
                sum_dc += np.sum(dcdp*darray)


        return sum_cost, sum_dc




    def adjoint(self, tao, x, G):
        dx, dy, dz = self.dx, self.dy, self.dz
        nx, ny, nz = self.nx, self.ny, self.nz
        (minX, maxX), (minY, maxY), (minZ, maxZ) = self.mesh.dm.getLocalBoundingBox()

        k_list, H_list, a_list = np.array_split(x.array[:-1], 3)
        q0 = x.array[-1]

        # Unpack vectors onto mesh
        k0, H, a = self.map(k_list, H_list, a_list)
        self.mesh.update_properties(k0, H)
        self.mesh.boundary_condition('maxZ', 298.0, flux=False)
        self.mesh.boundary_condition('minZ', q0, flux=True)
        b = self.mesh.construct_rhs()

        k = [k0]
        T = [None]

        error_local  = np.array([True])
        error_global = np.ones(comm.size, dtype=bool)
        i = 0
        while error_global.any():
            self.mesh.update_properties(k[i], H)
            A = self.mesh.construct_matrix()
            self.ksp.solve(b._gdata, self.temperature)
            self.mesh.dm.globalToLocal(self.temperature, self.mesh.lvec)

            T.append(self.mesh.lvec.array.copy())
            k.append(k0*(298.0/T[-1])**a)

            error_local[0] = np.absolute(k[-1] - k[-2]).max() > 1e-6
            comm.Allgather([error_local, MPI.BOOL], [error_global, MPI.BOOL])
            i += 1


        dT_ad = np.zeros_like(k0)
        dk_ad = np.zeros_like(k0)
        dH_ad = np.zeros_like(k0)
        da_ad = np.zeros_like(k0)
        dq0_ad = np.array(0.0)
        dk_list_ad = np.zeros_like(k_list)
        dH_list_ad = np.zeros_like(H_list)
        da_list_ad = np.zeros_like(a_list)
        dq0_list_ad = np.array(0.0)

        cost = np.array(0.0)
        sum_cost = np.array(0.0)

        # Cost observations
        if self.observation.has_key('q'):
            obs = self.observation['q']

            # Compute heat flux
            gradTz, gradTy, gradTx = np.gradient(T[-1].reshape(self.mesh.n), *self.mesh.grid_coords[::-1])
            heatflux = -k[-1].reshape(self.mesh.n)*(gradTz + gradTy + gradTx)
            q_interp = heatflux.ravel()
            
            if obs[2] is not None:
                self.interp.values = heatflux
                q_interp = self.interp(obs[2], method='nearest')

            cost += self.objective_function(q_interp, obs[0], obs[1], obs[-1])

            ## AD ##
            dcdq = self.objective_function_ad(q_interp, obs[0], obs[1])
            dq_ad = dcdq*1.0
            if obs[2] is not None:
                dq_interp_ad = dcdq*1.0
                dq_ad = self.interp.adjoint(obs[2], dq_interp_ad, method='nearest').ravel()
                # print "ad\n", dq_interp_ad
                self.mesh.lvec.setArray(dq_ad)
                self.mesh.dm.localToGlobal(self.mesh.lvec, self.mesh.gvec)
                self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
                dq_ad = self.mesh.lvec.array.copy() #/self.ghost_weights
                # print "dq_interp_ad", dq_ad_interp
                # print obs[-1]
                # print "dq_ad\n", dq_ad.min(), dq_ad.mean(), dq_ad.max()
                # print "dq_ad\n", np.hstack([dq_ad[dq_ad>0].reshape(-1,1), self.mesh.coords[dq_ad>0]])
                # print "np.nan", np.where(dq_ad==np.nan)

            dqdTz = -k[-1]/(nz*dz)
            dqdTy = -k[-1]/(ny*dy)
            dqdTx = -k[-1]/(nx*dz)
            dqdk = -(gradTz + gradTy + gradTx).ravel()

            dk_ad += dqdk*dq_ad
            dT_ad += dqdTx*dq_ad + dqdTy*dq_ad + dqdTz*dq_ad
            # print dT_ad.min(), dT_ad.max()

        if self.observation.has_key('T'):
            obs = self.observation['T']
            T_interp = T[-1]
            if obs[2] is not None:
                self.interp.values = T[-1].reshape(self.mesh.n)
                T_interp = self.interp(obs[2], method='nearest')
            cost += self.objective_function(T_interp, obs[0], obs[1], obs[-1])

            ## AD ##
            dcdT = self.objective_function_ad(T_interp, obs[0], obs[1])
            dT_ad2 = dcdT*1.0
            if obs[2] is not None:
                dT_interp_ad = dcdT*1.0
                dT_ad2 = self.interp.adjoint(obs[2], dq_interp_ad, method='nearest').ravel()
            
            self.mesh.lvec.setArray(dT_ad2)
            self.mesh.dm.localToGlobal(self.mesh.lvec, self.mesh.gvec)
            self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
            dT_ad2 = self.mesh.lvec.array.copy()
            # print "dT_ad\n", dT_ad2.min(), dT_ad2.mean(), dT_ad2.max()

            dT_ad += dT_ad2


        comm.Allreduce([cost, MPI.DOUBLE], [sum_cost, MPI.DOUBLE], op=MPI.SUM)

        # Cost priors
        for key, array, array_ad in [('k', k_list, dk_list_ad),
                                     ('H', H_list, dH_list_ad),
                                     ('a', a_list, da_list_ad),
                                     ('q0', q0, dq0_list_ad)]:
            if self.prior.has_key(key):
                prior = self.prior[key]
                sum_cost += self.objective_function(array, prior[0], prior[1])

                ## AD ##
                dcdp = self.objective_function_ad(array, prior[0], prior[1])
                # array_ad += dcdp*1.0

        # print "dT", comm.rank, dT_ad
        # print "dK", comm.rank, dk_ad

        dk0_ad = np.zeros_like(k0)

        idx_local = np.array([True])
        idx_global = np.ones(comm.size, dtype=bool)

        idx_lowerBC = self.mesh.bc['minZ']['mask']
        idx_upperBC = self.mesh.bc['maxZ']['mask']


        kspT = PETSc.KSP().create(comm)
        kspT.setType('bcgs')
        kspT.setTolerances(1e-12, 1e-12)
        kspT.setFromOptions()
        # kspT.setDM(self.mesh.dm)
        # pc = kspT.getPC()
        # pc.setType('gamg')

        dAdklT = self.mesh.gvec.duplicate()


        for j in range(i):
            dkda = np.log(298.0/T[-1-j])*k0*(298.0/T[-1-j])**a
            dkdk0 = (298.0/T[-1-j])**a
            dkdT = -a*k0/T[-1-j]*(298.0/T[-1-j])**a

            dk0_ad += dkdk0*dk_ad
            dT_ad  += dkdT*dk_ad
            da_ad  += dkda*dk_ad

            dk_ad.fill(0.0)


            self.mesh.update_properties(k[-1-j], H)
            self.mesh.boundary_condition('maxZ', 298.0, flux=False)
            self.mesh.boundary_condition('minZ', q0, flux=True)
            A = self.mesh.construct_matrix()


            AT = self.mesh._initialise_matrix()
            A.transpose(AT)
            self.mesh.lvec.setArray(dT_ad)
            self.mesh.dm.localToGlobal(self.mesh.lvec, b._gdata)

            kspT.setOperators(AT)
            kspT.solve(b._gdata, self.mesh.gvec)
            self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
            db_ad = self.mesh.lvec.array

            dH_ad += -db_ad
            dH_ad[idx_lowerBC] += db_ad[idx_lowerBC]/dy
            dq0_ad += np.sum(-db_ad[idx_lowerBC]/dy/self.ghost_weights[idx_lowerBC])

            A.scale(-1.0)


            # self.mesh.boundary_condition('maxZ', 0.0, flux=False)
            # self.mesh.diffusivity.fill(1.0)
            # dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)

            # self.mesh.lvec.setArray(T[-1-j])
            # self.mesh.dm.localToGlobal(self.mesh.lvec, self._temperature)
            # dAdkl.mult(self._temperature, dAdklT)
            # self.ksp.solve(dAdklT, self.mesh.gvec)

            # dk_ad += dT_ad.dot(self.mesh.lvec.array)

            self.mesh.lvec.setArray(T[-1-j])
            self.mesh.dm.localToGlobal(self.mesh.lvec, self._temperature)


            kappa = np.zeros_like(H)

            for l, lith in enumerate(self.lithology_index):
                idx = self.lithology == lith
                idx_dT = dT_ad != 0.0
                idx_n  = np.logical_and(idx, idx_dT)
                idx_local[0] = idx_n.any()
                comm.Allgather([idx_local, MPI.BOOL], [idx_global, MPI.BOOL])
                if idx_global.any():
                    self.mesh.boundary_condition('maxZ', 0.0, flux=False)
                    kappa.fill(0.0)
                    kappa[idx] = 1.0
                    self.mesh.diffusivity[:] = kappa
                    dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
                    # diag = dAdkl.getDiagonal()
                    # diag.array[idx_upperBC] = 0.0
                    # dAdkl.setDiagonal(diag)
                    dAdkl.mult(self._temperature, dAdklT)
                    self.ksp.setOperators(A)
                    self.ksp.solve(dAdklT, self.mesh.gvec)
                    self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
                    if idx_local[0]:
                        dk_ad[idx_n] += dT_ad.dot(self.mesh.lvec.array)/idx_n.sum()
                    # print self.mesh.lvec.array.mean(), dk_ad.mean(), dk0_ad.mean(), idx_n.any(), idx_n.sum()


            dT_ad.fill(0.0)

        dk0_ad += dk_ad

        kspT.destroy()

        dk0_ad /= self.ghost_weights
        dH_ad /= self.ghost_weights
        da_ad /= self.ghost_weights

        for i, index in enumerate(self.lithology_index):
            idx = self.lithology == index
            dk_list_ad[i] += dk0_ad[idx].sum()
            dH_list_ad[i] += dH_ad[idx].sum()
            da_list_ad[i] += da_ad[idx].sum()


        sum_dk_list_ad = np.zeros_like(dk_list_ad)
        sum_dH_list_ad = np.zeros_like(dk_list_ad)
        sum_da_list_ad = np.zeros_like(dk_list_ad)
        sum_dq0_ad = np.array(0.0)

        comm.Allreduce([dk_list_ad, MPI.DOUBLE], [sum_dk_list_ad, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([dH_list_ad, MPI.DOUBLE], [sum_dH_list_ad, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([da_list_ad, MPI.DOUBLE], [sum_da_list_ad, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([dq0_ad, MPI.DOUBLE], [sum_dq0_ad, MPI.DOUBLE], op=MPI.SUM)


        dq0_list_ad += sum_dq0_ad

        # Procs have unique lithololgy, need to communicate the gradients after vectors are all been packed up
        # I think these fellows ought to have their prior sensitivities added at the end since these are global.
        # for i, index in enumerate(self.lithology_index):
        #     idx = self.lithology == index
        #     dk_list_ad[i] += sum_dk0_ad[idx].sum()
        #     dH_list_ad[i] += sum_dH_ad[idx].sum()
        #     da_list_ad[i] += sum_da_ad[idx].sum()

        # procs should have their part of the sensitivities summed. Even if this doesn't result in any difference,
        # performance should be improved by communicating smaller arrays

        for key, array, array_ad in [('k', k_list, sum_dk_list_ad),
                                     ('H', H_list, sum_dH_list_ad),
                                     ('a', a_list, sum_da_list_ad),
                                     ('q0', q0, sum_dq0_ad)]:
            if key in self.prior:
                prior = self.prior[key]
                array_ad += self.objective_function_ad(array, prior[0], prior[1])


        G.setArray(np.concatenate([sum_dk_list_ad, sum_dH_list_ad, sum_da_list_ad, [sum_dq0_ad]]))

        print("cost = {}".format(sum_cost))
        
        return sum_cost