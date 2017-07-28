try: range = xrange
except: pass

import numpy as np
from ..interpolation import RegularGridInterpolator
from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD

class Inversion(object):

    def __init__(self, lithology, mesh):
        self.mesh = mesh
        self.lithology = np.array(lithology).ravel()
        self.lithology_index = np.unique(lithology)
        self.lithology_index.sort()

        minX, minY, minZ = mesh.coords.min(axis=0)
        maxX, maxY, maxZ = mesh.coords.max(axis=0)

        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        Xcoords = np.linspace(minX, maxX, nx)
        Ycoords = np.linspace(minY, maxY, ny)
        Zcoords = np.linspace(minZ, maxZ, nz)


        self.interp = RegularGridInterpolator((Zcoords, Ycoords, Xcoords),
                                               np.zeros((mesh.nz, mesh.ny, mesh.nx)),
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
        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setType('gmres')
        self.ksp.setOperators(self.mesh.mat)
        # self.ksp.setComputeOperators(self.mesh.mat)
        # self.ksp.setDMActive(True)
        # self.ksp.setDM(self.mesh.dm)
        self.ksp.setTolerances(1e-12, 1e-12)
        self.ksp.setFromOptions()


        self.temperature = self.mesh.gvec.duplicate()
        self._temperature = self.mesh.gvec.duplicate()


    
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
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz

        for arg in kwargs:
            o = list(kwargs[arg])
            o[0] = np.ma.array(o[0], mask=o[1]==0.0)
            if len(o) == 2 or o[-1] is None:
                # constant across the whole field
                ghost_weight = 1.0/self.ghost_weights
            else:
                xi = o[2]
                self.interp.fill_value = -1.
                self.interp.values = self.ghost_weights.reshape(nz,ny,nx)
                w = self.interp(xi)
                ghost_weight = 1.0/np.floor(w+1e-12) # eliminate round-off error
                ghost_weight[ghost_weight==-1.] = 0.0

            o.append(ghost_weight)
            self.observation[arg] = o

        self.interp.fill_value = fill_value


    def objective_function(self, x, x0, sigma_x0, ghost_weight=1.0):
        x = np.ma.array(x, mask=np.isnan(x))
        C = (x - x0)**2/sigma_x0**2
        C *= ghost_weight
        return C.sum()

    def objective_function_ad(self, x, x0, sigma_x0, ghost_weight=1.0):
        x = np.ma.array(x, mask=np.isnan(x))
        C_ad = (2.0*x - 2.0*x0)/sigma_x0**2
        C_ad *= ghost_weight
        return C_ad


    def map(self, *args):
        """
        Requires a tuple of vectors corresponding to an inversion variable
        these are mapped to the mesh.

        tuple(vec1, vec2, vecN) --> tuple(field1, field2, fieldN)
        """

        n = len(args)

        # preallocate memory
        mesh_variables = np.zeros((n, self.lithology.size))

        # unpack vector to field
        for i, index in enumerate(self.lithology_index):
            idx = self.lithology == index
            for f in range(n):
                mesh_variables[f,idx] = args[f][i]

        mesh_variables = np.vsplit(mesh_variables, n)
        for f in range(n):
            mesh_variables[f] = mesh_variables[f][0] # flatten array

        return mesh_variables

    def map_ad(self, *args):
        """
        Map mesh variables back to the list
        """
        
        n = len(args)

        lith_variables = np.zeros((n, self.lithology_index.size))

        for i, index in enumerate(self.lithology_index):
            idx = self.lithololgy == index
            for f in range(n):
                lith_variables[f,i] += args[f][idx].sum()

        lith_variables = np.vsplit(lith_variables, n)
        for f in range(n):
            lith_variables[f] = lith_variables[f][0]

        return lith_variables


    def forward_model(self, x):
        """
        x : inversion variables vector
        need to map x to map() and user defined functions
        
        k and H are compulsory (should they be specified in the first part of x?)
        
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
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

            self.mesh.diffusivity = k
            A = self.mesh.construct_matrix()
            self.ksp.solve(b, self.temperature)
            self.mesh.dm.globalToLocal(self.temperature, self.mesh.lvec)
            T = self.mesh.lvec.array.copy()

            k = k0*(298.0/T)**a

            error_local = np.absolute(k - k_last).max() > 1e-6
            comm.Allgather([error_local, MPI.BOOL], [error_global, MPI.BOOL])
            i += 1

        idx_lowerBC = self.mesh.bc['bottom']['mask']
        idx_upperBC = self.mesh.bc['top']['mask']

        # print gradT[nz//2,0,:]
        # print T.reshape(nz,ny,nx)[nz//2, -1, :]
        # print T[idx_lowerBC]

        cost = np.array(0.0)
        sum_cost = np.array(0.0)

        # Cost observations
        if 'q' in self.observation:
            obs = self.observation['q']

            # Compute heat flux
            gradTz, gradTy, gradTx = np.gradient(T.reshape(nz,ny,nx), dz,dy,dx)
            heatflux = k.reshape(nz,ny,nx)*(gradTz + gradTy + gradTx)
            q_interp = heatflux.ravel()
            if obs[2] is not None:
                self.interp.values = heatflux
                q_interp = self.interp(obs[2])

            cost += self.objective_function(q_interp, obs[0], obs[1], obs[-1])


        if 'T' in self.observation:
            obs = self.observation['T']
            T_interp = T
            if obs[2] is not None:
                self.interp.values = T.reshape(mesh.nz, mesh.ny, mesh.nx)
                T_interp = self.interp(obs[2])
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
        hx, hy, hz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz

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
            self.mesh.boundary_condition('top', 298.0, flux=False)
            self.mesh.boundary_condition('bottom', q0, flux=True)
            A = self.mesh.construct_matrix()
            b = self.mesh.construct_rhs()

            self.ksp.solve(b, self.temperature)

            self.mesh.update_properties(dk, dH)
            self.mesh.boundary_condition('top', 0.0, flux=False)
            self.mesh.boundary_condition('bottom', dq0, flux=True)
            dA = self.mesh.construct_matrix(in_place=False, derivative=True)
            db = self.mesh.construct_rhs(in_place=False)
            
            # dT = A-1*db - A-1*dA*A-1*b
            self.ksp.solve(db, self._temperature)


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
            gradTz, gradTy, gradTx = np.gradient(T.reshape(nz,ny,nx),hz,hy,hx)
            heatflux = k.reshape(nz,ny,nx)*(gradTz + gradTy + gradTx)
            q_interp = heatflux.ravel()
            
            if obs[2] is not None:
                self.interp.values = heatflux
                q_interp = self.interp(obs[2], method='nearest')
            cost += self.objective_function(q_interp, obs[0], obs[1], obs[-1])

            # dqdT = k
            dqdTz = k/(nz*hz)
            dqdTy = k/(ny*hy)
            dqdTx = k/(nx*hx)
            dqdk = gradTz + gradTy + gradTx

            # dq = dqdT*dT + dqdk.ravel()*dk
            dq = dqdTz*dT + dqdTy*dT + dqdTx*dT + dqdk.ravel()*dk
            dq_interp = dq
            if obs[2] is not None:
                self.interp.values = dq.reshape(nz,ny,nx)
                dq_interp = self.interp(obs[2], method='nearest')
            print obs[-1], "\n", q_interp, "\n", dq_interp
            dcdq = self.objective_function_ad(q_interp, obs[0], obs[1], obs[-1])
            dc += np.sum(dcdq*dq_interp)


        if 'T' in self.observation:
            obs = self.observation['T']
            T_interp = T
            if obs[2] is not None:
                self.interp.values = T.reshape(nz,ny,nx)
                T_interp = self.interp(obs[2])
            cost += self.objective_function(T_interp, obs[0], obs[1], obs[-1])

            dT_interp = dT
            if obs[2] is not None:
                self.interp.values = dT.reshape(nz,ny,nx)
                dT_interp = self.interp(obs[2])
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
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        (minX, maxX), (minY, maxY), (minZ, maxZ) = self.mesh.dm.getLocalBoundingBox()

        k_list, H_list, a_list = np.array_split(x.array[:-1], 3)
        q0 = x.array[-1]

        # Unpack vectors onto mesh
        k0, H, a = self.map(k_list, H_list, a_list)
        self.mesh.update_properties(k0, H)
        self.mesh.boundary_condition('top', 298.0, flux=False)
        self.mesh.boundary_condition('bottom', q0, flux=True)
        b = self.mesh.construct_rhs()

        k = [k0]
        T = [None]

        error_local  = np.array([True])
        error_global = np.ones(comm.size, dtype=bool)
        i = 0
        while error_global.any():
            self.mesh.diffusivity = k[i]
            A = self.mesh.construct_matrix()
            self.ksp.solve(b, self.temperature)
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
            gradTz, gradTy, gradTx = np.gradient(T[-1].reshape(nz,ny,nx), dz,dy,dx)
            heatflux = k[-1].reshape(nz,ny,nx)*(gradTz + gradTy + gradTx)
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
                self.mesh.lvec.setArray(dq_ad)
                self.mesh.dm.localToGlobal(self.mesh.lvec, self.mesh.gvec)
                self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
                dq_ad = self.mesh.lvec.array.copy() #/self.ghost_weights
                # print "dq_interp_ad", dq_ad_interp
                print obs[-1]
                print "dq_ad\n", dq_ad.min(), dq_ad.mean(), dq_ad.max()
                # print "dq_ad\n", np.hstack([dq_ad[dq_ad>0].reshape(-1,1), self.mesh.coords[dq_ad>0]])
                # print "np.nan", np.where(dq_ad==np.nan)

            dqdTz = k[-1]/(nz*dz)
            dqdTy = k[-1]/(ny*dy)
            dqdTx = k[-1]/(nx*dz)
            dqdk = (gradTz + gradTy + gradTx).ravel()

            dk_ad += dqdk*dq_ad
            dT_ad += dqdTx*dq_ad + dqdTy*dq_ad + dqdTz*dq_ad
            # print dT_ad.min(), dT_ad.max()

        if self.observation.has_key('T'):
            obs = self.observation['T']
            T_interp = T[-1]
            if obs[2] is not None:
                self.interp.values = T[-1].reshape(mesh.nz, mesh.ny, mesh.nx)
                T_interp = self.interp(obs[2])
            cost += self.objective_function(T_interp, obs[0], obs[1], obs[-1])

            ## AD ##
            dcdT = self.objective_function_ad(T_interp, obs[0], obs[1])
            dT_ad2 = dcdT*1.0
            if obs[2] is not None:
                dT_interp_ad = dcdT*1.0
                dT_ad2 = self.interp.adjoint(obs[2], dq_interp_ad).ravel()
            
            self.mesh.lvec.setArray(dT_ad2)
            self.mesh.dm.localToGlobal(self.mesh.lvec, self.mesh.gvec)
            self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
            dT_ad2 = self.mesh.lvec.array.copy()
            print "dT_ad\n", dT_ad2.min(), dT_ad2.mean(), dT_ad2.max()

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

        idx_lowerBC = self.mesh.bc['bottom']['mask']
        idx_upperBC = self.mesh.bc['top']['mask']


        kspT = PETSc.KSP().create(comm)
        kspT.setType('gmres')
        kspT.setTolerances(1e-12, 1e-12)
        kspT.setFromOptions()
        # kspT.setDM(self.mesh.dm)
        pc = kspT.getPC()
        pc.setType('gamg')

        dAdklT = self.mesh.gvec.duplicate()


        for j in range(i):
            dkda = np.log(298.0/T[-1-j])*k0*(298.0/T[-1-j])**a
            dkdk0 = (298.0/T[-1-j])**a
            dkdT = -a*k0/T[-1-j]*(298.0/T[-1-j])**a

            dk0_ad += dkdk0*dk_ad
            dT_ad  += dkdT*dk_ad
            da_ad  += dkda*dk_ad

            dk_ad.fill(0.0)


            self.mesh.diffusivity = k[-1-j]
            self.mesh.boundary_condition('top', 298.0, flux=False)
            self.mesh.boundary_condition('bottom', q0, flux=True)
            A = self.mesh.construct_matrix()


            AT = A.copy()
            AT.transpose()
            self.mesh.lvec.setArray(dT_ad)
            self.mesh.dm.localToGlobal(self.mesh.lvec, b)

            kspT.setOperators(AT)
            kspT.solve(b, self.mesh.gvec)
            self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
            db_ad = self.mesh.lvec.array

            dH_ad += -db_ad
            dH_ad[idx_lowerBC] += db_ad[idx_lowerBC]/dy
            dq0_ad += np.sum(-db_ad[idx_lowerBC]/dy/self.ghost_weights[idx_lowerBC])

            A.scale(-1.0)


            # self.mesh.boundary_condition('top', 0.0, flux=False)
            # self.mesh.diffusivity.fill(1.0)
            # dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)

            # self.mesh.lvec.setArray(T[-1-j])
            # self.mesh.dm.localToGlobal(self.mesh.lvec, self._temperature)
            # dAdkl.mult(self._temperature, dAdklT)
            # self.ksp.solve(dAdklT, self.mesh.gvec)

            # dk_ad += dT_ad.dot(self.mesh.lvec.array)

            self.mesh.lvec.setArray(T[-1-j])
            self.mesh.dm.localToGlobal(self.mesh.lvec, self._temperature)


            for l, lith in enumerate(self.lithology_index):
                idx = self.lithology == lith
                idx_dT = dT_ad != 0.0
                idx_n  = np.logical_and(idx, idx_dT)
                idx_local[0] = idx_n.any()
                comm.Allgather([idx_local, MPI.BOOL], [idx_global, MPI.BOOL])
                if idx_global.any():
                    self.mesh.boundary_condition('top', 0.0, flux=False)
                    self.mesh.diffusivity.fill(0.0)
                    self.mesh.diffusivity[idx] = 1.0
                    dAdkl = self.mesh.construct_matrix(in_place=False, derivative=True)
                    # diag = dAdkl.getDiagonal()
                    # diag.array[idx_upperBC] = 0.0
                    # dAdkl.setDiagonal(diag)
                    dAdkl.mult(self._temperature, dAdklT)
                    self.ksp.solve(dAdklT, self.mesh.gvec)
                    self.mesh.dm.globalToLocal(self.mesh.gvec, self.mesh.lvec)
                    dk_ad[idx_n] += dT_ad.dot(self.mesh.lvec.array)/idx_n.sum()
                    # print self.mesh.lvec.array.mean(), dk_ad.mean(), dk0_ad.mean()


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

        return sum_cost