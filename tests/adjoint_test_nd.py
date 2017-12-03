
# coding: utf-8

# # Inversion
# 
# We benchmark the following models to the inverse heat conduction problem:
# 
# 1. Finite difference - difference between two forward models
# 2. Tangent linear - derivative of the problem in forward mode
# 3. Adjoint model - gradient of the objective function w.r.t. inversion variables

# In[1]:

import numpy as np
from time import clock
from conduction import ConductionND
from conduction.inversion import InvObservation, InvPrior
from conduction import InversionND
from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD


def forward_model(x, self, bc='Z'):
    """
    N-dimensional nonlinear model with flux lower BC
    Implements Hofmeister 1999 temperature-dependent
    conductivity law

    Arguments
    ---------
     x : [k_list, H_list, a_list, q0]

    Returns
    -------
     cost : scalar

    """
    def hofmeister1999(k0, T, a=0.25, c=0.0):
        return k0*(298.0/T)**a + c*T**3

    k_list, H_list, a_list = np.array_split(x[:-1], 3)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a = self.map(k_list, H_list, a_list)
    k = k0.copy()
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    while error > tolerance:
        k_last = k.copy()
        self.mesh.diffusivity[:] = k
        T = self.linear_solve(rhs=rhs) # solve
        k = hofmeister1999(k0, T, a)
        error = np.absolute(k - k_last).max()
        i += 1
        
    q = self.heatflux(T, k)
    delT = self.gradient(T)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, q0=q0) # priors
    return cost


def adjoint_model(x, self, bc='Z'):
    """
    N-dimensional nonlinear model with flux lower BC
    Implements Hofmeister 1999 temperature-dependent
    conductivity law

    Arguments
    ---------
     x : [k_list, H_list, a_list, q0]

    Returns
    -------
     cost : scalar
     grad : [dk_list, dH_list, da_list, dq0]
    """
    def hofmeister1999(k0, T, a=0.25, c=0.0):
        return k0*(298.0/T)**a + c*T**3
    k_list, H_list, a_list = np.array_split(x[:-1], 3)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a = self.map(k_list, H_list, a_list)
    k = [k0.copy()]
    T = [None]
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    self.mesh.temperature._gdata.set(0.)
    while error > tolerance:
        self.mesh.diffusivity[:] = k[i]
        # solve
        Ti = self.linear_solve(rhs=rhs)
        ki = hofmeister1999(k0, Ti, a)
        T.append(Ti.copy())
        k.append(ki.copy())
        error = np.absolute(k[-1] - k[-2]).max()
        i += 1

    q = self.heatflux(T[-1], k[-1])
    delT = self.gradient(T[-1])
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T[-1], delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, q0=q0) # priors
    
    ## AD ##
    dk = np.zeros_like(H)
    dH = np.zeros_like(H)
    dT = np.zeros_like(H)
    da = np.zeros_like(H)
    dk0 = np.zeros_like(H)
    dq0 = np.array(0.0)
    
    # priors
    dcdk_list = self.objective_routine_ad(k=k_list)
    dcdH_list = self.objective_routine_ad(H=H_list)
    dcda_list = self.objective_routine_ad(a=a_list)
    dcdq0 = self.objective_routine_ad(q0=q0)
    # observations
    dT += self.objective_routine_ad(T=T[-1])

    dq = np.zeros_like(q)
    dq[0] = self.objective_routine_ad(q=q[0])
    
    ddelT = np.zeros_like(delT)
    ddelT[0] = self.objective_routine_ad(delT=delT[0])
    

    dTd = self.gradient_ad(ddelT, T[-1])
    dT += dTd
    
    dTq, dkq = self.heatflux_ad(dq, q, T[-1], k[-1])
    dT += dTq
    dk += dkq
    

    # solve
    for j in range(i):
        dkda = np.log(298.0/T[-1-j])*k0*(298.0/T[-1-j])**a
        dkdk0 = (298.0/T[-1-j])**a
        dkdT = -a*k0/T[-1-j]*(298.0/T[-1-j])**a
        
        dk0 += dkdk0*dk
        dT  += dkdT*dk
        da  += dkda*dk
        
        dk.fill(0.0)
        
        self.mesh.diffusivity[:] = k[-1-j]
        dA, db = self.linear_solve_ad(T[-1-j], dT)
        dk += dA
        dH += -db
        dz = self.grid_delta[-1]
        lowerBC_mask = self.mesh.bc["min"+bc]["mask"]
        dq0_local = np.sum(-db[lowerBC_mask]/dz/inv.ghost_weights[lowerBC_mask])
        dq0_global = np.array(0.0)
        comm.Allreduce([dq0_local, MPI.DOUBLE], [dq0_global, MPI.DOUBLE], op=MPI.SUM)
        dq0 += dq0_global
        
        dT.fill(0.0)
        
    dk0 += dk
        
    # pack to lists
    dk_list, dH_list, da_list = inv.map_ad(dk0, dH, da)
    dk_list += dcdk_list
    dH_list += dcdH_list
    da_list += dcda_list
    dq0 += dcdq0
    
    dx = np.hstack([dk_list, dH_list, da_list, [dq0]])
    
    return cost, dx





minX, maxX = 0.0, 1000.0
minY, maxY = 0.0, 1000.0
minZ, maxZ = -35e3, 1000.0
nx, ny, nz = 10, 10, 10
n = nx*ny*nz

mesh = ConductionND((minX, minY, minZ), (maxX, maxY, maxZ), (nx,ny,nz))

# BCs
mesh.boundary_condition('maxZ', 298.0, flux=False)
mesh.boundary_condition('minZ', 1e3, flux=True)


# In[3]: Global lithology

lithology = np.zeros((nz,ny,nx), dtype='int32')
lithology[:,3:7,:] = 1
lithology[:,7:,:]  = 2

# Need to slice this bad boy up: Local lithology

(minI, maxI), (minJ, maxJ), (minK, maxK) = mesh.dm.getGhostRanges()
lithology = lithology[minK:maxK, minJ:maxJ, minI:maxI]


inv = InversionND(lithology.flatten(), mesh)

k = np.array([3.5, 2.0, 3.2])
H = np.array([0.5e-6, 1e-6, 2e-6])
a = np.array([0.3, 0.3, 0.3])
q0 = 35e-3
sigma_q0 = 5e-3

# Inversion variables
x = np.hstack([k, H, a, [q0]])
dx = x*0.01

# Priors
k_prior = k*1.1
H_prior = H*1.1
a_prior = a*1.1
sigma_k = k*0.1
sigma_H = H*0.1
sigma_a = a*0.1

kp = InvPrior(k_prior, sigma_k)
Hp = InvPrior(H_prior, sigma_H)
ap = InvPrior(a_prior, sigma_a)
q0p = InvPrior(q0, sigma_q0)
inv.add_prior(k=ap, H=Hp, a=ap, q0=q0p)


fm0 = forward_model(x, inv)
fm1 = forward_model(x+dx, inv)
ad  = adjoint_model(x, inv)

print "\n--- LITHOLOGY PRIORS ---"
print "finite difference", (fm1 - fm0)
print "adjoint", ad[1].dot(dx)




inv = InversionND(lithology.flatten(), mesh)

np.random.seed(0)

q_obs = np.ones(5)*0.03
sigma_q = q_obs*0.5
q_coord = np.zeros((5,3))
q_coord[:,0] = np.random.random(5)*1e3
q_coord[:,1] = np.random.random(5)*1e3
q_coord[:,2] = 0.0
q_coord = q_coord

qobs = InvObservation(q_obs, sigma_q, q_coord)
inv.add_observation(q=qobs)


fm0 = forward_model(x, inv)
fm1 = forward_model(x+dx, inv)
ad  = adjoint_model(x, inv)

print "\n--- HEAT FLOW OBSERVATIONS ---"
print "finite difference", (fm1 - fm0)
print "adjoint", ad[1].dot(dx)




inv = InversionND(lithology.flatten(), mesh)

T_prior = np.ones(mesh.nn)*400.0
sigma_T = T_prior*0.01

Tobs = InvObservation(T_prior, sigma_T, mesh.coords)
inv.add_observation(T=Tobs)


fm0 = forward_model(x, inv)
fm1 = forward_model(x+dx, inv)
ad  = adjoint_model(x, inv)

print "\n--- TEMPERATURE PRIOR ---"
print "finite difference", (fm1 - fm0)
print "adjoint", ad[1].dot(dx), #"\n", gradient.array


# ## TAO solve
# 
# Solve a minimisation problem. TAO provides a number of minimisation schemes,
# some which require the gradient and Hessian.

# inv_x = x.copy()

# lower_bound = x.duplicate()
# lower_bound.array[:3] = 0.5 # conductivity
# lower_bound.array[-1] = 5e-3 # q0

# upper_bound = x.duplicate()
# upper_bound.array[:3] = 4.5 #conductivity
# upper_bound.array[3:6] = 5e-6 # heat production
# upper_bound.array[6:9] = 1.0 # a
# upper_bound.array[-1] = 45e-3 # q0


# tao = PETSc.TAO().create(comm)
# tao.setType('blmvm')
# tao.setVariableBounds(lower_bound, upper_bound)
# tao.setObjectiveGradient(adjoint_model)
# tao.setFromOptions()

# t = clock()
# tao.solve(inv_x)
# print "\nCompleted in", clock()-t
# print tao.getConvergedReason(), tao.getIterationNumber()
# print x.array