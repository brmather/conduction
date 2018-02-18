
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


# In[2]:

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
H = np.array([0.0, 1e-6, 2e-6])
a = np.array([0.3, 0.3, 0.3])
q0 = 35e-3

k = np.array([3.5, 2.0, 3.2])
H = np.array([0.0, 1e-6, 2e-6])
a = np.array([0.3, 0.3, 0.3])
q0 = 35e-3

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

fm0 = inv.forward_model(x)
fm1 = inv.forward_model(x+dx)
tl  = inv.tangent_linear(x, dx)
ad  = inv.adjoint(None, x, gradient)

if comm.rank == 0:
    print "\n--- LITHOLOGY PRIORS ---"
    print "finite difference", (fm1 - fm0)
    print "tangent linear", tl[1]
    print "adjoint", gradient.array.dot(dx)




inv = Inversion(lithology.flatten(), mesh)

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

## Replace with coordinates on the grid
# minX, maxX = mesh.coords[:,0].min(), mesh.coords[:,0].max()
# minY, maxY = mesh.coords[:,1].min(), mesh.coords[:,1].max()
# minZ, maxZ = mesh.coords[:,2].min(), mesh.coords[:,2].max()
# Xcoords = np.linspace(minX, maxX, mesh.nx)
# Ycoords = np.linspace(minY, maxY, mesh.ny)
# Zcoords = np.linspace(minZ, maxZ, mesh.nz)

# q_coord[:,0] = Xcoords[:5]
# q_coord[:,2] = Zcoords[:5]

###


# print "{}\n{}\n{} - {}".format(comm.rank, q_coord, inv.mesh.coords.min(axis=0), inv.mesh.coords.max(axis=0))

inv.add_observation(q=(q_obs, sigma_q, q_coord[:,::-1]))
# for o in inv.observation['q']:
    # print comm.rank, o

dx = x*0.001
# dx.array[9] = 0.


fm0 = inv.forward_model(x)
fm1 = inv.forward_model(x+dx)
tl  = inv.tangent_linear(x,dx)
ad  = inv.adjoint(None, x, gradient)

if comm.rank == 0:
    print "\n--- HEAT FLOW OBSERVATIONS ---"
    print "finite difference", (fm1 - fm0)
    print "tangent linear", tl[1]
    print "adjoint", gradient.array.dot(dx)




inv = Inversion(lithology.flatten(), mesh)

T_prior = np.ones(mesh.nn)*400.0
sigma_T = T_prior*0.01

inv.add_observation(T=(T_prior,sigma_T,None))


dx = x*0.01
# dx.array[9] = 0.

fm0 = inv.forward_model(x)
fm1 = inv.forward_model(x+dx)
tl  = inv.tangent_linear(x,dx)
ad  = inv.adjoint(None, x, gradient)

if comm.rank == 0:
    print "\n--- TEMPERATURE PRIOR ---"
    print "finite difference", (fm1 - fm0)
    print "tangent linear", tl[1]
    print "adjoint", gradient.array.dot(dx), #"\n", gradient.array


# ## TAO solve
# 
# Solve a minimisation problem. TAO provides a number of minimisation schemes,
# some which require the gradient and Hessian.

inv_x = x.copy()

lower_bound = x.duplicate()
lower_bound.array[:3] = 0.5 # conductivity
lower_bound.array[-1] = 5e-3 # q0

upper_bound = x.duplicate()
upper_bound.array[:3] = 4.5 #conductivity
upper_bound.array[3:6] = 5e-6 # heat production
upper_bound.array[6:9] = 1.0 # a
upper_bound.array[-1] = 45e-3 # q0


tao = PETSc.TAO().create(comm)
tao.setType('blmvm')
tao.setVariableBounds(lower_bound, upper_bound)
tao.setObjectiveGradient(inv.adjoint)
tao.setFromOptions()


t = clock()
tao.solve(inv_x)
print "\nCompleted in", clock()-t
print tao.getConvergedReason(), tao.getIterationNumber()
print x.array