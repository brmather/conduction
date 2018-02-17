import numpy as np
import matplotlib.pyplot as plt

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD
from conduction import ConductionND


minX, maxX = 0.0, 1.0
minY, maxY = 1.0, 2.0
minZ, maxZ = 3.0, 4.0
nx, ny, nz = 10, 10, 10

ode = ConductionND((minX, minY, minZ), (maxX, maxY, maxZ), (nx,ny,nz))


indices = ode.dm.getLGMap().indices
# ao = ode.dm.getAO()
# indices = ao.petsc2app(indices)
# print comm.rank, indices.reshape(ode.nz,ode.ny,ode.nx)


conductivity = np.ones(ode.nn)
heat_sources = np.ones_like(conductivity) #* 1e1
ode.update_properties(conductivity, heat_sources)


ode.boundary_condition('maxY', 0.0, False)
ode.boundary_condition('minY', 1.0, True)

# print comm.rank, ode.dirichlet_mask.reshape(ode.nz, ode.ny, ode.nx)
# ode.lvec.set(1.0)
# ode.gvec.set(0.0)
# ode.dm.localToGlobal(ode.lvec, ode.gvec, addv=True)
# ode.dm.globalToLocal(ode.gvec, ode.lvec)
# print comm.rank, ode.lvec.array.astype(int).reshape(ode.nz,ode.ny,ode.nx)

print comm.rank, ode.dm.getLocalBoundingBox()
print comm.rank, ode.coords.min(axis=0), ode.coords.max(axis=0)

mat = ode.construct_matrix()
rhs = ode.construct_rhs()

# print comm.rank, ode.lvec.array.reshape(ode.nz,ode.ny,ode.nx)

# mat.view()
# print mat.getSizes()

# print comm.rank, "\n",ode.dirichlet_mask, "\n",ode.ghost_mask

ksp = PETSc.KSP().create()
ksp.setType('gmres')
ksp.setTolerances(rtol=1e-10, atol=1e-50)
ksp.setOperators(mat)
ksp.setFromOptions()

res = ode.dm.createGlobalVector()

ksp.solve(rhs._gdata, res)

tozero, zvec = PETSc.Scatter.toZero(rhs._gdata)

U = ode.dm.createNaturalVector()
ode.dm.globalToNatural(res, U)
tozero.scatter(U, zvec)
T = zvec.array.copy()
ode.lvec.setArray(ode.dirichlet_mask)
ode.dm.localToGlobal(ode.lvec, res)
tozero.scatter(rhs._gdata, zvec)
M = zvec.array.copy().astype(bool)


if comm.rank == 0:

    T = T.reshape(nz,ny,nx)
    gradTx, gradTy, gradTz = np.gradient(T)
    gradT = gradTx + gradTy + gradTz
    # np.savez('conduction3D_benchmark_gradT_{}proc'.format(comm.rank), gradT=gradT)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(T[1,:,:], interpolation='bilinear')
    fig.colorbar(im1)
    plt.show()

    # print T

# ode.mat.view()