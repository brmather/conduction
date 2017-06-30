import numpy as np
import matplotlib.pyplot as plt

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD
from conduction import Conduction3D
from time import clock

try: range=xrange
except: pass

nsim = 5


def timings(func, *args):
    t = clock()

    for i in range(0,nsim):
        func(*args)

    ttime = (clock() - t)/nsim
    if comm.rank == 0:
        print("{} took {}s".format(func.__name__, ttime))




minX, maxX = 0.0, 1.0
minY, maxY = 1.0, 2.0
minZ, maxZ = 3.0, 4.0
nx, ny, nz = 100, 100, 100

ode = Conduction3D((minX, minY, minZ), (maxX, maxY, maxZ), (nx,ny,nz))


conductivity = np.ones(ode.nx*ode.ny*ode.nz)
heat_sources = np.ones_like(conductivity)
ode.update_properties(conductivity, heat_sources)


ode.boundary_condition('maxY', 0.0, False)
ode.boundary_condition('minY', 1.0, True)


mat = ode.construct_matrix()
rhs = ode.construct_rhs()

res = ode.dm.createGlobalVector()

# setup solver
ksp = PETSc.KSP().create()
ksp.setType('cgs')
ksp.setTolerances(rtol=1e-10, atol=1e-50)
ksp.setOperators(mat)
ksp.setFromOptions()


ksp.solve(rhs, res)



### Now benchmark!

timings(ode.__init__, (minX, minY, minZ), (maxX, maxY, maxZ), (nx,ny,nz))
timings(ode.update_properties, conductivity, heat_sources)
timings(ode.boundary_condition, 'maxY', 0.0, False)
timings(ode.boundary_condition, 'minY', 1.0, True)
timings(ode.construct_matrix)
timings(ode.construct_rhs)
timings(ksp.solve, rhs, res)