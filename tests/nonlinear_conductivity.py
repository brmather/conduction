
# coding: utf-8

# # Nonlinear conductivity
# 
# Conductivity should depend on temperature. We use Hofmeister's 1999 law to calculate the effective conductivity, $k(T)$.

# In[1]:


import numpy as np
import conduction
from conduction import Conduction3D
from time import clock
from mpi4py import MPI
comm = MPI.COMM_WORLD


# In[2]:


minX, maxX = 0.0, 1.0
minY, maxY = 0.0, 1.0
minZ, maxZ = 0.0, 1.0
nx, ny, nz = 30, 30, 30

ode = Conduction3D((minX, minY, minZ), (maxX, maxY, maxZ), (nx,ny,nz))

n = ode.nx*ode.ny*ode.nz


# In[3]:


k0 = np.ones(n)
k  = np.ones(n)
H  = np.zeros(n)
a  = np.ones(n)

# BCs
ode.boundary_condition('maxZ', 298.0, flux=False)
ode.boundary_condition('minZ', 1e3, flux=True)


# In[4]:


H5_file = 'nonlinear-conductivity.h5'

ode.save_mesh_to_hdf5(H5_file)


# In[5]:
ode.update_properties(k0, H)


def hofmeister1999(k0, T, a):
    return k0*(298.0/T)**a

def nonlinear_conductivity(self, k0, tolerance, k_fn, *args):
    k = k0.copy()
    self.update_properties(k, self.heat_sources)

    error = np.array(10.0)
    i = 0
    t = clock()

    while error > tolerance:
        k_last = self.diffusivity.copy()
        self.update_properties(k, self.heat_sources)

        T = self.solve()
        k = k_fn(*args)

        err = np.absolute(k - k_last).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)
        i += 1

    print "{} iterations in {} seconds".format(i, clock()-t)


nonlinear_conductivity(ode, k0, 1e-10, hofmeister1999, k0, ode.temperature, a)


ode.save_field_to_hdf5(H5_file, T=ode.temperature, k=ode.diffusivity)

if comm.rank == 0:
    conduction.tools.generateXdmf(H5_file)