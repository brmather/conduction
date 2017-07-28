
# coding: utf-8

# # Geological model
# 
# Assemble the geological model from xyz data for each surface.

nonlinear = False


import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, SmoothBivariateSpline, RectBivariateSpline
import conduction
from time import clock

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD



directory = '/opt/ben/'

layer_attributes = np.loadtxt(directory+'layers.info', skiprows=1, usecols=(2,3,4,5,6,7,8,9,10))
layer_number = np.loadtxt(directory+'layers.info', dtype=int, skiprows=1, usecols=(0,))
layer_name   = np.loadtxt(directory+'layers.info', dtype=str, skiprows=1, usecols=(1,))

layer_header = ['body number', 'density', 'alpha', 'thermal conductivity', 'heat production rate',\
                'pressure coefficient', 'Gruneisen parameter', 'pressure derivative of bulk modulus', 'man']


layer = dict()
for i in xrange(0, 10):
    data = np.loadtxt(directory+'layers_xy/layer{}.xyz'.format(i))
    layer[i] = data


Xcoords = np.unique(data[:,0])
Ycoords = np.unique(data[:,1])

nx, ny = Xcoords.size, Ycoords.size


minZ = -130e3
maxZ = 600.0

### REFINE GRANITES
minX, minY = 475e3, 775e3
maxX, maxY = 650e3, 960e3


if comm.rank == 0:
    print("min/max:\n x {}\n y {}\n z {}".format((minX, maxX),
                                                 (minY, maxY),
                                                 (minZ, maxZ)))


spl = dict()

for i in xrange(10):
    data = layer[i]
    xl = data[:,0]
    yl = data[:,1]
    zl = data[:,2].reshape(ny,nx)
    spl[i] = RectBivariateSpline(Ycoords, Xcoords, zl)


## Setup the hexahedral mesh

Nx, Ny, Nz = 50, 50, 200

mesh = conduction.Conduction3D((minX, minY, minZ), (maxX, maxY, maxZ), (Nx, Ny, Nz))

coords = mesh.coords

Xcoords = np.unique(coords[:,0])
Ycoords = np.unique(coords[:,1])
Zcoords = np.unique(coords[:,2])

nx, ny, nz = mesh.nx, mesh.ny, mesh.nz



## Fill the volume between each surface

xq, yq = np.meshgrid(Xcoords, Ycoords)
xq = xq.ravel()
yq = yq.ravel()

horizontal_slice = np.column_stack([xq, yq])


# create layer voxel
layer_voxel = np.zeros((nz, ny, nx), dtype=np.int8)
layer_mask = np.zeros(nx*ny*nz, dtype=bool)

# create KDTree
tree = cKDTree(coords)


def query_nearest(l):
    """
    Need to be careful in parallel -
     can't have a processor filling in a region off-processor
     (although it appears the shadow zones save us here)

    """
    layer_mask.fill(0)
    
    zq = spl[l].ev(yq, xq)
    d, idx = tree.query(np.column_stack([xq, yq, zq]))
    layer_mask[idx] = True
    
    return np.where(layer_mask.reshape(nz,ny,nx))

layer_voxel.fill(-1)


for l in xrange(0,10):
    i0, j0, k0 = query_nearest(l)

    for i in xrange(i0.size):
        layer_voxel[:i0[i], j0[i], k0[i]] = l+1

    if comm.rank == 0:
        print("mapped layer {}".format(l))


# This makes it the same as xyz files
# layer_voxel = np.rot90(layer_voxel, axes=(1,2))
# layer_voxel = layer_voxel[:,::-1,:]


# Now map properties to these layers. Where these layers are not defined we have a default value assigned to them.

k = np.ones_like(layer_voxel, dtype=np.float32)*3
H = np.zeros_like(layer_voxel, dtype=np.float32)

for i, l in enumerate(layer_number):
    name = layer_name[i]
    mask = layer_voxel == l
    ki = layer_attributes[i,2]
    Hi = layer_attributes[i,3]
    k[mask] = ki
    H[mask] = Hi
    if comm.rank == 0:
        print('{} {} \t k = {}, H = {}'.format(l, name, ki, Hi))
    

k = k.ravel()
H = H.ravel()

# Update properties
mesh.update_properties(k, H)


# Boundary conditions
topBC = 298.0
bottomBC = 1300.0

mesh.boundary_condition('maxZ', topBC, flux=False)
mesh.boundary_condition('minZ', bottomBC, flux=False)


# Make the top and bottom BC conform to geometry
layer0 = layer_voxel == 0
layer9 = layer_voxel > 8

layer0 = layer0.ravel()
layer9 = layer9.ravel()

# mesh.dirichlet_mask[layer0] = True
mesh.dirichlet_mask[layer9] = True

mesh.construct_matrix()
mesh.construct_rhs()

layer0_gidx = mesh.lgmap.apply(np.nonzero(layer0)[0].astype(np.int32))
layer9_gidx = mesh.lgmap.apply(np.nonzero(layer9)[0].astype(np.int32))

mesh.rhs.assemblyBegin()
# mesh.rhs.setValues(layer0_gidx, np.ones(layer0_gidx.size)*topBC)
mesh.rhs.setValues(layer9_gidx, np.ones(layer9_gidx.size)*bottomBC)
mesh.rhs.assemblyEnd()



def solve(self, solver='bcgs'):
    """
    Construct the matrix A and vector b in Ax = b
    and solve for x

    GMRES method is default
    """
    matrix = self.mat
    rhs = self.rhs
    res = self.res
    lres = self.lres

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(solver)
    ksp.setOperators(matrix)
    # pc = ksp.getPC()
    # pc.setType('gamg')
    ksp.setFromOptions()
    ksp.setTolerances(1e-10, 1e-50)
    ksp.solve(rhs, res)
    # We should hand this back to local vectors
    self.dm.globalToLocal(res, lres)
    return lres.array


def hofmeister1999(k0, T, a, c):
    return k0*(298.0/T)**a + c*T**3

def nonlinear_conductivity(self, k0, tolerance, k_fn, *args):
    k = k0.copy()
    self.update_properties(k, self.heat_sources)
    self.construct_matrix()

    error = np.array(10.0)
    i = 0
    t = clock()

    while (error > tolerance):
        k_last = self.diffusivity.copy()
        self.update_properties(k, self.heat_sources)
        self.construct_matrix()

        T = solve(self)
        k = k_fn(*args)

        err = np.absolute(k - k_last).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)
        i += 1

        if comm.rank == 0:
            print("{} iterations in {} seconds, residual = {}".format(i, clock()-t, error))



if not nonlinear:
    t = clock()
    sol = solve(mesh)
    if comm.rank == 0:
        print("linear solve time {} s".format(clock() -t))

    H5_file = 'geological_model_granites.h5'


else:
    ## Nonlinear conductivity

    a = 0.33
    c = 1e-10
    k0 = mesh.diffusivity.copy()

    nonlinear_conductivity(mesh, k0, 1e-5, hofmeister1999, k0, mesh.temperature, a, c)

    H5_file = 'geological_model_granites_nonlinear.h5'


# Calculate heat flow
qx, qy, qz = mesh.heatflux()


# Save to H5 file
mesh.save_mesh_to_hdf5(H5_file)
mesh.save_field_to_hdf5(H5_file,\
                        layer_ID=layer_voxel.ravel(),\
                        conductivity=mesh.diffusivity,\
                        heat_production=mesh.heat_sources*1e6,\
                        temperature=mesh.temperature-273.14,\
                        heat_flux=(qx+qy+qz)*1e3)
mesh.save_vector_to_hdf5(H5_file,\
                         heat_flux_vector=(qx*1e3,qy*1e3,qz*1e3),\
                         heat_flux_vector_lateral=(qx*1e3,qy*1e3,np.zeros_like(qz)))
if comm.rank == 0:
    conduction.generateXdmf(H5_file)
