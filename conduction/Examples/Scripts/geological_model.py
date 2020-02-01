# coding: utf-8

## Geological model
# Assemble the geological model from xyz data for each surface.

import numpy as np
import argparse
import conduction
from time import time
import os

from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Process some model arguments.')
parser.add_argument('--nonlinear', required=False, action='store_true', default=False, help='Nonlinear solver')
parser.add_argument('--gravity', required=False, action='store_true', default=False, help='Gravity solver')
parser.add_argument('echo', type=str, metavar='PATH', help='Input folder location')
args = parser.parse_args()


try: range=xrange
except: pass


def initialise_ksp(matrix, solver='bcgs'):
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(solver)
    ksp.setOperators(matrix)
    # pc = ksp.getPC()
    # pc.setType('gamg')
    ksp.setTolerances(1e-10, 1e-50)
    ksp.setFromOptions()
    return ksp

def solve(self, ksp, matrix=None, rhs=None):
    """
    Construct the matrix A and vector b in Ax = b
    and solve for x

    GMRES method is default
    """
    if type(matrix) == type(None):
        matrix = self.construct_matrix()
    if type(rhs) is type(None):
        rhs = self.construct_rhs()
    res = self.temperature
    ksp.solve(rhs._gdata, res._gdata)
    # We should hand this back to local vectors
    return res[:]

def hofmeister1999(k0, T):
    a = 2.85e-5
    b = 1.008e-8
    c = -0.384
    d = 0.0

    k_con  = k0*(298.0/T)**0.45
    k_rad  = 0.0175 - 1.0374e-4*T + 2.245*T**2/1e7 - 3.407*T**3/1e11
    k_pres = 1.0 + (K_0p*g_av*3350.0*np.abs(mesh.coords[:,2])*1e-9)/K_T
    k_exp  = np.exp(-(a*(T - 298.0) + b*0.5*(T**2 - 88804.0) + \
                     c*(3.3557e-3 - 1.0/T) + d*(T**5 - 2.35e12)/5.0)* \
                     (gmma_T*4 + 1.0/3))
    k_new = k_con*k_exp*k_pres + k_rad
    k_new[grun_mask] = k0[grun_mask]
    k_new[kmask] = k0[kmask]
    return k_new

def nonlinear_conductivity(self, k0, tolerance):
    rhs = self.rhs
    k = k0.copy()
    self.diffusivity[:] = k
    mat = self.construct_matrix()
    ksp = initialise_ksp(mat)

    error = np.array(10.0)
    i = 0

    while (error > tolerance):
        t = time()
        k_last = self.diffusivity[:].copy()

        mat = self.construct_matrix()
        T = solve(self, ksp, matrix=mat, rhs=rhs)
        k = hofmeister1999(k0, T)
        self.diffusivity[:] = k

        err = np.absolute(k - k_last).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)
        i += 1
        if comm.rank == 0:
            print("iteration {} in {:.3f} seconds, residual = {:.2e}".format(i, time()-t, float(error)))

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

def map_properties(self, lithology, lithology_index, *args):
    """
    Requires a tuple of vectors corresponding to an inversion variable
    these are mapped to the mesh.

    tuple(vec1, vec2, vecN) --> tuple(field1, field2, fieldN)
    """
    nf = len(args)
    nl = len(lithology_index)

    # preallocate memory
    mesh_variables = np.zeros((nf, lithology.size))

    # unpack vector to field
    for i in range(nl):
        idx = lithology == lithology_index[i]
        for f in range(nf):
            mesh_variables[f,idx] = args[f][i]

    # create MeshVariable to sync fields across processors
    var = self.create_meshVariable("dummy")
    for f in range(nf):
        var[:] = mesh_variables[f]
        mesh_variables[f] = var[:].copy()

    return list(mesh_variables)


directory = args.echo
layer_attributes = np.loadtxt(directory+'layers.info', skiprows=1, usecols=list(range(2,11)))
layer_number = np.loadtxt(directory+'layers.info', dtype=int, skiprows=1, usecols=(0,))
layer_name   = np.loadtxt(directory+'layers.info', dtype=str, skiprows=1, usecols=(1,))

layer_header = ['body number', 'density', 'alpha', 'thermal conductivity', 'heat production rate',\
                'pressure coefficient', 'Gruneisen parameter', 'pressure derivative of bulk modulus', 'man']


# count layers
nl = 0
for layer in os.listdir(directory+'layers_xy'):
    if layer.endswith('.xyz'):
        nl += 1

spl = dict()
for l in range(nl):
    data = np.loadtxt(directory+'layers_xy/layer{}.xyz'.format(l))
    xl = data[:,0]
    yl = data[:,1]

    Xcoords = np.unique(xl)
    Ycoords = np.unique(yl)
    nx, ny = Xcoords.size, Ycoords.size

    zl = data[:,2].reshape(ny,nx)
    spl[l] = RectBivariateSpline(Ycoords, Xcoords, zl)

# Overwrite model extents
minX, maxX = 350000.0, 788000.0
minY, maxY = 480000.0, 1000000.0
minZ, maxZ = -130e3, 600.0

if comm.rank == 0:
    print("min/max:\n x {}\n y {}\n z {}".format((minX, maxX),
                                                 (minY, maxY),
                                                 (minZ, maxZ)))

## Setup the hexahedral mesh

Nx, Ny, Nz = 300, 300, 600

mesh = conduction.ConductionND((minX, minY, minZ), (maxX, maxY, maxZ), (Nx, Ny, Nz))

coords = mesh.coords
Xcoords, Ycoords, Zcoords = mesh.grid_coords
nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size



## Fill the volume between each surface

xq, yq = np.meshgrid(Xcoords, Ycoords)
xq = xq.ravel()
yq = yq.ravel()

horizontal_slice = np.column_stack([xq, yq])


# create layer voxel
layer_voxel = np.zeros((nz, ny, nx), dtype=np.int8)
layer_mask = np.zeros(nx*ny*nz, dtype=bool)

tree = cKDTree(coords)
layer_voxel.fill(-1)

for l in range(nl):
    i0, j0, k0 = query_nearest(l)
    for i in range(i0.size):
        layer_voxel[:i0[i], j0[i], k0[i]] = l+1
    if comm.rank == 0:
        print("mapped layer {}".format(l))


# This makes it the same as xyz files
# layer_voxel = np.rot90(layer_voxel, axes=(1,2))
# layer_voxel = layer_voxel[:,::-1,:]


# map properties
rho, alpha, k0, H, beta, gmma_T, K_0p, K_T, man = map_properties(mesh, layer_voxel.ravel(), layer_number,\
                                                                 *layer_attributes.T)
grun_mask = gmma_T == 0
K_T[K_T == 0] = 1e99


# Update properties
mesh.update_properties(k0, H)

# gravity constants
g_s = 9.81
g_400 = 9.97
depth = abs(minZ)
d_gz = (g_400 - g_s)/400e3
g_av = g_s + d_gz*depth*0.5 # Average gravity attraction for the thermal calculation

kmask = k0 == 0.0

# we differentiate air and asthenosphere by 20km depth
air_mask = np.logical_and(kmask, mesh.coords[:,2]>-20e3)
lab_mask = np.logical_and(kmask, mesh.coords[:,2]<-20e3)



# Boundary conditions
topBC = 298.0
bottomBC = 1300.0

mesh.boundary_condition('maxZ', topBC, flux=False)
mesh.boundary_condition('minZ', bottomBC, flux=False)


air_idx = np.nonzero(air_mask)[0].astype(np.int32)
lab_idx = np.nonzero(lab_mask)[0].astype(np.int32)


# Manually overwrite Dirichlet BCs
rhs = mesh.construct_rhs()
mesh.dirichlet_mask[air_idx] = True
mesh.dirichlet_mask[lab_idx] = True

rhs[air_idx] = np.ones(air_idx.size)*topBC
rhs[lab_idx] = np.ones(lab_idx.size)*bottomBC


# initial guess x0
mesh.temperature[:] = rhs[:].copy()


if not args.nonlinear:
    t = time()
    mat = mesh.construct_matrix()
    ksp = initialise_ksp(mat)
    sol = solve(mesh, ksp, matrix=mat, rhs=rhs)
    if comm.rank == 0:
        print("linear solve time {} s".format(time() -t))
    H5_file = 'geological_model.h5'

else:
    ## Nonlinear conductivity
    k0 = mesh.diffusivity[:].copy()
    nonlinear_conductivity(mesh, k0, 1e-5)
    H5_file = 'geological_model_nonlinear.h5'

# Calculate heat flow
qz, qy, qx = mesh.heatflux()

# Save to H5 file
mesh.save_mesh_to_hdf5(H5_file)
mesh.save_field_to_hdf5(H5_file,\
                        layer_ID=layer_voxel.ravel(),\
                        conductivity=mesh.diffusivity[:],\
                        heat_production=mesh.heat_sources[:]*1e6,\
                        temperature=mesh.temperature[:]-273.14,\
                        heat_flux=(qx+qy+qz)*1e3)
mesh.save_vector_to_hdf5(H5_file,\
                         heat_flux_vector=(qx*1e3,qy*1e3,qz*1e3),\
                         heat_flux_vector_lateral=(qx*1e3,qy*1e3,np.zeros_like(qz)))


if args.gravity:
    gamma = 6.67408e-11

    t = time()
    mesh.mat.destroy() # save some memory
    phi = np.ones(mesh.nn)
    rvec = -rho*gamma*4.0*np.pi
    mesh.update_properties(phi, rvec)

    for boundary in mesh.bc.keys():
        mesh.boundary_condition(boundary, 0.0, flux=False)

    # mesh.dirichlet_mask[air_idx] = True
    # mesh.dirichlet_mask[lab_idx] = True
    rhs[air_idx] = np.zeros(air_idx.size)
    rhs[lab_idx] = np.zeros(lab_idx.size)

    mat = mesh.construct_matrix(in_place=False)
    rhs = mesh.construct_rhs()
    ksp = initialise_ksp(mat, solver='gmres')
    sol = solve(mesh, ksp, matrix=mat, rhs=rhs)
    if comm.rank == 0:
        print("gravity solve time {} s".format(time() -t))

    # calculate gradient
    gz, gy, gx = mesh.gradient(sol)

    # Save to H5 file
    mesh.save_field_to_hdf5(H5_file, gravity=gz)


if comm.rank == 0:
    conduction.generateXdmf(H5_file)
