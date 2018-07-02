# coding: utf-8

## Geological model
# Assemble the geological model from xyz data for each surface.

import numpy as np
import argparse
from time import time
import os

from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

import conduction
from conduction.inversion import InvObservation, InvPrior
from conduction.inversion import create_covariance_matrix
from conduction.inversion import gaussian_function

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Process some model arguments.')
parser.add_argument('-x', type=str, metavar='PATH', help='Path to minimiser results')
parser.add_argument('echo', type=str, metavar='PATH', help='Output file path')
parser.add_argument('--gravity', required=False, action='store_true', default=False, help='Gravity solver')
args = parser.parse_args()

try: range=xrange
except: pass


def hofmeister1999(k0, T):
    a = 2.85e-5
    b = 1.008e-8
    c = -0.384
    d = 0.0

    k_con  = k0*(298.0/T)**0.45
    k_rad  = 0.0175 - 1.0374e-4*T + 2.245e-7*T**2 - 3.407e-11*T**3
    k_pres = 1.0 + (K_0p*g_av*3350.0*np.abs(mesh.coords[:,2])*1e-9)/K_T
    k_exp  = np.exp(-(a*(T - 298.0) + b*0.5*(T**2 - 88804.0) + \
                     c*(3.3557e-3 - 1.0/T) + d*(T**5 - 2.35e12)/5.0)* \
                     (gmma_T*4 + 1.0/3))
    k_new = k_con*k_exp*k_pres + k_rad
    k_new[grun_mask] = k0[grun_mask]
    k_new[kmask] = k0[kmask]
    return k_new

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

def forward_model(x, self, bc='Z'):
    """
    N-dimensional nonlinear model with flux lower BC
    Implements Hofmeister 1999 temperature-dependent
    conductivity law

    Arguments
    ---------
     x : [k_list, H_list, Tb]

    Returns
    -------
     cost : scalar

    """
    k_list, H_list = np.array_split(x[:-1], 2)
    Tb = x[-1]
    
    # map to mesh
    k0, H = self.map(k_list, H_list)
    k = k0.copy()
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, Tb, flux=False)
    rhs = self.mesh.construct_rhs()
    
    error = np.array(10.0)
    tolerance = 1e-5
    i = 0
    while error > tolerance:
        k_last = k.copy()
        self.mesh.diffusivity[:] = k
        T = self.linear_solve(rhs=rhs) # solve
        k = hofmeister1999(k0, T)
        err = np.absolute(k - k_last).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)
        i += 1
    
    q = self.heatflux(T, k)
    delT = self.gradient(T)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, Tb=Tb) # priors
    return cost

directory = '../data/Ireland_model/'
layer_attributes = np.loadtxt(directory+'layers.info', skiprows=1, usecols=list(range(2,11)))
layer_number = np.loadtxt(directory+'layers.info', dtype=int, skiprows=1, usecols=(0,))
layer_name   = np.loadtxt(directory+'layers.info', dtype=str, skiprows=1, usecols=(1,))

layer_header = ['body number', 'density', 'alpha', 'thermal conductivity', 'heat production rate',\
                'pressure coefficient', 'Gruneisen parameter', 'pressure derivative of bulk modulus', 'man']


# load minimiser_results.npz file
with np.load(args.x, 'r') as xNfile:
    xN = np.hstack([xNfile['k'], xNfile['H'], [xNfile['Tb']]])


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
    print(" min/max:\n x {}\n y {}\n z {}".format((minX, maxX),
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
        print(" mapped layer {}".format(l))


inv = conduction.InversionND(layer_voxel, mesh, lithology_index=layer_number) # solver="bcgs"
inv.nIter = 0

# map properties to mesh
rho, alpha, k0, H, beta, gmma_T, K_0p, K_T, man = inv.map(*layer_attributes.T)
grun_mask = gmma_T == 0
K_T[K_T == 0] = 1e99


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
mesh.bc['maxZ']['mask'] = air_mask
mesh.bc['minZ']['mask'] = lab_mask
mesh.dirichlet_mask[kmask] = True


# Boundary conditions
topBC = 298.0
bottomBC = 1300.0 + 273.14

mesh.boundary_condition('maxZ', topBC, flux=False)
mesh.boundary_condition('minZ', bottomBC, flux=False)


cost = forward_model(xN, inv, 'Z')

# write output
H5_file = args.echo
if not H5_file.endswith('.h5'):
    H5_file += '.h5'

# Calculate heat flow
qz, qy, qx = mesh.heatflux()

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
