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
from conduction.tools import PerplexTable
from conduction.inversion import InvObservation, InvPrior
from conduction.inversion import create_covariance_matrix
from conduction.inversion import gaussian_function

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD


parser = argparse.ArgumentParser(description='Process some model arguments.')
parser.add_argument('--log', required=False, action='store_true', default=False, help='Log the adjoint solves')
parser.add_argument('echo', type=str, metavar='PATH', help='Output file path')
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

def hofmeister1999_ad(k0, T):
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
    # k_new = k_con*k_exp*k_pres + k_rad

    dkdk0 = k_pres*k_exp*(298.0/T)**0.45
    dkdT  = -3.0*3.407e-11*T**2 + 2.0*T*2.245e-7 - 1.0374e-4 + \
            k_pres*k_con*(4.0*gmma_T + 1.0/3)*(-T**4*d - T*b - a - c/T**2) - \
            k_pres*0.45/T*k_con*k_exp
    return dkdk0, dkdT


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

def save_variables(filename, x, **kwargs):
    kt, Ht = np.array_split(x[:-1], 2)
    Tb = x[-1]
    np.savez_compressed(filename, k=kt, H=Ht, Tb=Tb, **kwargs)

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


def adjoint_model(x, self, bc='Z'):
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
     grad : [dk_list, dH_list, dTb]

    """
    k_list, H_list = np.array_split(x[:-1], 2)
    Tb = x[-1]
    
    # map to mesh
    k0, H = self.map(k_list, H_list)
    k = [k0.copy()]
    T = [None]
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, Tb, flux=False)
    rhs = self.mesh.construct_rhs()
    
    error = np.array(10.0)
    tolerance = 1e-5
    i = 0
    self.mesh.temperature._gdata.set(0.)
    while error > tolerance:
        self.mesh.diffusivity[:] = k[i]
        # solve
        Ti = self.linear_solve(rhs=rhs)
        ki = hofmeister1999(k0, Ti)
        T.append(Ti.copy())
        k.append(ki.copy())
        err = np.absolute(k[-1] - k[-2]).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)
        i += 1

    q = self.heatflux(T[-1], k[-1])
    delT = self.gradient(T[-1])
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T[-1], delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, Tb=Tb) # priors
    
    ## AD ##
    dk = np.zeros_like(H)
    dH = np.zeros_like(H)
    dT = np.zeros_like(H)
    dk0 = np.zeros_like(H)
    dTb = np.array(0.0)
    
    # priors
    dcdk_list = self.objective_routine_ad(k=k_list)
    dcdH_list = self.objective_routine_ad(H=H_list)
    dcdTb = self.objective_routine_ad(Tb=Tb)

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
        dkdk0, dkdT = hofmeister1999_ad(k0, T[-1-j])
        dk0 += dkdk0*dk
        dT  += dkdT*dk
        
        dk.fill(0.0)
        
        self.mesh.diffusivity[:] = k[-1-j]
        dA, db = self.linear_solve_ad(T[-1-j], dT)
        dk += dA
        dH += -db
        lowerBC_mask = self.mesh.bc["min"+bc]["mask"]
        dTb_local  = np.sum(db[lowerBC_mask]/inv.ghost_weights[lowerBC_mask])
        dTb_global = np.array(0.0)
        comm.Allreduce([dTb_local, MPI.DOUBLE], [dTb_global, MPI.DOUBLE], op=MPI.SUM)
        dTb += dTb_global
        
        dT.fill(0.0)
        
    dk0 += dk
        
    # pack to lists
    dk_list, dH_list = inv.map_ad(dk0, dH)
    dk_list += dcdk_list
    dH_list += dcdH_list
    dTb += dcdTb
    
    dx = np.hstack([dk_list, dH_list, [dTb]])

    if args.log:
        save_variables(args.echo + '_step{:04d}.npz'.format(self.nIter), x, T=T[-1], q=q, cost=cost)
    self.nIter += 1
    
    return cost, dx


directory = '../data/Ireland_model/'
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
    print(" min/max:\n x {}\n y {}\n z {}".format((minX, maxX),
                                                  (minY, maxY),
                                                  (minZ, maxZ)))

## Setup the hexahedral mesh

Nx, Ny, Nz = 51, 51, 204

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



## Priors
mat = np.loadtxt(directory + 'material_properties.csv', delimiter=',', usecols=(0,1,2,3,4,5), dtype=str, skiprows=1)
mat_name = mat[:,0]
mat_ID   = mat[:,1].astype(int)
mat_k    = mat[:,2:4].astype(float)
mat_H    = mat[:,4:6].astype(float)

assert np.all(mat_ID == inv.lithology_index), "Material Index should be identical"

size = len(inv.lithology_index)

if comm.rank == 0:
    row_format = " {0:2} | {1:35}| {2:.2f} | {3:.2f} | {4:.2e} | {5:.2e}"
    for i in range(size):
        print(row_format.format(mat_ID[i], mat_name[i], mat_k[i,0], mat_k[i,1], mat_H[i,0], mat_H[i,1]))

kp  = InvPrior(mat_k[:,0], mat_k[:,1])
Hp  = InvPrior(mat_H[:,0], mat_H[:,1])
Tbp = InvPrior(bottomBC, 100.0)
inv.add_prior(k=kp, H=Hp, Tb=Tbp)


## Observations
HF_file = directory + "ireland_heat_flow_proj.csv"
ireland_HF = np.loadtxt(HF_file, delimiter=',', usecols=(2,3,4,11,12), skiprows=1)
qmask = ireland_HF[:,4] != 0
eire_HF  = ireland_HF[qmask,3] * 1e-3
eire_dHF = ireland_HF[qmask,4] * 1e-3
eire_xyz = ireland_HF[qmask,0:3]
eire_xyz[:,2] = -600.0

qobs = InvObservation(eire_HF, eire_dHF, eire_xyz)
inv.add_observation(q=qobs)


curie_file = directory + "Li_2017_curiedepth_ireland.txt"
cpd = np.loadtxt(curie_file, skiprows=1)
cpd_xyz = cpd[:,:3]
cpd_xyz[:,2] *= -1
cpd_T  = np.ones(cpd.shape[0])*(580+273.14)
cpd_dT = cpd[:,4]

# half window sizes
L1 = 98.8e3/4
L2 = 195.0e3/4
L3 = 296.4e3/4
cov1 = create_covariance_matrix(cpd_dT, cpd_xyz[:,:2], L1*4, gaussian_function, length_scale=L1)
cov2 = create_covariance_matrix(cpd_dT, cpd_xyz[:,:2], L2*4, gaussian_function, length_scale=L2)
cov3 = create_covariance_matrix(cpd_dT, cpd_xyz[:,:2], L3*4, gaussian_function, length_scale=L3)
# average these 3 matrices
cov_mean = cov1 + cov2 + cov3
cov_mean.scale(1.0/3)

Tobs = InvObservation(cpd_T, cpd_dT, cpd_xyz, cov_mean)
inv.add_observation(T=Tobs)



# Starting x at priors
x0 = np.hstack([kp.v, Hp.v, [Tbp.v]])


# bounded optimisation
bounds = {'k' : (0.5, 10.0),
          'H' : (0.0, 10e-6),
          'a' : (0.0, 1.0),
          'q0': (5e-3, 70e-3),
          'Tb': (1000.0, 3000.0)}


# Create bounds the same length as x
k_lower   = np.ones(size)*bounds['k'][0]
H_lower   = np.ones(size)*bounds['H'][0]
Tb_lower  = bounds['Tb'][0]
k_upper   = np.ones(size)*bounds['k'][1]
H_upper   = np.ones(size)*bounds['H'][1]
Tb_upper  = bounds['Tb'][1]

x_lower = np.hstack([k_lower, H_lower, [Tb_lower]])
x_upper = np.hstack([k_upper, H_upper, [Tb_upper]])
x_bounds = list(zip(x_lower, x_upper))


## Test
x = x0 + 0.01*x0
dx = 0.01*x
fm0 = forward_model(x, inv, 'Z')
fm1 = forward_model(x+dx, inv, 'Z')
ad  = adjoint_model(x, inv, 'Z')
print("finite differences = {}".format(fm1-fm0))
print("adjoint model = {}".format(ad[1].dot(dx)))


## Commence inversion
options = {'maxiter': len(x0)*100}
if comm.rank == 0:
    options['disp'] = 1

t = time()
res = minimize(adjoint_model, x0, args=(inv, 'Z'), method='TNC', jac=True, bounds=x_bounds, options=options)

if comm.rank == 0:
    print(res)
    print("Completed in {} minutes".format((time() - t)/60))

    save_variables(args.echo + "_minimiser_results.npz", res.x)