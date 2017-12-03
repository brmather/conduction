
# coding: utf-8

# In[17]:


import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, SmoothBivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
import conduction
from conduction.tools import PerplexTable
from conduction.inversion import InvObservation, InvPrior
import time

from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD
import argparse

parser = argparse.ArgumentParser(description='Process some model arguments.')
parser.add_argument('--log', required=False, action='store_true', default=False, help='Log the adjoint solves')
parser.add_argument('echo', type=str, metavar='PATH', help='Output file location')
args = parser.parse_args()


def save_variables(filename, x, **kwargs):
    kt, Ht, at = np.array_split(x[:-1], 3)
    q0t = x[-1]
    np.savez_compressed(filename, k=kt, H=Ht, a=at, q0=q0t, **kwargs)


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
        dq0 += np.sum(-db[lowerBC_mask]/dz/inv.ghost_weights[lowerBC_mask])
        
        dT.fill(0.0)
        
    dk0 += dk
        
    # pack to lists
    dk_list, dH_list, da_list = inv.map_ad(dk0, dH, da)
    dk_list += dcdk_list
    dH_list += dcdH_list
    da_list += dcda_list
    dq0 += dcdq0
    
    dx = np.hstack([dk_list, dH_list, da_list, [dq0]])

    if args.log:
        save_variables(args.echo + '_step{:04d}.npz'.format(self.nIter), x, T=T[-1], q=q, cost=cost)
    self.nIter += 1
    
    return cost, dx


# In[19]:


directory = '/opt/ben/Dropbox/GOTherm/LITMOD3D_2/'

layer_attributes = np.loadtxt(directory+'layers.info', skiprows=1, usecols=(2,3,4,5,6,7,8,9,10))
layer_number = np.loadtxt(directory+'layers.info', dtype=int, skiprows=1, usecols=(0,))
layer_name   = np.loadtxt(directory+'layers.info', dtype=str, skiprows=1, usecols=(1,))

layer_header = ['body number', 'density', 'alpha', 'thermal conductivity', 'heat production rate', 'pressure coefficient', 'Gruneisen parameter', 'pressure derivative of bulk modulus', 'man']

minZ, maxZ = 0.0, 0.0

layer = dict()
for i in xrange(0, 10):
    data = np.loadtxt(directory+'layers_xy/layer{}.xyz'.format(i))
    layer[i] = data

    dminX, dminY, dminZ = data.min(axis=0)
    dmaxX, dmaxY, dmaxZ = data.max(axis=0)
    minZ = min(minZ, dminZ)
    maxZ = max(maxZ, dmaxZ)


# In[20]:


Xcoords = np.unique(data[:,0])
Ycoords = np.unique(data[:,1])

nx, ny = Xcoords.size, Ycoords.size

minX, minY = dminX, dminY
maxX, maxY = dmaxX, dmaxY

# Overwrite dimensions
## default are
## min/max:
##  x (350000000.0, 788000000.0)
##  y (480000000.0, 1000000000.0)
##  z (-35000.0, 122305302.84092408)
## Nx, Ny, Nz = 55, 55, 105

minZ = -35e3
maxZ = 800.0
Nx, Ny, Nz = 50, 50, 35

if comm.rank == 0:
    print("min/max:\n x {}\n y {}\n z {}".format((minX, maxX),
                                                 (minY, maxY),
                                                 (minZ, maxZ)))


# In[21]:


spl = dict()

for i in xrange(10):
    data = layer[i]
    xl = data[:,0]
    yl = data[:,1]
    zl = data[:,2].reshape(ny,nx)
    spl[i] = RectBivariateSpline(Ycoords, Xcoords, zl)


## Setup the hexahedral mesh

mesh = conduction.ConductionND((minX, minY, minZ), (maxX, maxY, maxZ), (Nx, Ny, Nz))

coords = mesh.coords

Xcoords = np.unique(coords[:,0])
Ycoords = np.unique(coords[:,1])
Zcoords = np.unique(coords[:,2])

nz, ny, nx = mesh.n




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

layer_voxel.fill(0)


for l in xrange(0,10):
    i0, j0, k0 = query_nearest(l)

    for i in xrange(i0.size):
        layer_voxel[:i0[i], j0[i], k0[i]] = l+1

    if comm.rank == 0:
        print("mapped layer {}".format(l))



# Fill everything above value
layer_voxel[layer_voxel > 9] = 9

lithology_index = np.unique(layer_voxel)


k  = np.zeros_like(layer_voxel, dtype=np.float)
H  = np.zeros_like(layer_voxel, dtype=np.float)
a  = np.zeros_like(layer_voxel, dtype=np.float)
q0 = 40e-3
sigma_k  = np.zeros_like(layer_voxel, dtype=np.float)
sigma_H  = np.zeros_like(layer_voxel, dtype=np.float)
sigma_a  = np.zeros_like(layer_voxel, dtype=np.float)
sigma_q0 = 10e-3/4

thermal_cond = layer_attributes[:,2]
heat_sources = layer_attributes[:,3]


# air + moho layer
layer_name = np.hstack([['Air'], layer_name, ['Moho']])
thermal_cond = np.hstack([[3.5],thermal_cond,[3.5]])
heat_sources = np.hstack([[0.0],heat_sources,[0.0]])

row_format = "{0:2} | {1:25}| {2:.2f} | {3:.2f} | {4:.2f}"

for i, l in enumerate(lithology_index):
    idx = layer_voxel == l

    name = layer_name[i]
    ki = thermal_cond[i]
    Hi = heat_sources[i]
    ai = 0.33

    k[idx] = ki
    H[idx] = Hi
    a[idx] = ai

    sigma_k[idx] = ki*0.1
    sigma_H[idx] = 1e-7
    sigma_a[idx] = 0.1

    if comm.rank == 0:
        print(row_format.format(i, name, ki, Hi*1e6, ai))


k = k.ravel()
H = H.ravel()
a = a.ravel()
sigma_k = sigma_k.ravel()
sigma_H = sigma_H.ravel()
sigma_a = sigma_a.ravel()


size = layer_voxel.size

# bounded optimisation
bounds = {'k' : (0.5, 5.5),
          'H' : (0.0, 10e-6),
          'a' : (0.0, 1.0),
          'q0': (5e-3, 50e-3)}

size = mesh.nn
x_bounds = [bounds['k'],]*size
x_bounds.extend([bounds['H'],]*size)
x_bounds.extend([bounds['a'],]*size)
x_bounds.append(tuple(bounds['q0']))


# lithologies correspond to number of nodes
inv_variables = np.arange(0, size)



mesh.boundary_condition('maxZ', 298.0, flux=False)
mesh.boundary_condition('minZ', q0, flux=True)

inv = conduction.InversionND(inv_variables, mesh)
inv.nIter = 0


# Priors
kp   = InvPrior(k, sigma_k)
Hp   = InvPrior(H, sigma_H)
ap   = InvPrior(a, sigma_a)
q0p  = InvPrior(q0, sigma_q0)

inv.add_prior(k=kp, H=Hp, a=ap, q0=q0p)



# Observations
ireland_HF = np.loadtxt('/opt/ben/Dropbox/GOTherm/data/ireland_heat_flow_proj.csv', delimiter=',', dtype=str, skiprows=1)

eire_ID  = ireland_HF[:,1]
eire_E   = ireland_HF[:,2].astype(float)
eire_N   = ireland_HF[:,3].astype(float)
eire_HF  = ireland_HF[:,9].astype(float) * 1e-3
eire_dHF = ireland_HF[:,10].astype(np.float) * 1e-3
eire_dHF[eire_dHF==0] = eire_HF[eire_dHF==0]*0.2

eire_coord = np.zeros((eire_HF.size, 3))
eire_coord[:,0] = eire_E
eire_coord[:,1] = eire_N

qobs = InvObservation(eire_HF, eire_dHF, eire_coord)
inv.add_observation(q=qobs)

curie_coords = np.loadtxt('/opt/ben/Dropbox/GOTherm/curie-var-heat-vert-const_proj.xyz')
curie_coords[:,2] *= 1e3

curie_T = np.ones(curie_coords.shape[0])*(273.14 + 580.0)
sigma_curie_T = curie_T*0.2

Tobs = InvObservation(curie_T, sigma_curie_T, curie_coords)
inv.add_observation(T=Tobs)

# In[41]:


inv.ksp = inv._initialise_ksp(solver='gmres', pc='bjacobi')
inv.ksp_ad  = inv._initialise_ksp(solver='bcgs', pc='bjacobi')
inv.ksp_adT = inv._initialise_ksp(solver='bcgs')


# In[37]:

# starting x0
x0 = np.hstack([k, H, a, [q0]])
x = x0
dx = 0.01*x


# Save mesh attributes
# create attributes
(xmin, xmax), (ymin, ymax), (zmin, zmax) = inv.mesh.dm.getBoundingBox()
minCoord = np.array([xmin, ymin, zmin])
maxCoord = np.array([xmax, ymax, zmax])
shape = inv.mesh.dm.getSizes()
np.savez_compressed(args.echo + '_mesh_topology.npz', minCoord=minCoord, maxCoord=maxCoord, shape=shape)


walltime = time.time()

fm0 = forward_model(x, inv, 'Z')
fm1 = forward_model(x+dx, inv, 'Z')
ad = adjoint_model(x, inv, 'Z')

print "finite difference", (fm1 - fm0)
print "adjoint", ad[1].dot(dx)
print "\nwalltime", time.time() - walltime

options = {'disp': 1, 'maxiter': len(x0)*100}

t = time.clock()
res = minimize(adjoint_model, x0, args=(inv, 'Z'), method='TNC', jac=True, bounds=x_bounds, options=options)
if comm.rank == 0:
    print res
    print "Completed in {}".format(time.clock()-t)

    save_variables(args.echo + '_minimiser_results.npz', res.x)
