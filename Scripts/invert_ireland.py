
import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, SmoothBivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
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



minX, minY, minZ = data.min(axis=0)
maxX, maxY, maxZ = data.max(axis=0)

# minZ = -400e3
minZ = -130e3
maxZ = 600.0

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

Nx, Ny, Nz = 55, 55, 105

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


# This makes it the same as xyz files
# layer_voxel = np.rot90(layer_voxel, axes=(1,2))
# layer_voxel = layer_voxel[:,::-1,:]




# Boundary conditions
topBC = 298.0
bottomBC = 0.03

mesh.boundary_condition('maxZ', topBC, flux=False)
mesh.boundary_condition('minZ', bottomBC, flux=True)

inv = conduction.Inversion(layer_voxel.flatten(), mesh)

k = layer_attributes[:,2]
H = layer_attributes[:,3]
a = np.zeros_like(k)
q0 = 30e-3

# air + moho layer
layer_name = np.hstack([['Air'], layer_name, ['Moho']])
k = np.hstack([[3.5], k, [3.5]])
H = np.hstack([[0.0], H, [0.0]])
a = np.zeros_like(k)

for i, l in enumerate(inv.lithology_index):
    name = layer_name[i]
    ki = k[i]
    Hi = H[i]
    if comm.rank == 0:
        print('{:2} {:15} \t k = {}, H = {}'.format(l, name, ki, Hi))


x = PETSc.Vec().createWithArray(np.hstack([k, H, a, [30e-3]]))
dx = 0.01*x
gradient = x.duplicate()


# Priors
k_prior = k.copy()
H_prior = H.copy()
a_prior = a.copy()

sigma_k = k*0.2
sigma_H = H*0.2
sigma_a = a*0.2

inv.add_prior(k=(k_prior,sigma_k), H=(H_prior,sigma_H), q0=(30e-3, 5e-3))


# Observations
brock_1989 = np.loadtxt('/opt/ben/Dropbox/GOTherm/data/ireland_heat_flow.csv', delimiter=',', dtype=str, skiprows=1)

brock_ID  = brock_1989[:,1]
brock_lat = brock_1989[:,2].astype(float)
brock_lon = brock_1989[:,3].astype(float)
brock_HF  = brock_1989[:,9].astype(float) * 1e-3
brock_dHF = brock_1989[:,10].astype(float)* 1e-3

import benpy
brock_eastings, brock_northings = benpy.transform_coordinates(brock_lon, brock_lat, 4326, 2157)
brock_coord = np.zeros((brock_HF.size, 3))
brock_coord[:,0] = brock_eastings
brock_coord[:,1] = brock_northings
brock_coord[:,2] = maxZ

inv.add_observation(q=(brock_HF, brock_dHF, brock_coord[:,::-1]))

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(inv.mesh.coords[:,0], inv.mesh.coords[:,1], inv.mesh.coords[:,2], c=inv.mesh.bc['maxZ']['mask'])
# ax1.scatter(brock_coord[:,0], brock_coord[:,1], brock_coord[:,2])
# plt.show()

fm0 = inv.forward_model(x)
fm1 = inv.forward_model(x+dx)
tl  = inv.tangent_linear(x, dx)
ad  = inv.adjoint(None, x, gradient)

if comm.rank == 0:
    print "--- TEST ADJOINT ---"
    print "finite difference", (fm1 - fm0)
    print "tangent linear", tl[1]
print "adjoint", ad, gradient.array.dot(dx)
# print "\ncommencing inversion..."


## Inversion
class InversionHelper(object):
    """ compatibility layer for scipy """
    def __init__(self, inv, vec):
        self.inv = inv
        self.x = PETSc.Vec().createWithArray(vec)
        self.G = x.duplicate()
    def forward_model(self, x):
        self.x.setArray(x)
        cost = self.inv.forward_model(self.x)
        return cost
    def adjoint(self, x):
        self.x.setArray(x)
        cost = self.inv.adjoint(None, self.x, self.G)
        return cost, self.G.array.copy()

sinv = InversionHelper(inv, x)

# bounded optimisation
bounds = {'k' : (0.5, 5.5),
          'H' : (0.0, 10e-6),
          'a' : (0.0, 0.0),
          'q0': (5e-3, 50e-3)}

size = len(inv.lithology_index)
x_bounds = [bounds['k'],]*size
x_bounds.extend([bounds['H'],]*size)
x_bounds.extend([bounds['a'],]*size)
x_bounds.append(tuple(bounds['q0']))

# air
# x_bounds[0] = (5.0, 5.0)
# x_bounds[size] = (0.0, 0.0)


if comm.rank == 0:
    options = {'disp': 1, 'maxiter': x.array.size*100}
else:
    options = {'maxiter': x.array.size*100}
res = minimize(sinv.adjoint, x.array.copy(), method='TNC', jac=True, bounds=x_bounds, options=options)
if comm.rank == 0:
    print res

H5_file = 'invert_ireland.h5'


# run forward model with inverted x
sinv.forward_model(res.x)

# Calculate heat flow
qx, qy, qz = inv.mesh.heatflux()


# Save to H5 file
inv.mesh.save_mesh_to_hdf5(H5_file)
inv.mesh.save_field_to_hdf5(H5_file,\
                        layer_ID=layer_voxel.ravel(),\
                        conductivity=inv.mesh.diffusivity[:],\
                        heat_production=inv.mesh.heat_sources[:]*1e6,\
                        temperature=inv.mesh.temperature[:]-273.14,\
                        heat_flux=(qx+qy+qz)*1e3)
inv.mesh.save_vector_to_hdf5(H5_file,\
                        heat_flux_vector=(qx*1e3,qy*1e3,qz*1e3),\
                        heat_flux_vector_lateral=(qx*1e3,qy*1e3,np.zeros_like(qz)))
if comm.rank == 0:
    conduction.generateXdmf(H5_file)

    # Save to CSV file
    import csv
    kHa = np.reshape(res.x[:-1], (-1,3), order='F')
    with open('invert_ireland.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['lithology', 'index', 'k', 'H', 'a', 'q0'])
        for row in xrange(0, kHa.shape[0]):
            writer.writerow([layer_name[row],inv.lithology_index[row], kHa[row,0], kHa[row,1], kHa[row,2], res.x[-1]])