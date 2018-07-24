import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def objective_function_fm(x, x0, cov):
    """ forward model """
    misfit = np.array(x - x0)
    res = 0.5*misfit.T*spsolve(cov, misfit)
    return res.sum()

def objective_function_tl(x, dx, x0, cov):
    """ tangent linear """
    misfit = np.array(x - x0)
    res  = spsolve(cov, misfit)
    res *= dx
    return res.sum()

def objective_function_ad(x, x0, cov):
    """ adjoint model """
    misfit = np.array(x - x0)
    ones = np.ones_like(misfit)
    res  = spsolve(cov, misfit)
    return res

def gaussian_function(sigma, distance, length_scale):
    return sigma**2 * np.exp(-distance**2/(2.0*length_scale**2))

def construct_covariance_matrix(sigma, coords, length_scale):
    size = len(sigma)
    mat = sparse.lil_matrix((size, size))

    for i in range(size):
        distance = coords[i] - coords
        mat[i,:] = gaussian_function(sigma, distance, length_scale)

    return mat.tocsc()


size = 10
x = np.linspace(5, 10, 10)
x0 = np.ones(size)*2.5
sigma_x = np.ones(size)*10.0
dx = x*0.1

coords = np.linspace(0, 1, size)


cov = construct_covariance_matrix(sigma_x, coords, 1.0)

cov_dense = cov.todense()
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
im1 = ax1.imshow(cov_dense)
fig.colorbar(im1)
plt.show()


fm0 = objective_function_fm(x, x0, cov)
fm1 = objective_function_fm(x + dx, x0, cov)
tl = objective_function_tl(x, dx, x0, cov)
ad = objective_function_ad(x, x0, cov)

print("finite differences = {}".format(fm1-fm0))
print("tangent linear = {}".format(tl))
print("adjoint model = {}".format(ad.dot(dx)))


