"""
Copyright 2017 Ben Mather

This file is part of Conduction <https://git.dias.ie/itherc/conduction/>

Conduction is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Conduction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Conduction.  If not, see <http://www.gnu.org/licenses/>.
"""
try: range=xrange
except: pass

import numpy as np

def gaussian(sigma, distance, length_scale):
    """
    Gaussian function
    """
    return sigma**2 * np.exp(-distance**2/(2.0*length_scale**2))

def create_covariance_matrix(sigma, coords, max_dist, func, *args):
    """
    Create a covariance matrix based on distance.
    Euclidean distance between a set of points is queried from a KDTree
    where max_dist sets the maximum radius between points to cut-off.

    Parameters
    ----------
     sigma    : values of uncertainty for each point
     coords   : coordindates in n-dimensions for each point
     max_dist : maximum radius to search for points
     func     : covariance function (default is Gaussian)
        (pass a length parameter if using default)
     args     : arguments to pass to func

    Returns
    -------
     mat      : covariance matrix

    Notes
    -----
     Increasing max_dist increases the number of nonzeros in the
     covariance matrix.

     func should always receive sigma and distance as first
     two inputs
    """
    from scipy.spatial import cKDTree
    from petsc4py import PETSc

    size = len(sigma)

    # set up matrix
    mat = PETSc.Mat().create()
    mat.setType(mat.Type.AIJ)
    mat.setSizes((size, size))
    mat.setPreallocationNNZ((size,1))
    mat.setFromOptions()
    mat.assemblyBegin()

    # set up KDTree and max_dist to query
    tree = cKDTree(coords)

    for i in range(0, size):
        idx = tree.query_ball_point(coords[i], max_dist)
        dist = np.linalg.norm(coords[i] - coords[idx], axis=1)
        
        row = i
        col = idx
        val = func(sigma[idx], dist, *args)
        
        mat.setValues(row, col, val)

    mat.assemblyEnd()
    return mat
