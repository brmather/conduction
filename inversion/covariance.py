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
from mpi4py import MPI
comm = MPI.COMM_WORLD


def gaussian(sigma, distance, length_scale):
    """
    Gaussian function
    """
    return sigma**2 * np.exp(-distance**2/(2.0*length_scale**2))

def create_covariance_matrix(sigma, coords, max_dist, func, *args, **kwargs):
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
     kwargs   : keyword arguments to pass to func

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

    ncov = len(sigma)
    nn = np.hstack(sigma).size
    if ncov < nn:
        # this means we have a piecewise covariance matrix
        if len(coords) != ncov or len(max_dist) != ncov:
            raise ValueError("sigma, coords, max_dist must be the same dimensions")

        # find the size of each list
        size = np.zeros(ncov, dtype=PETSc.IntType)
        nnz = []
        for j in range(0,ncov):
            size[j] = len(sigma[j])

            dist = np.linalg.norm(coords[j] - coords[j].mean(axis=0), axis=1)
            nnz_j = int(1.5*(dist <= max(max_dist[j])).sum())
            nnz.append( np.ones(size[j], dtype=PETSc.IntType)*nnz_j )

        nnz = np.clip(np.hstack(nnz), 1, 99999)


    elif ncov == nn:
        ncov = 1
        size = [len(sigma)]

        # find distance between coords and centroid
        dist = np.linalg.norm(coords - coords.mean(axis=0), axis=1)

        # pad these within a list object
        coords = [coords]
        max_dist = [np.ones_like(sigma)*max_dist]
        sigma = [sigma]

        nnz = int(1.5*(dist <= max(max_dist)).sum())


    # set up matrix
    mat = PETSc.Mat().create(comm)
    mat.setType('aij')
    mat.setSizes((nn, nn))
    mat.setPreallocationNNZ((nnz,1))
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, 0)
    mat.setFromOptions()
    mat.assemblyBegin()

    colsum = np.cumsum(np.insert(size, 0, 0))

    row = 0
    for j in range(0, ncov):
        icoord = coords[j]
        imax_dist = max_dist[j]
        isigma = sigma[j]

        # set up KDTree and max_dist to query
        tree = cKDTree(icoord)

        for i in range(0, size[j]):
            idx = tree.query_ball_point(icoord[i], imax_dist[i])
            dist = np.linalg.norm(icoord[i] - icoord[idx], axis=1)
            
            col = np.array(idx + colsum[j], dtype=PETSc.IntType)
            val = func(isigma[idx], dist, *args, **kwargs)
            mat.setValues(row, col, val)
            
            row += 1

    mat.assemblyEnd()
    return mat


def create_covariance_matrix_index(sigma, coords, max_dist, index, func, *args, **kwargs):
    """
    Create a covariance matrix based on distance.
    Euclidean distance between a set of points is queried from a KDTree
    where max_dist sets the maximum radius between points to cut-off.

    Parameters
    ----------
     sigma    : values of uncertainty for each point
     coords   : coordindates in n-dimensions for each point
     max_dist : maximum radius to search for points
     index    : map covariance function to regions of identical index
     func     : covariance function (default is Gaussian)
        (pass a length parameter if using default)
     args     : arguments to pass to func
     kwargs   : keyword arguments to pass to func

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

    # find distance between coords and centroid
    dist = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    nnz = int(1.5*(dist <= max_dist).sum())


    # set up matrix
    mat = PETSc.Mat().create(comm)
    mat.setType('aij')
    mat.setSizes((size, size))
    mat.setPreallocationNNZ((nnz,1))
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, 0)
    mat.setFromOptions()
    mat.assemblyBegin()
    
    
    lith_index = np.unique(index)

    for l in lith_index:
        indices = np.nonzero(index == l)[0].astype(PETSc.IntType)

        icoord = coords[indices]
        isigma = sigma[indices]
        tree = cKDTree(icoord)

        for i in indices:
            idx = tree.query_ball_point(coords[i], max_dist)
            dist = np.linalg.norm(coords[i] - icoord[idx], axis=1)
            
            row = i
            col = indices[idx]
            val = func(isigma[idx], dist, *args, **kwargs)
            mat.setValues(row, col, val)

    mat.assemblyEnd()
    return mat