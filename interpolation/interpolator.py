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

from scipy.interpolate import RegularGridInterpolator as RGI
import itertools
import numpy as np

class RegularGridInterpolator(RGI):

    def __init__(self, points, values, method="linear", bounds_error=False, fill_value=np.nan):

        super(RegularGridInterpolator, self).__init__(points, values, method, bounds_error, fill_value)


    def adjoint(self, xi, dxi, method=None):
        """
        Interpolation adjoint using the derivatives dxi at coordinates xi
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        dxi : ndarray of shape (..., ndim)
             The derivatives at the coordinates xi
        method : str
                The method of interplolation to perform.
                Supports either 'linear' or 'nearest'
        """
        if method is None:
            method = self.method
            
        self.derivative = dxi
        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear_adjoint(indices, norm_distances, out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest_adjoint(indices, norm_distances, out_of_bounds)
        return result

    def _evaluate_linear_adjoint(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = np.zeros_like(self.values)
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values[edge_indices] += np.asarray(self.derivative) * weight[vslice]
        return values

    def _evaluate_nearest_adjoint(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        values = np.zeros_like(self.values)
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= 0.5, i, i + 1))
        values[idx_res] = self.derivative
        return values


class KDTreeInterpolator(object):
    """
    This object implements scipy.spatial.cKDTree for fast nearest-neighbour lookup
    and interpolates the coordinate based on the next closes value

    Arguments
    ---------
     points        : ndarray(n,dim) arbitrary set of points to interpolate over
     values        : ndarray(n,) values at the points
     bounds_error  : bool, optional
     fill_value    : number, optional

    Methods
    -------
     __call__
     adjoint

    """
    def __init__(self, points, values, bounds_error=False, fill_value=np.nan):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(points)
        data = self.tree.data
        npoints, ndim = data.shape
        
        self.fill_value = fill_value
        self.bounds_error = bounds_error
        
        bbox = []
        for i in range(0, ndim):
            bbox.append((data[:,i].min(), data[:,i].max()))
        
        self.npoints = npoints
        self.ndim = ndim
        self.bbox = bbox
        
        self._values = np.ravel(values)
        
    def __call__(self, xi, *args, **kwargs):
        idx, d, bmask = self._find_indices(xi)
        
        if self.bounds_error and bmask.any():
            bidx = np.nonzero(bmask)[0]
            raise ValueError("Coordinates in xi are out of bounds in:\n {}".format(bidx))
        
        vi = self.values[idx]
        if not self.bounds_error and self.fill_value is not None:
            vi[bmask] = self.fill_value
        
        return vi
    
    def adjoint(self, xi, dxi, *args, **kwargs):
        idx, d, bmask = self._find_indices(xi)
        
        if self.bounds_error and bmask.any():
            bidx = np.nonzero(bmask)[0]
            raise ValueError("Coordinates in xi are out of bounds in:\n {}".format(bidx))
        
        dv = np.zeros_like(self.values)

        # remove indices that are out of bounds
        idx_inbounds = idx[~bmask]
        
        ux = np.unique(idx_inbounds)
        for u in ux:
            dv[u] = dxi[idx==u].sum()

        if not self.bounds_error and self.fill_value is not None:
            dv[idx][bmask] = self.fill_value
            
        return dv
    
    def _find_indices(self, xi):
        d, idx = self.tree.query(xi)
        bbox = self.bbox
        ndim = self.ndim
        
        bounds = np.zeros(idx.size, dtype=bool)
        for i in range(0, ndim):
            bmin, bmax = bbox[i]
            mask = np.logical_or(xi[:,i] < bmin, xi[:,i] > bmax)
            bounds[mask] = True
            
        return idx, d, bounds


    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, value):
        self._values = value.ravel()
    @values.getter
    def values(self):
        return self._values