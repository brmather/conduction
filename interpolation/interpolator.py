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