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

import numpy as np
from .covariance import gaussian as gaussian_function
from .covariance import create_covariance_matrix as covariance_matrix

class InvObservation(object):
    """
    Inversion Observation for use in an objective function

    This requires an interpolation object with information on
    ghost points

    Arguments
    ---------
     label      : label to give observation
     obs        : observations
     obs_err    : observational uncertainty
     obs_coords : coordinates in Cartesian N-dimensional space
                : (optional) ndarray shape (n, dim)
     cov_mat    : data covariance matrix
                  (optional) uses the l2-norm otherwise
    """
    def __init__(self, obs, obs_err, obs_coords=None, cov_mat=None):

        self.v = obs
        self.dv = obs_err
        self.coords = obs_coords
        self.cov = cov_mat

        # self.gweight = self.ghost_weights()

    def __delete__(self):
        if type(self.cov) != type(None):
            self.cov.destroy()

    def construct_covariance_matrix(self, max_dist, func=gaussian_function, *args, **kwargs):
        """
        Construct a covariance matrix based on the uncertainty
        of the data and a distance scale.

        See inversion.covariance.create_covariance_matrix for details.

        Arguments
        ---------
         max_dist : maximum radius to search for points
         func     : covariance function (default is Gaussian)
            (pass a length parameter if using default)
         args     : arguments to pass to func
         kwargs   : keyword arguments to pass to func
        """
        sigma = self.dv
        self.cov = covariance_matrix(sigma, self.coords, max_dist, func, *args, **kwargs)


    def ghost_weights(self):

        interp = self.interp

        w = interp(self.coords)
        w = 1.0/np.floor(w + 1e-12)
        offproc = np.isnan(w)
        w[offproc] = 0.0 # these are weighted with zeros

        return w



class InvPrior(object):
    """
    Prior for use in an objective function

    Arguments
    ---------
     label        : label to give prior
     prior        : prior
     prior_err    : prior uncertainty
     prior_coords : prior coordinates
                    (optional) ndarray shape(n, dim)
     cov_mat      : prior covariance matrix
                    (optional) uses the l2-norm otherwise
    """
    def __init__(self, prior, prior_err, prior_coords=None, cov_mat=None):

        self.v = prior
        self.dv = prior_err
        self.coords = prior_coords
        self.cov = cov_mat
        # self.gweight = 1.0

    def __delete__(self):
        if type(self.cov) != type(None):
            self.cov.destroy()

    def construct_covariance_matrix(self, max_dist, func=gaussian_function, *args, **kwargs):
        """
        Construct a covariance matrix based on the uncertainty
        of the prior and a distance scale.

        See inversion.covariance.create_covariance_matrix for details.

        Arguments
        ---------
         max_dist : maximum radius to search for points
         func     : covariance function (default is Gaussian)
            (pass a length parameter if using default)
         args     : arguments to pass to func
         kwargs   : keyword arguments to pass to func
        """
        sigma = self.dv
        self.cov = covariance_matrix(sigma, self.coords, max_dist, func, *args, **kwargs)