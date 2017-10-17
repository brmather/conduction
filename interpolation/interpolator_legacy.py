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
from scipy.interpolate import interp1d

class LinearInterpolation(interp1d):
    """
    Extends basic 1D interpolation
    """
    def __init__(self, x, y):
        """
        Inherit the interp1D class from SciPy
        This includes all methods, such as __call__()
        """
        interp1d.__init__(self, x, y, kind='linear')

    def gradient(self, xi):
        """
        In the equation  y = mx + c
        Return m
        """
        srt = np.absolute(xi - self.x).argsort()
        x0, x1 = self.x[srt[0]], self.x[srt[1]]
        y0, y1 = self.y[srt[0]], self.y[srt[1]]

        return (y1-y0)/(x1-x0)


    def tangent_linear(self, xi):
        y_tl = np.zeros_like(xi)

        for i in range(len(xi)):
            srt = np.absolute(xi[i] - self.x).argsort()
            x0, x1 = self.x[srt[0]], self.x[srt[1]]
            dy0, dy1 = self.y[srt[0]], self.y[srt[1]]

            dydy0 = 1.0 - (xi[i]-x0) / (x1-x0)
            dydy1 = (xi[i]-x0) / (x1-x0)

            y_tl[i] = dydy0 * dy0 + dydy1 * dy1

        return y_tl


    def adjoint_legacy(self, xi, yi):
        """
        Evaluates the value at neighbouring grid points given pos and val
            Arguments
                xi  : (1,n) array
                yi  : (1,n) array
            Returns
                y_adj
        """
        y_adj = np.zeros_like(self.y)

        for i in range(len(xi)):
            srt = np.absolute(xi[i] - self.x).argsort()
            x0, x1 = self.x[srt[0]], self.x[srt[1]]
            y0, y1 = self.y[srt[0]], self.y[srt[1]]

            m = (y1-y0)/(x1-x0)
            c = -m*xi[i] + yi[i]

            y_adj[srt[0]] += m*x0 + c
            y_adj[srt[1]] += m*x1 + c

        return y_adj


    def adjoint(self, xi, dy):
        """
        Evaluates the value at neighbouring grid points given pos and val
            Arguments
                xi  : (1,n) array
                dy  : (1,n) array
            Returns
                y_adj
        """
        y_adj = np.zeros_like(self.y)

        for i in range(len(xi)):
            srt = np.absolute(xi[i] - self.x).argsort()
            x0, x1 = self.x[srt[0]], self.x[srt[1]]

            dydy0 = 1.0 - (xi[i]-x0) / (x1-x0)
            dydy1 = (xi[i]-x0) / (x1-x0)

            y_adj[srt[0]] += dydy0 * dy[i]
            y_adj[srt[1]] += dydy1 * dy[i]

        return y_adj
