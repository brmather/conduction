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
from scipy.interpolate import griddata

try: range=xrange
except: pass

class PerplexTable(object):
    """
    Import tables produced by Perplex for fast lookup using a KDTree
    Temperature and pressure are always constant and thus define our
    'mesh', then we place fields on top of this.

    Multiple mesh variables (fields) can be added for a specific
    lithology index. All fields are returned when called.

    Methods
    -------
     add_field  : add new field(s) for a given lithology index
     __call__   : return field(s) at given T-P
    
    Arguments
    ---------
     T  : shape(n,) temperature
     P  : shape(n,) pressure
    """

    def __init__(self, T, P):
        from scipy.spatial import cKDTree
        coords = np.column_stack([T, P])
        self.tree = cKDTree(coords)
        self.table = dict()
        self.ncol = 1
        self.nfield = 0

        self.T_range = np.unique(T)
        self.P_range = np.unique(P)

    def add_field(self, field, index):
        """
        Add new field(s) to the mesh for a given lithology index
        
        Arguments
        ---------
         field  : (ncol, n) Vp, Vs, rho, etc.
         index  : int
        """
        if index not in self.table:
            self.nfield += 1
        # field = self.process_tables(field)
        ncol = 1
        if field.ndim > 1:
            ncol = field.shape[1]

        self.ncol = ncol
        self.table[index] = field

    def __call__(self, T, P, index):
        xi = np.column_stack([T,P])
        d, idx = self.tree.query(xi)
        return self.table[index][idx]


    def process_tables(self, phi):
        """
        Interpolate over NaNs or zero values in grid

        Arguments
        ---------
         phi   : input field(s)

        Returns
        -------
         phi   : processed field(s)
        """
        grid_T, grid_P = np.meshgrid(self.T_range, self.P_range)
        xi = self.tree.data

        # convert to column vector (if not already)
        phi = phi.reshape(-1, ncol)

        for i in range(ncol):
            # find all invalid points
            mask = np.logical_or(phi[:,i]==0, np.isnan(phi[:,i]))
            if mask.any():
                imask = np.invert(mask)
                phi[:,i] = griddata(xi[imask], phi[:,i][imask], (grid_T, grid_P), method='cubic').ravel()

                # Replace with neighbour if the edges have NaNs
                mask = np.logical_or(phi[:,i]==0, np.isnan(phi[:,i]))
                if mask.any():
                    imask = np.invert(mask)
                    phi[:,i] = griddata(xi[imask], phi[:,i][imask], (grid_T, grid_P), method='nearest').ravel()

        phi = phi.astype(np.float16)

        if ncol == 1:
            return phi.ravel()
        else:
            return phi