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
from scipy.interpolate import griddata, RegularGridInterpolator

try: range=xrange
except: pass


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


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
     args
    """

    def __init__(self, T, P, *args, **kwargs):

        self.T_range = np.unique(T)
        self.P_range = np.unique(P)

        # convert everything to keyword arguments
        kwdict = kwargs
        for i, arg in enumerate(args):
            key = i
            if key in kwdict:
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg


        keys = sorted(kwdict.keys())
        fields = []
        for key in keys:
            field = kwdict[key]
            fields.append(field)

        field = self.process_tables(*fields)

        for i, key in enumerate(keys):
            kwdict[key] = field[i]


        # save RGI object placeholder
        self.rgi = RegularGridInterpolator((P, T), field[0])
        self.table = kwdict
        self.keys = keys


    def add_fields(self, *args, **kwargs):
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

    def __call__(self, T, P, key=None):
        xi = np.column_stack([P,T])

        if type(key) == type(None):
            values = []
            for key in self.keys:
                self.rgi.values = self.table[key]
                val = self.rgi(xi)
                values.append(val)
        else:
            self.rgi.values = self.table[key]
            values = self.rgi(xi)
        return values


    def process_tables(self, *args):
        """
        Interpolate over NaNs or zero values in grid

        Arguments
        ---------
         phi   : input field(s)

        Returns
        -------
         phi   : processed field(s)
        """
        nT = self.T_range.size
        nP = self.P_range.size
        shape = (np, nT)

        grid_T, grid_P = np.meshgrid(self.T_range, self.P_range)
        xi = np.column_stack([grid_T.ravel(), grid_P.ravel()])

        tables = []
        nargs = len(args)

        for i in range(nargs):
            phi = args[i]
            # find all invalid points
            mask = np.logical_or(phi==0, np.isnan(phi))
            if mask.any():
                imask = np.invert(mask)
                phi = griddata(xi[imask], phi[imask], (grid_T, grid_P), method='cubic')

                # Replace with neighbour if the edges have NaNs
                mask = np.logical_or(phi==0, np.isnan(phi))
                if mask.any():
                    imask = np.invert(mask)
                    phi = griddata(xi[imask], phi[imask], (grid_T, grid_P), method='nearest').ravel()

            tables.append( phi.reshape(shape) )

        return np.array(tables)