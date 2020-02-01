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

def gradient_ad(df, *varargs, **kwargs):
    """
    Adjoint to the numpy.gradient function.
    
    Interior
    --------
     out[slice1] = (f[slice4] - f[slice2]) / (2. * dx[i])

     dgdf4 = 1/(2*dx)
     dgdf2 = -1/(2*dx)

     df_ad[slice4] += dgdf4*df[axis][slice1]
     df_ad[slice2] += dgdf2*df[axis][slice1]

    Edges
    -----
     out[slice1] = (f[slice2] - f[slice3])/dx

     dgdf2 = 1/dx
     dgdf3 = -1/dx

     df_ad[slice2] += dgdf2*df[slice1]
     df_ad[slice3] += dgdf3*df[slice1]

    """
    import numpy as np

    df = np.asanyarray(df)
    N = df.ndim - 1  # number of dimensions

    axes = kwargs.pop('axis', None)
    if axes is None:
        axes = tuple(list(range(N)))
    else:
        axes = _nx.normalize_axis_tuple(axes, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == len_axes or (n == 1 and np.isscalar(varargs[0])):
        dx = list(varargs)
        for i, distances in enumerate(dx):
            if np.isscalar(distances):
                continue
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("distances must be either scalars or match "
                                 "the length of the corresponding dimension")
            diffx = np.diff(dx[i])
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
        if len(dx) == 1:
            dx *= len_axes
    else:
        raise TypeError("invalid number of arguments")


    edge_order = kwargs.pop('edge_order', 1)
    if kwargs:
        raise TypeError('"{}" are not valid keyword arguments.'.format(
                                                  '", "'.join(kwargs.keys())))
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")


    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = df.dtype.char
    if otype not in ['f', 'd', 'F', 'D', 'm', 'M']:
        otype = 'd'
        
    df_ad = np.zeros_like(df[0], dtype=otype)
        
    for i, axis in enumerate(axes):
        uniform_spacing = np.isscalar(dx[i])

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        # out[slice1] = a * f[slice2] + b * f[slice3] + c * f[slice4]
        # out[slice1] = (f[slice4] - f[slice2]) / (2. * dx[i])

        dgdf4 = 1./(2*dx[i])
        dgdf2 = -1./(2*dx[i])

        df_ad[tuple(slice4)] += dgdf4*df[i][tuple(slice1)]
        df_ad[tuple(slice2)] += dgdf2*df[i][tuple(slice1)]

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = dx[i] if uniform_spacing else dx[i][0]

            # 1D equivalent -- out[0] = (y[1] - y[0]) / (x[1] - x[0])
            # out[slice1] = (y[slice2] - y[slice3]) / dx_0

            dgdf2 = 1./dx_0
            dgdf3 = -1./dx_0

            df_ad[tuple(slice2)] += dgdf2*df[i][tuple(slice1)]
            df_ad[tuple(slice3)] += dgdf3*df[i][tuple(slice1)]

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = dx[i] if uniform_spacing else dx[i][-1]

            # 1D equivalent -- out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
            # out[slice1] = (y[slice2] - y[slice3]) / dx_n

            dgdf2 = 1./dx_n
            dgdf3 = -1./dx_n

            df_ad[tuple(slice2)] += dgdf2*df[i][tuple(slice1)]
            df_ad[tuple(slice3)] += dgdf3*df[i][tuple(slice1)]
        
        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
            
    return df_ad