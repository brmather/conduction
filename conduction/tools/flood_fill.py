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
try: range = xrange
except: pass

def flood_fill(grid, spts):
    """
    Poisson disc sampler in two dimensions.
    This is a flood-fill algorithm for generating points that are
    separated by a minimum radius.

    Arguments
    ---------
     minX, maxX, minY, maxY : float
        coordinates of the domain
     spacing : float
        constant radius to sample across the domain
        every point is guaranteed to be no less than this distance
        from each other
     k : int (default: k=30)
        number of random samples to generate around a single point
        30 generally gives good results
     r_grid : array of floats, optional, shape (height,width)
        support for variable radii
        radius is ignored if an array is given here
     cpts : array of floats, optional, shape (n,2)
        points that must be sampled; useful for irregular boundaries
     spts : array of floats, optional, shape (s,2)
        points used to seed the flood-fill algorithm,
        samples are generated outwards from these seed points

    Returns
    -------
     pts : array of floats, shape (N,2)
        x, y coordinates for each sample point
     cpts_mask : array of bools, shape (N,2)
        boolean array where new points are True and
        cpts are False

    Notes
    -----
     One should aim to sample around 10,000 points, much more than that
     and the algorithm slows rapidly.
    """


    def random_point_around(point, k=7):
        """ Generate neighbouring points """
        P[:] = point # fill with point

        closure = [-1, 0, 0, 1, 0, 1, 1]
        for i in range(k):
            r = closure[i]
            c = closure[-1+i]
            d = closure[-2+i]

            P[i] += [r, c, d]

        return P

    def in_neighbourhood(point):
        """ Checks if point is in the neighbourhood """
        i, j, k = point
        if M[i,j,k]:
            return True
        return False

    def in_limits(point):
        """ Returns True if point is within box """
        return 0 <= point[0] < width and 0 <= point[1] < height and 0 <= point[2] < depth

    def add_point(point, index):
        """ Append point to the points list """
        points.append(point)

        i, j, k = point
        M[i,j,k] = True
        grid[i,j,k] = index


    # Size of the domain
    dim = len(grid.shape)
    n = np.prod(grid.shape)
    width, height, depth = grid.shape


    # Position cells
    P = np.empty((7, dim), dtype=np.int32)
    M = np.zeros((width, height, depth), dtype=bool)
    i, j, k = np.nonzero(grid)
    M[i,j,k] = True


    # Add seed points
    points = []
    if spts is not None:
        spts = spts.reshape(-1,dim)
        for index, pt in enumerate(spts):
            add_point(tuple(pt), index+1)
    else:
        # add a random initial point
        add_point((np.random.uniform(0, width),\
                   np.random.uniform(0, height),\
                   np.random.uniform(0, depth)),\
                   index=1)


    length = len(points)
    while length:
        i = np.random.randint(0,length)
        pt = points.pop(i)
        i, j, k = pt
        index = grid[i,j,k]

        qt = random_point_around(pt)
        for q in qt:
            if in_limits(q) and not in_neighbourhood(q):
                add_point(q, index)

        # re-evaluate length
        length = len(points)

    return grid
