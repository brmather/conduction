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

try: range = xrange
except: pass

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD

from ..tools import sum_duplicates
from ..mesh import MeshVariable

class ConductionND(object):
    """
    Implicit 3D steady-state heat equation solver over a structured grid using PETSc
    """

    def __init__(self, minCoord, maxCoord, res):

        dim = len(res)
        extent = np.zeros(dim*2)

        index = 0
        for i in range(0, dim):
            extent[index]   = minCoord[i]
            extent[index+1] = maxCoord[i]
            index += 2


        dm = PETSc.DMDA().create(dim=dim, sizes=res, stencil_width=1, comm=comm)
        dm.setUniformCoordinates(*extent)

        self.dm = dm
        self.lvec = dm.createLocalVector()
        self.gvec = dm.createGlobalVector()
        self.lgmap = dm.getLGMap()

        # Setup matrix sizes
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()
        self.dim = dim
        self.extent = extent

        # include ghost nodes in local domain
        # (minI, maxI), (minJ, maxJ), (minK, maxK) = dm.getGhostRanges()
        ghost_ranges = dm.getGhostRanges()

        n = np.zeros(dim, dtype=PETSc.IntType)
        nn = 1
        for i, (gs, ge) in enumerate(ghost_ranges):
            n[i] = ge - gs
            nn  *= n[i]

        self.n = n[::-1]
        self.nn = nn

        # stencil size
        self.stencil_width = 2*dim + 1

        # local numbering
        self.nodes = np.arange(0, nn, dtype=PETSc.IntType)

        # closure depends on dim
        if dim == 1:
            closure = [(0,-2),(2,0),(1,-1)]
        elif dim == 2:
            closure = [(0,-2), (1,-1), (2,0), (1,-1), (1,-1)]
        elif dim == 3:
            closure = [(0,-2), (1,-1), (1,-1), (2,0), (1,-1), (1,-1), (1,-1)]
        self.closure = self._create_closure_object(closure)


        # interior slices
        self.interior_slice = [None]*dim
        for i in range(0, dim):
            self.interior_slice[i] = slice(1, -1)


        self._initialise_mesh_variables()
        self._initialise_boundary_dictionary()
        self.mat = self._initialise_matrix()
        self._initialise_COO_vectors()

        # thermal properties
        self.diffusivity  = MeshVariable('diffusivity', dm)
        self.heat_sources = MeshVariable('heat_sources', dm)
        self.temperature  = MeshVariable('temperature', dm)

        # right hand side vector
        self.rhs = MeshVariable('rhs', dm)


    def _initialise_COO_vectors(self):

        nn = self.nn
        n = self.n

        index = np.empty(n + 2, dtype=PETSc.IntType)
        index.fill(-1)
        index[self.interior_slice] = self.nodes.reshape(n)
        self.index = index

        self.rows = np.empty((self.stencil_width, nn), dtype=PETSc.IntType)
        self.cols = np.empty((self.stencil_width, nn), dtype=PETSc.IntType)
        self.vals = np.empty((self.stencil_width, nn))



    def _initialise_mesh_variables(self):

        dim = self.dim
        bbox = self.dm.getBoundingBox()

        extent = np.zeros(dim*2)

        index = 0
        for bs, be in bbox:
            extent[index]   = bs
            extent[index+1] = be
            index += 2

        self.extent = extent

        # local coordinates
        self.coords = self.dm.getCoordinatesLocal().array.reshape(-1, dim)

        grid_coords = [None]*dim
        for i in range(0, dim):
            grid_coords[i] = np.unique(self.coords[:,i])

        self.grid_coords = grid_coords


    def _initialise_boundary_dictionary(self):

        coords = self.coords
        grid_coords = self.grid_coords
        dim = self.dim

        minCoords = coords.min(axis=0)
        maxCoords = coords.max(axis=0)

        bbox = self.dm.getBoundingBox()
        sizes = self.dm.getSizes()

        # Setup boundary dictionary
        bc = dict()

        wall = [("minX", "maxX"), ("minY", "maxY"), ("minZ", "maxZ")]

        for i in range(0, dim):
            w0, w1 = wall[i]
            c0, c1 = bbox[i]
            m0, m1 = self.coords[:,i] == c0, self.coords[:,i] == c1
            d0 = d1 = (c1 - c0)/(sizes[i] - 1)

            bc[w0] = {"val": 0.0, "delta": d0, "flux": True, "mask": m0}
            bc[w1] = {"val": 0.0, "delta": d1, "flux": True, "mask": m1}

        self.bc = bc
        self.dirichlet_mask = np.zeros(self.nn, dtype=bool)


    def _initialise_matrix(self, nnz=None):
        """
        There should be no mallocs but we turn off the error just to be sure.
        If there is it will be from users adjusting the BCs.

        Could push zeros into the matrix to allocate all potential entries
        but that would lengthen the build stage.
        """
        if nnz is None:
            nnz = (self.stencil_width, self.dim*2)

        mat = PETSc.Mat().create(comm=comm)
        mat.setType('aij')
        mat.setSizes(self.sizes)
        mat.setLGMap(self.lgmap)
        mat.setPreallocationNNZ(nnz)
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, 0)
        mat.setFromOptions()
        
        return mat


    def _create_closure_object(self, closure):

        n = self.n
        obj = [[0] * self.dim for i in range(self.stencil_width)]

        for i in range(0, self.stencil_width):
            # construct slicing object
            for j in range(0, self.dim):
                start, end = closure[i-j]
                obj[i][j] = slice(start, n[j]+end+2)

        return obj


    def refine(self, fn, axis):
        """
        Pass a function to apply to the x,y,z coordinates on the mesh.
        The domain will be redefined accordingly.

        Notes
        -----
         We do it this way to make sure the domain is balanced across
         processors. Adding new nodes would imbalance the matrix.
        """
        v = self.dm.getCoordinatesLocal()
        coords = v.array.reshape(-1, self.dim)

        coords[:,axis] = fn(coords[:,axis])

        if not np.isfinite(coords).all():
            raise ValueError('This function has created NaNs or Inf numbers')

        v.setArray(coords.ravel())

        self.dm.setCoordinatesLocal(v)

        self._initialise_mesh_variables()
        self._initialise_boundary_dictionary()
        self.mat = self._initialise_matrix()


    def create_meshVariable(self, name):
        return MeshVariable(name, self.dm)


    def update_properties(self, diffusivity, heat_sources):
        """
        Update diffusivity and heat sources
        """


        self.diffusivity[:] = diffusivity
        self.heat_sources[:] = heat_sources


    def boundary_condition(self, wall, val, flux=True):
        """
        Set the boundary conditions on each wall of the domain.
        By default each wall is a Neumann (flux) condition.
        If flux=True, positive val indicates a flux vector towards the centre
        of the domain.

        val can be a vector with the same number of elements as the wall
        """
        wall = str(wall)

        if wall in self.bc:
            self.bc[wall]["val"]  = val
            self.bc[wall]["flux"] = flux
            d = self.bc[wall]

            mask = d['mask']

            if flux:
                self.dirichlet_mask[mask] = False
                self.bc[wall]["val"] /= -d['delta']
            else:
                self.dirichlet_mask[mask] = True

        else:
            raise ValueError("Wall should be one of {}".format(self.bc.keys()))



    def construct_matrix(self, in_place=True, derivative=False):
        """
        Construct the coefficient matrix
        i.e. matrix A in Ax = b

        We vectorise the 7-point stencil for fast matrix insertion.
        An extra border of dummy values around the domain allows for automatic
        Neumann (flux) boundary creation.
        These are stomped on if there are any Dirichlet conditions.

        """

        if in_place:
            mat = self.mat
        else:
            mat = self._initialise_matrix()

        nodes = self.nodes
        nn = self.nn
        n = self.n
        dim = self.dim

        index = self.index

        rows = self.rows
        cols = self.cols
        vals = self.vals

        dirichlet_mask = self.dirichlet_mask

        u = self.diffusivity[:].reshape(n)

        k = np.zeros(n + 2)
        k[self.interior_slice] = u

        for i in range(0, self.stencil_width):
            obj = self.closure[i]

            rows[i] = nodes
            cols[i] = index[obj].ravel()

            distance = np.linalg.norm(self.coords[cols[i]] - self.coords, axis=1)
            distance[distance==0] = 1e-12 # protect against dividing by zero
            delta = 1.0/(2.0*distance**2)

            vals[i] = delta*(k[obj] + u).ravel()


        # Dirichlet boundary conditions (duplicates are summed)
        cols[:,dirichlet_mask] = nodes[dirichlet_mask]
        vals[:,dirichlet_mask] = 0.0

        # zero off-grid coordinates
        vals[cols < 0] = 0.0

        # centre point
        vals[-1] = 0.0
        if derivative:
            vals[-1][dirichlet_mask] = 0.
        else:
            vals[-1][dirichlet_mask] = -1.0


        row = rows.ravel()
        col = cols.ravel()
        val = vals.ravel()


        # mask off-grid entries and sum duplicates
        mask = col >= 0
        row, col, val = sum_duplicates(row[mask], col[mask], val[mask])


        # indptr, col, val = coo_tocsr(row, col, val)
        nnz = np.bincount(row)
        indptr = np.insert(np.cumsum(nnz),0,0)


        mat.assemblyBegin()
        mat.setValuesLocalCSR(indptr.astype(PETSc.IntType), col, val)
        mat.assemblyEnd()

        # set diagonal vector
        diag = mat.getRowSum()
        diag.scale(-1.0)
        mat.setDiagonal(diag)

        return mat


    def construct_rhs(self, in_place=True):
        """
        Construct the right-hand-side vector
        i.e. vector b in Ax = b

        Boundary conditions are grabbed from the dictionary and
        summed to the rhs.
        Be careful of duplicate entries on the corners!!
        """
        if in_place:
            rhs = self.rhs
        else:
            rhs = MeshVariable('rhs', self.dm)
        
        vec = -1.0*self.heat_sources[:]

        for wall in self.bc:
            val  = self.bc[wall]['val']
            flux = self.bc[wall]['flux']
            mask = self.bc[wall]['mask']
            if flux:
                vec[mask] += val
            else:
                vec[mask] = val

        rhs[:] = vec
        return rhs


    def solve(self, matrix=None, rhs=None, solver='bcgs'):
        """
        Construct the matrix A and vector b in Ax = b
        and solve for x

        GMRES method is default
        """
        if matrix is None:
            matrix = self.construct_matrix()
        if rhs is None:
            rhs = self.construct_rhs()
        res = self.temperature

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setType(solver)
        ksp.setOperators(matrix)
        # pc = ksp.getPC()
        # pc.setType('gamg')
        ksp.setFromOptions()
        ksp.setTolerances(1e-20, 1e-50)
        ksp.solve(rhs._gdata, res._gdata)
        # We should hand this back to local vectors
        return res[:]


    def sync(self, vector):
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()


    def gradient(self, vector, **kwargs):

        return np.gradient(vector.reshape(self.n), *self.grid_coords[::-1], **kwargs)


    def heatflux(self):

        T = self.temperature[:]
        k = self.diffusivity[:] * -1
        divT = self.gradient(T)
        for i in range(0, self.dim):
            div = k*divT[i].ravel()
            divT[i] = div

        return divT


    def save_mesh_to_hdf5(self, filename):

        import h5py

        filename = str(filename)
        if not filename.endswith('.h5'):
            filename += '.h5'

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(filename, mode='w')
        ViewHDF5.view(obj=self.dm)
        ViewHDF5.destroy()

        # Every processor is writing the same thing
        f = h5py.File(filename, 'r+')
        f.create_group('topology')
        topo = f['topology']

        # create attributes
        (minX, maxX), (minY, maxY), (minZ, maxZ) = self.dm.getBoundingBox()
        minCoord = np.array([minX, minY, minZ])
        maxCoord = np.array([maxX, maxY, maxZ])
        shape = self.dm.getSizes()

        topo.attrs.create('minCoord', minCoord[::-1])
        topo.attrs.create('maxCoord', maxCoord[::-1])
        topo.attrs.create('shape', np.array(shape)[::-1])

        f.close()


    def save_field_to_hdf5(self, filename, *args, **kwargs):
        """
        Saves data on the mesh to an HDF5 file
         e.g. height, rainfall, sea level, etc.

        Pass these as arguments or keyword arguments for
        their names to be saved to the hdf5 file
        """
        import os.path

        filename = str(filename)
        if not filename.endswith('.h5'):
            filename += '.h5'

        # write mesh if it doesn't exist
        # if not os.path.isfile(file):
        #     self.save_mesh_to_hdf5(file)

        kwdict = kwargs
        for i, arg in enumerate(args):
            key = "arr_{}".format(i)
            if key in kwdict.keys():
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg

        vec = self.gvec.duplicate()

        for key in kwdict:
            val = kwdict[key]
            try:
                vec.setArray(val)
            except:
                self.lvec.setArray(val)
                self.dm.localToGlobal(self.lvec, vec)

            vec.setName(key)

            ViewHDF5 = PETSc.Viewer()
            ViewHDF5.createHDF5(filename, mode='a')
            ViewHDF5.view(obj=vec)
            ViewHDF5.destroy()

        vec.destroy()


    def save_vector_to_hdf5(self, filename, *args, **kwargs):
        """
        Saves vector on the mesh to an HDF5 file
         e.g. heat flux field.

        Pass these as arguments or keyword arguments for
        their names to be saved to the hdf5 file

        Each argument with x,y,z direction tuple
         e.g. Q=(Qx, Qy, Qz)
        """
        import os.path

        filename = str(filename)
        if not filename.endswith('.h5'):
            filename += '.h5'

        kwdict = kwargs
        for i, arg in enumerate(args):
            key = "arr_{}".format(i)
            if key in kwdict.keys():
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg


        # This is a flattened 3xn global vector
        gvec = self.dm.getCoordinates().duplicate()

        for key in kwdict:
            vx, vy, vz = kwdict[key]
            val = np.column_stack([vx, vy, vz]).ravel()

            gvec.assemblyBegin()
            gvec.setValuesLocal(np.arange(val.size, dtype=PETSc.IntType), val)
            gvec.assemblyEnd()
            gvec.setName(key)

            ViewHDF5 = PETSc.Viewer()
            ViewHDF5.createHDF5(filename, mode='a')
            ViewHDF5.view(obj=gvec)
            ViewHDF5.destroy()

        gvec.destroy()