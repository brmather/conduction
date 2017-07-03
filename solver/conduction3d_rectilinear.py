try: range = xrange
except: pass

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
comm = MPI.COMM_WORLD

class Conduction3D(object):
    """
    Implicit 3D steady-state heat equation solver over a structured grid using PETSc
    """

    def __init__(self, minCoord, maxCoord, res):

        minX, minY, minZ = tuple(minCoord)
        maxX, maxY, maxZ = tuple(maxCoord)
        resI, resJ, resK = tuple(res)


        dm = PETSc.DMDA().create(dim=3, sizes=[resI, resJ, resK], stencil_width=1, comm=comm)
        dm.setUniformCoordinates(minX, maxX, minY, maxY, minZ, maxZ)

        self.dm = dm
        self.lvec = dm.createLocalVector()
        self.gvec = dm.createGlobalVector()
        self.rhs = dm.createGlobalVector()
        self.res = dm.createGlobalVector()
        self.lres = dm.createLocalVector()
        self.lgmap = dm.getLGMap()

        # Setup matrix sizes
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()

        Nx, Ny, Nz = dm.getSizes()
        N = Nx*Ny*Nz

        # include ghost nodes in local domain
        (minI, maxI), (minJ, maxJ), (minK, maxK) = dm.getGhostRanges()

        nx = maxI - minI
        ny = maxJ - minJ
        nz = maxK - minK

        self.nx, self.ny, self.nz = nx, ny, nz

        # local numbering
        self.nodes = np.arange(0, nx*ny*nz, dtype=PETSc.IntType)


        self._initialise_mesh_variables()
        self._initialise_boundary_dictionary()
        self.mat = self._initialise_matrix()
        self._initialise_COO_vectors()

        # thermal properties
        self.diffusivity  = None
        self.heat_sources = None
        self.temperature = self.lres.array


    def _initialise_COO_vectors(self):

        nx, ny, nz = self.nx, self.ny, self.nz
        n = nx*ny*nz

        index = np.empty((nz+2, ny+2, nx+2), dtype=PETSc.IntType)
        index.fill(-1)
        index[1:-1,1:-1,1:-1] = self.nodes.reshape(nz,ny,nx)
        self.index = index

        self.rows = np.empty((7,n), dtype=PETSc.IntType)
        self.cols = np.empty((7,n), dtype=PETSc.IntType)
        self.vals = np.empty((7,n))



    def _initialise_mesh_variables(self):

        # local coordinates
        self.coords = self.dm.getCoordinatesLocal().array.reshape(-1,3)

        minX, minY, minZ = self.coords.min(axis=0)
        maxX, maxY, maxZ = self.coords.max(axis=0)

        self.minX, self.maxX = minX, maxX
        self.minY, self.maxY = minY, maxY
        self.minZ, self.maxZ = minZ, maxZ


    def _initialise_boundary_dictionary(self):

        coords = self.coords

        minX, minY, minZ = self.minX, self.minY, self.minZ
        maxX, maxY, maxZ = self.maxX, self.maxY, self.maxZ

        nx, ny, nz = self.nx, self.ny, self.nz

        unique_x = np.unique(coords[:,0])
        unique_y = np.unique(coords[:,1])
        unique_z = np.unique(coords[:,2])

        dminX = unique_x[1] - unique_x[0]
        dminY = unique_y[1] - unique_y[0]
        dminZ = unique_z[1] - unique_z[0]

        dmaxX = unique_x[-1] - unique_x[-2]
        dmaxY = unique_y[-1] - unique_y[-2]
        dmaxZ = unique_z[-1] - unique_z[-2]

        # Setup boundary dictionary
        self.bc = dict()
        self.bc["minX"] = {"val": 0.0, "delta": dminX, "flux": True, "mask": coords[:,0]==minX}
        self.bc["maxX"] = {"val": 0.0, "delta": dmaxX, "flux": True, "mask": coords[:,0]==maxX}
        self.bc["minY"] = {"val": 0.0, "delta": dminY, "flux": True, "mask": coords[:,1]==minY}
        self.bc["maxY"] = {"val": 0.0, "delta": dmaxY, "flux": True, "mask": coords[:,1]==maxY}
        self.bc["minZ"] = {"val": 0.0, "delta": dminZ, "flux": True, "mask": coords[:,2]==minZ}
        self.bc["maxZ"] = {"val": 0.0, "delta": dmaxZ, "flux": True, "mask": coords[:,2]==maxZ}


        self.dirichlet_mask = np.zeros(nx*ny*nz, dtype=bool)


    def _initialise_matrix(self):
        """
        There should be no mallocs but we turn off the error just to be sure.
        If there is it will be from users adjusting the BCs.

        Could push zeros into the matrix to allocate all potential entries
        but that would lengthen the build stage.
        """
        mat = PETSc.Mat().create(comm=comm)
        mat.setType('aij')
        mat.setSizes(self.sizes)
        mat.setLGMap(self.lgmap)
        mat.setPreallocationNNZ((7,6))
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, 0)
        mat.setFromOptions()
        
        return mat


    def refine(self, x_fn=None, y_fn=None, z_fn=None):
        """
        Pass a function to apply to the x,y,z coordinates on the mesh.
        The domain will be redefined accordingly.
        """
        fn = lambda x: x
        if x_fn is None: x_fn = fn
        if y_fn is None: y_fn = fn
        if z_fn is None: z_fn = fn

        v = self.dm.getCoordinatesLocal()
        coords = v.array.reshape(-1,3)

        coords[:,0] = x_fn(coords[:,0])
        coords[:,1] = y_fn(coords[:,1])
        coords[:,2] = z_fn(coords[:,2])

        if not np.isfinite(coords).all():
            raise ValueError('A function has created NaNs or Inf numbers')

        v.setArray(coords.ravel())

        self.dm.setCoordinatesLocal(v)

        self._initialise_mesh_variables()
        self._initialise_boundary_dictionary()
        self.mat = self._initialise_matrix()


    def update_properties(self, diffusivity, heat_sources):
        """
        Update diffusivity and heat sources
        """


        self.diffusivity = self.sync(diffusivity)
        self.heat_sources = self.sync(heat_sources)


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
        nx, ny, nz = self.nx, self.ny, self.nz
        n = nx*ny*nz

        index = self.index

        rows = self.rows
        cols = self.cols
        vals = self.vals

        dirichlet_mask = self.dirichlet_mask

        u = self.diffusivity.reshape(nz,ny,nx)

        k = np.zeros((nz+2, ny+2, nx+2))
        k[1:-1,1:-1,1:-1] = u


        closure = [(0,-2), (1,-1), (1,-1), (2,0), (1,-1), (1,-1), (1,-1)]
        #         N    W    F    S    E    B    C

        for i in range(0, 7):
            rs, re = closure[i]
            cs, ce = closure[-1+i]
            ds, de = closure[-2+i]

            rows[i] = nodes
            cols[i] = index[ds:nz+de+2,rs:ny+re+2,cs:nx+ce+2].ravel()

            distance = np.linalg.norm(self.coords[cols[i]] - self.coords, axis=1)
            distance[distance==0] = 1e-12 # protect against dividing by zero
            delta = 1.0/(2.0*distance**2)

            vals[i] = delta*(k[ds:nz+de+2,rs:ny+re+2,cs:nx+ce+2] + u).ravel()


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
            rhs = self.gvec.duplicate()
        
        vec = -1.0*self.heat_sources.copy()

        for wall in self.bc:
            val  = self.bc[wall]['val']
            flux = self.bc[wall]['flux']
            mask = self.bc[wall]['mask']
            if flux:
                vec[mask] += val
            else:
                vec[mask] = val

        self.lvec.setArray(vec)
        self.dm.localToGlobal(self.lvec, rhs)

        return rhs


    def solve(self, solver='bcgs'):
        """
        Construct the matrix A and vector b in Ax = b
        and solve for x

        GMRES method is default
        """
        matrix = self.construct_matrix()
        rhs = self.construct_rhs()
        res = self.res
        lres = self.lres

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setType(solver)
        ksp.setOperators(matrix)
        # pc = ksp.getPC()
        # pc.setType('gamg')
        ksp.setFromOptions()
        ksp.setTolerances(1e-10, 1e-50)
        ksp.solve(rhs, res)
        # We should hand this back to local vectors
        self.dm.globalToLocal(res, lres)
        return lres.array


    def sync(self, vector):
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()


    def gradient(self, vector):
        # OK for the rectilinear this will take a bit more work
        pass


    def save_mesh_to_hdf5(self, filename):

        import h5py

        filename = str(filename)
        if not filename.endswith('.h5'):
            filename += '.h5'

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(filename, mode='w')
        ViewHDF5.view(obj=self.dm)
        ViewHDF5.destroy()

        if comm.rank == 0:
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


def csr_tocoo(indptr, indices, data):
    """ Convert from CSR to COO sparse matrix format """
    d = np.diff(indptr)
    I = np.repeat(np.arange(0,d.size,dtype='int32'), d)
    return I, indices, data

def coo_tocsr(I, J, V):
    """ Convert from COO to CSR sparse matrix format """
    nnz = np.bincount(I)
    indptr = np.insert(np.cumsum(nnz),0,0)
    return indptr, J, V

def sum_duplicates(I, J, V):
    """
    Sum all duplicate entries in the matrix
    """
    order = np.lexsort((J, I))
    I, J, V = I[order], J[order], V[order]
    unique_mask = ((I[1:] != I[:-1]) |
                   (J[1:] != J[:-1]))
    unique_mask = np.append(True, unique_mask)
    unique_inds, = np.nonzero(unique_mask)
    return I[unique_mask], J[unique_mask], np.add.reduceat(V, unique_inds)