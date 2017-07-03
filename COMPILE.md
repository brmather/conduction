# Compile dependencies

These are instructions to compiling dependencies on your computer. Most of these will be available through regular software repositories (e.g. `pip`, `apt`, `dnf`, `homebrew`) but these may be outdated. We recommend compiling from the source you encounter compatibility issues.

## MPI

Download the latest stable [openmpi](https://www.open-mpi.org/software/ompi)/[mpich](https://www.mpich.org/downloads) release and extract it anywhere. Navigate inside the extracted folder from the terminal and run the following commands:

    ./configure --enable-shared --prefix=/path/to/install-dir
    make all
    make install
    make check

Now export the path

    export MPI_DIR=/path/to/install-dir
    export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MPI_DIR/bin:$PATH

## HDF5

It is imperitive that you have the parallel version of HDF5. Obtain the latest version of [HDF5](https://support.hdfgroup.org/HDF5/release/obtainsrc518.html) and extract it anywhere. You must link your installed MPI path as follows:

    CC=$MPI_DIR/bin/mpicc FC=$MPI_DIR/bin/mpifort ./configure --enable-shared --enable-parallel --enable-fortran --prefix=/path/to/install-dir
    make all
    make install
    make check

Now export the path

    export HDF5_DIR=/path/to/install-dir
    export LD_LIBRARY_PATH=$HDF5_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$HDF5_DIR/bin:$PATH

## PETSc

Download the latest version of [PETSc](https://www.mcs.anl.gov/petsc/download/index.html) and extract it anywhere. PETSc can download and compile a number of useful external packages automatically during the configure stage.

    ./configure MPI-DIR=$MPI_DIR --with-mpi-dir=$MPI_DIR --with-hdf5-dir=$HDF5_DIR --with-shared-libraries --download-fblaslapack --download-scalapack --download-triangle --download-mumps --download-chaco --download-ctetgen --download-hypre --prefix=/path/to/install-dir

Follow the make instructions from the command line once configuration is complete. This will install PETSc and run some tests to ensure it is functioning properly.


## Install Python bindings

We recommend doing this using `pip`,

    [sudo] apt-get install pip

or equivalent on other operating systems.
`pip` can be instructed to install to your home directory with the `--user` flag, e.g. `pip install --user numpy`. You can also upgrade an existing package by passing the `--upgrade` flag.

### mpi4py

    CC=$MPI_DIR/bin/mpicc MPICC=$MPI_DIR/bin/mpicc MPI_DIR=$MPI_DIR pip install mpi4py

### h5py

    CC=$MPI_DIR/bin/mpicc HDF5_DIR=$HDF5_DIR HDF5_MPI="on" pip install h5py

### petsc4py

    PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH pip install petsc4py

