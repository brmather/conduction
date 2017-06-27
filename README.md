# Conduction

Implicit heat conduction solver on a structured grid written in Python. It interfaces with PETSc to provide highly scalable meshes and solve the steady-state heat equation using direct or iterative methods.

## Dependencies

- Python 2.7 and above
- Numpy 1.9 and above
- Scipy 0.14 and above
- [mpi4py](http://pythonhosted.org/mpi4py/usrman/index.html)
- [petsc4py](https://pythonhosted.org/petsc4py/usrman/install.html)
- [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5) (optional - for saving parallel data)
- Matplotlib (optional - for visualisation)

### PETSc installation

PETSc is used extensively via the Python frontend, petsc4py. It is required that PETSc be configured and installed on your local machine prior to using Quagmire. You can use pip to install petsc4py and its dependencies.

```
pip install [--user] numpy mpi4py
pip install [--user] petsc petsc4py
```

If that fails you must compile these dependencies manually.

## Usage

All of the scripts in the *tests* subdirectory can be run in parallel, e.g.

```
mpirun -np 4 python conduction3d_benchmark.py
```

where the number after the `np` flag specifies the number of processors.
