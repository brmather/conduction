import pytest

def test_numpy_import():
	import numpy

def test_scipy_import():
	import scipy

def test_conduction_modules():
	from conduction import ConductionND
	from conduction import DiffusionND
	from conduction import InversionND
	from conduction import RegularGridInterpolator

def test_mpi4py_import():
	from mpi4py import MPI
	comm = MPI.COMM_WORLD

def test_petsc4py_import():
	from petsc4py import PETSc
	mat = PETSc.Mat().create()

def test_h5py_import():
	import h5py