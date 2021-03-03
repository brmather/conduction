import pytest
from conftest import load_multi_mesh as mesh

import numpy as np
from conduction import MeshVariable

def test_material_properties(mesh):
    n = mesh.npoints

    k = np.ones(n)
    H = np.ones(n)

    mesh.update_properties(k, H)

    assert (mesh.diffusivity[:] == 1).all(), "conductivity assignment failed"
    assert (mesh.heat_sources[:] == 1).all(), "heat sources assignment failed"

def test_boundary_conditions(mesh):

    mask0 = mesh.dirichlet_mask.copy()

    mesh.boundary_condition('minX', 1.0, flux=True)
    mesh.boundary_condition('maxX', 0.0, flux=False)

    mask1 = mesh.dirichlet_mask.copy()

    assert (mask0 != mask1).any(), "dirichlet_mask is not set"


def test_solve_linear_geotherm(mesh):

    wall = ['X', 'Y', 'Z'][mesh.dim-1]

    obj = np.array(mesh.n)//2
    obj = list(obj)
    obj[0] = slice(0,mesh.n[0])
    obj = tuple(obj)

    # update BCs
    mesh.boundary_condition('min{}'.format(wall), 1.0, flux=False)
    mesh.boundary_condition('max{}'.format(wall), 0.0, flux=False)

    # update material properties
    k = np.ones(mesh.npoints)
    H = np.zeros(mesh.npoints)
    mesh.update_properties(k, H)

    geotherm = np.linspace(1.0, 0.0, mesh.n[0])

    # solve temperature
    T = mesh.solve()
    T = T.reshape(mesh.n)
    T_geotherm = T[obj]

    # compare geotherm with temperature
    err_msg = "should be a linear geotherm from 0.0 to 1.0"
    np.testing.assert_allclose(geotherm, T_geotherm, err_msg=err_msg)


def test_continental_geotherm(mesh):

    wall = ['X', 'Y', 'Z'][mesh.dim-1]

    def ss_conduction(z, hr, T0, qm, k, H):
        """ Eqn. 4.31 pp. 259 in Turcotte & Schubert """
        return T0 + qm*z/k + H*hr**2/k * (1.0 - np.exp(-z/hr))

    k = 3.35
    H = 2.65e-6
    hr = 10e3
    qm = 0.03
    q0 = 0.0565
    T0 = 10

    z = np.linspace(0, 35e3, mesh.n[0])
    T_analytic = ss_conduction(z, hr, T0, qm, k, H)

    k_array = np.full(mesh.npoints, k)
    H_array = H*np.exp(-mesh.coords[:,0]/hr)

    mesh.boundary_condition('min{}'.format(wall), T0, flux=False)
    mesh.boundary_condition('max{}'.format(wall), qm, flux=True)
    mesh.update_properties(k_array, H_array)
    T_numeric = mesh.solve()

    obj = np.array(mesh.n)//2
    obj = list(obj)
    obj[0] = slice(0,mesh.n[0])
    obj = tuple(obj)

    T_numeric = T_numeric.reshape(mesh.n)[obj]

    err_msg = "analytic solution and numerical solution do not match"
    np.testing.assert_allclose(T_analytic, T_numeric, atol=80.0, err_msg=err_msg)


def test_nonlinear_conduction(mesh):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    wall = ['X', 'Y', 'Z'][mesh.dim-1]

    def hofmeister1999(k0, T, a):
        return k0*(298.0/T)**a

    k_array = np.full(mesh.npoints, 3.5)
    H_array = np.full(mesh.npoints, 1e-6)
    k0 = k_array.copy()

    mesh.boundary_condition('min{}'.format(wall), 10.0, flux=False)
    mesh.boundary_condition('max{}'.format(wall), 0.03, flux=True)
    mesh.update_properties(k_array, H_array)

    error = np.array(10.0)
    tolerance = 0.1
    converged = False

    for its in range(0, 100):
        T_last = mesh.temperature[:].copy()
        mesh.update_properties(k_array, H_array)

        T = mesh.solve()
        k_array = hofmeister1999(k0, T, 0.33)

        err = np.abs(T_last - T).max()
        comm.Allreduce([err, MPI.DOUBLE], [error, MPI.DOUBLE], op=MPI.MAX)

        if error < tolerance:
            converged = True
            break

    assert converged, "Nonlinear conduction has not converged after {} iterations".format(its)