import pytest

import numpy as np
from conduction import DiffusionND
from scipy.special import erfc


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_halfspace_cooling(theta):

    def halfspace_cooling(kappa, y, t):
        T = 1.0 - erfc(0.5 * y / np.sqrt(kappa * t))
        return T


    mesh = DiffusionND((0.,), (1.,), (100,), theta=theta)

    kappa = 1.0
    z = mesh.grid_coords[0]

    t1 = 0.01
    t2 = 0.04
    t3 = 0.05

    T_01 = halfspace_cooling(kappa, z, t1)
    T_04 = halfspace_cooling(kappa, z, t2)
    T_05 = halfspace_cooling(kappa, z, t3)


    mesh.boundary_condition("minX", 0.0, flux=False)
    mesh.boundary_condition("maxX", 1.0, flux=False)
    mesh.update_properties(np.full(mesh.npoints, kappa), np.zeros(mesh.npoints))

    # timestep size
    dt = mesh.calculate_dt()


    mesh.temperature.data = 1.0
    Ts_01 = mesh.timestep(steps=int(t1/dt))

    mesh.temperature.data = 1.0
    Ts_04 = mesh.timestep(steps=int(t2/dt))

    mesh.temperature.data = 1.0
    Ts_05 = mesh.timestep(steps=int(t3/dt))


    err_msg = "Halfspace cooling failed"
    np.testing.assert_allclose(T_01, Ts_01, atol=0.05, err_msg=err_msg)
    np.testing.assert_allclose(T_04, Ts_04, atol=0.05, err_msg=err_msg)
    np.testing.assert_allclose(T_05, Ts_05, atol=0.05, err_msg=err_msg)