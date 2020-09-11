import pytest
from conftest import load_multi_mesh as mesh

import numpy as np
from conduction import MeshVariable

def test_mesh_variable_instance(mesh):

    kappa = MeshVariable('kappa', mesh.dm)
    kappa[:] = np.ones(mesh.npoints)
    kappa[0] = 0

    assert kappa[0] == 0