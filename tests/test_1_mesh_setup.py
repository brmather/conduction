import pytest
from conftest import load_multi_mesh as mesh

import numpy as np
from conduction import ConductionND


def test_1D_mesh():
    minX, maxX = 0.0, 1.0
    nx = 30

    mesh = ConductionND((minX,), (maxX,), (nx,))

def test_2D_mesh():
    minX, maxX = 0.0, 1.0
    minY, maxY = 0.0, 1.0
    nx, ny = 30, 30

    mesh = ConductionND((minX,minY), (maxX, maxY), (nx,ny))


def test_3D_mesh():
    minX, maxX = 0.0, 1.0
    minY, maxY = 0.0, 1.0
    minZ, maxZ = 0.0, 1.0
    nx, ny, nz = 30, 30, 30

    mesh = ConductionND((minX, minY, minZ), (maxX, maxY, maxZ), (nx, ny, nz))
