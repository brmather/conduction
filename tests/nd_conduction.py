
import numpy as np
import conduction
from conduction import ConductionND
from conduction.solver.conduction3d import Conduction3D



minX, minY, minZ = 0., 0., 0.
maxX, maxY, maxZ = 1., 1., 1.
nx, ny, nz = 10, 9, 8


minCoords = np.array([minX, minY, minZ])
maxCoords = np.array([maxX, maxY, maxZ])
res = np.array([nx, ny, nz])

Conduction = {3: Conduction3D}

for n in range(3,4):

    mesh1 = Conduction[n](minCoords[0:n], maxCoords[0:n], res[0:n])
    mesh2 = ConductionND(minCoords[0:n], maxCoords[0:n], res[0:n])


    # material properties

    k = np.ones(mesh2.nn)
    H = np.ones(mesh2.nn)

    mesh1.update_properties(k, H)
    mesh2.update_properties(k, H)


    # BCs

    topBC = 0.0
    bottomBC = 1.0

    mesh1.boundary_condition("minX", bottomBC, flux=False)
    mesh2.boundary_condition("minX", bottomBC, flux=False)

    mesh1.boundary_condition("maxX", topBC, flux=False)
    mesh2.boundary_condition("maxX", topBC, flux=False)


    # solution

    sol1 = mesh1.solve()
    sol2 = mesh2.solve()


    # difference in solution
    print np.abs(sol1 - sol2).max()
