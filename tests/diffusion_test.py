
import numpy as np
import matplotlib.pyplot as plt
from conduction import DiffusionND

minX, maxX = 0.0, 1.0
minY, maxY = 0.0, 1.0
nx, ny = 20, 20

mesh = DiffusionND((minX, minY), (maxX, maxY), (nx, ny), theta=1.0)

# populate fields
diffusivity = np.ones(mesh.nn)
heat_sources = np.ones(mesh.nn)
mesh.update_properties(diffusivity, heat_sources)

# set initial conditions
mesh.temperature[:] = np.zeros(mesh.nn)

# set boundary conditions
mesh.boundary_condition("minY", 1.0, flux=False)
mesh.boundary_condition("maxY", 0.0, flux=False)

# solve timesteps
dt = mesh.calculate_dt()
print("Timestep size = {:e}".format(dt))


nsteps = 500
T = mesh.timestep(nsteps, dt=dt)


# plot results
fig = plt.figure()
ax1 = fig.add_subplot(111)
im1 = ax1.imshow(T.reshape(mesh.n), origin='lower', vmin=0, vmax=1)
fig.colorbar(im1)
plt.show()