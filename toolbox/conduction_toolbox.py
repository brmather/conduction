
# coding: utf-8




# variables
upperBC = 298.0 # Kelvin
lowerBC = 0.03 # W/m2

minX, maxX = 0.0, 50000.0
minY, maxY = -35000.0, 0.0



import numpy as np
import csv
import time
from scipy.ndimage import imread
from conductionNd_serial import ConductionND
try: range=xrange
except: pass



matfile = 'geology.png'
mat_img = imread(matfile, flatten=True)
mat_img = np.flipud(mat_img)
resY, resX = mat_img.shape

mat_index = np.unique(mat_img)[::-1]


lith_name = []
index = []
k_lith = []
H_lith = []


propfile = 'material_properties.csv'
with open(propfile, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    reader.next()
    for row in reader:
        lith_name.append(row[0])
        index.append(int(row[1]))
        k_lith.append(float(row[2]))
        H_lith.append(float(row[3]))


if len(index) != len(mat_index):
    msg = "number of lithologies in {} and {} must be identical"
    raise ValueError(msg.format(matfile, propfile))



# Map properties to mesh
k_field = np.zeros_like(mat_img, dtype=np.float)
H_field = np.zeros_like(mat_img, dtype=np.float)
M_field = np.zeros_like(mat_img, dtype=np.int)

fmt = "{:2} mapped {:15} with k = {:.2f} W/m/K,  H = {:.2f} uW/m3"

for i in range(0, len(index)):
    mask = mat_img == mat_index[i]
    M_field[mask] = index[i]
    k_field[mask] = k_lith[i]
    H_field[mask] = H_lith[i]
    print(fmt.format(index[i], lith_name[i], k_lith[i], H_lith[i]*1e6))
    
k_field = k_field.ravel()
H_field = H_field.ravel()
M_field = M_field.ravel()




# Setup mesh
mesh = ConductionND((minX,minY), (maxX,maxY), (resX, resY))





# update conductivity and heat production fields
mesh.update_properties(k_field, H_field)

# set boundary conditions
mesh.boundary_condition('maxY', upperBC, flux=False)
mesh.boundary_condition('minY', lowerBC, flux=True)





# solve
walltime = time.time()
T = mesh.solve()
qy, qx = mesh.heatflux()
print("\nTemperature solved in {:.1f} s\n".format(time.time()-walltime))


# save to disk
print("Saving table...")
fmt = '%13.3f,%13.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f'
header = 'x,y,T(C),k(W/m/K),H(uW/m3),HF_x(mW/m2),HF_y(mW/m2)'
out = np.hstack([mesh.coords, T.reshape(-1,1)-273.14,\
                 k_field.reshape(-1,1), H_field.reshape(-1,1)*1e6,\
                 qx.reshape(-1,1)*1e3, qy.reshape(-1,1)*1e3])
np.savetxt('out.csv', out, delimiter=',', fmt=fmt, header=header)


import matplotlib.pyplot as plt
def plot_figure(field, title, cmap):
    print("Saving {} figure...".format(title))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(field.reshape(resY,resX), cmap=cmap,\
                     extent=(minX, maxX, minY, maxY), origin='lower')
    ax1.set_title(title)
    fig.colorbar(im1)
    fig.savefig(title+'.png', bbox_inches='tight')
    plt.clf()

plot_figure(M_field, "material_map", 'Greys')
plot_figure(T-273.14, "temperature", 'jet') # Celsius
plot_figure(k_field, "conductivity", 'BuPu') # W/m/K
plot_figure(H_field*1e6, "heat_production", 'Reds') # uW/m3
plot_figure(qx*1e3, "heatflux_x", 'plasma') # mW/m2
plot_figure(qy*1e3, "heatflux_y", 'plasma') # mW/m2
