from numpy import dot, sum, array

def forward_model(x):
    r = x

    # normalise ratios
    rN = r/sum(r)

    nc = rN*N
    Z = nc*hz
    qs = sum(H*Z)
    c = sum((qs - qobs)**2/sigma_qobs**2)
    return c

def tangent_linear(x, dx):
    r = x
    dr = dx

    # normalise ratios
    rN = r/sum(r)
    drNdr = 1.0/sum(r)
    drNdrsum = -r/sum(r)**2
    drN = drNdr*dr + drNdrsum*sum(dr)

    nc = rN*N
    dncdrN = N
    dnc = dncdrN*drN

    Z = nc*hz
    dZdnc = hz
    dZ = dZdnc*dnc

    qs = sum(H*Z)
    dqsdZ = H
    dqs = sum(dqsdZ*dZ)

    c = sum((qs - qobs)**2/sigma_qobs**2)
    dcdqs = (2.0*qs - 2.0*qobs)/sigma_qobs**2
    dc = dcdqs*dqs
    return c, dc

def adjoint_model(x, df=1.0):
    r = x

    # normalise ratios
    rN = r/sum(r)

    nc = rN*N
    Z = nc*hz
    qs = sum(H*Z)
    c = sum((qs - qobs)**2/sigma_qobs**2)

    # ADJOINT PART
    dcdqs = (2.0*qs - 2.0*qobs)/sigma_qobs**2
    dqs = dcdqs*df

    dqsdZ = H
    dZ = dqsdZ*dqs

    dZdnc = hz
    dnc = dZdnc*dZ

    dncdrN = N
    drN = dncdrN*dnc

    drNdr = 1.0/sum(r)
    drNdrsum = -r/sum(r)**2
    dr = drNdr*drN
    drsum = drNdrsum*drN

    drsumdr = 1.0/N
    # dr += drsumdr*drsum
    dr += sum(drsum)

    return c, dr

# global variables
N = 10
H = array([5.0, 2.0, 1.0])
hz = 0.25

# cost function
qobs = 1.0
sigma_qobs = 0.1

# input vectors
x = array([0.2, 0.5, 0.3])
dx = array([0.1, -0.2, 0.0])

fm0 = forward_model(x)
fm1 = forward_model(x+dx)
tl  = tangent_linear(x, dx)
ad  = adjoint_model(x)

print("finite difference {}".format(fm1 - fm0))
print("tangent linear {}".format(tl[1]))
print("adjoint model {}".format(dot(ad[1], dx)))