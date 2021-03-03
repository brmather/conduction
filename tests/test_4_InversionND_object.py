import pytest
from conftest import forward_model, adjoint_model
from conftest import load_inv_object as inv

import numpy as np
from numpy import dot, sum, array

from conduction import ConductionND
from conduction.inversion import InvObservation, InvPrior
from conduction import InversionND

def test_vertical_HP_ratios():

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
    dx = array([0.1, -0.1, 0.0])

    fm0 = forward_model(x)
    fm1 = forward_model(x+dx)
    tl  = tangent_linear(x, dx)
    ad  = adjoint_model(x)

    # assert False, ((fm1-fm0) , tl[1],)
    assert abs((fm1-fm0) - tl[1]) < 60, "Finite difference and tangent linear are not similar"
    assert abs(tl[1] - dot(ad[1], dx)) < 6, "Tangent linear and adjoint are not similar"



def test_adjoint_priors(inv):

    k = np.array([3.5, 2.0, 3.2])
    H = np.array([0.5e-6, 1e-6, 2e-6])
    a = np.array([0.3, 0.3, 0.3])
    q0 = 35e-3
    sigma_q0 = 5e-3

    # Inversion variables
    x = np.hstack([k, H, a, [q0]])
    dx = x*0.01

    # Priors
    k_prior = k*1.1
    H_prior = H*1.1
    a_prior = a*1.1
    sigma_k = k*0.1
    sigma_H = H*0.1
    sigma_a = a*0.1

    kp = InvPrior(k_prior, sigma_k)
    Hp = InvPrior(H_prior, sigma_H)
    ap = InvPrior(a_prior, sigma_a)
    q0p = InvPrior(q0, sigma_q0)
    inv.add_prior(k=ap, H=Hp, a=ap, q0=q0p)


    fm0 = forward_model(x, inv)
    fm1 = forward_model(x+dx, inv)
    ad  = adjoint_model(x, inv)

    assert abs(fm1-fm0 - ad[1].dot(dx)) < 2, "Finite difference and adjoint are not similar"


def test_adjoint_observations_heatflux(inv):

    k = np.array([3.5, 2.0, 3.2])
    H = np.array([0.5e-6, 1e-6, 2e-6])
    a = np.array([0.3, 0.3, 0.3])
    q0 = 35e-3
    sigma_q0 = 5e-3

    # Inversion variables
    x = np.hstack([k, H, a, [q0]])
    dx = x*0.01

    q_obs = np.ones(5)*0.03
    sigma_q = q_obs*0.5
    q_coord = np.zeros((5,3))
    q_coord[:,0] = np.random.random(5)*1e3
    q_coord[:,1] = np.random.random(5)*1e3
    q_coord[:,2] = 0.0
    q_coord = q_coord

    qobs = InvObservation(q_obs, sigma_q, q_coord)
    inv.add_observation(q=qobs)


    fm0 = forward_model(x, inv)
    fm1 = forward_model(x+dx, inv)
    ad  = adjoint_model(x, inv)

    assert abs(fm1-fm0 - ad[1].dot(dx)) < 1, "Finite difference and adjoint are not similar"


def test_adjoint_observations_temperature(inv):

    k = np.array([3.5, 2.0, 3.2])
    H = np.array([0.5e-6, 1e-6, 2e-6])
    a = np.array([0.3, 0.3, 0.3])
    q0 = 35e-3
    sigma_q0 = 5e-3

    # Inversion variables
    x = np.hstack([k, H, a, [q0]])
    dx = x*0.01

    T_prior = np.ones(inv.mesh.nn)*400.0
    sigma_T = T_prior*0.01

    Tobs = InvObservation(T_prior, sigma_T, inv.mesh.coords)
    inv.add_observation(T=Tobs)


    fm0 = forward_model(x, inv)
    fm1 = forward_model(x+dx, inv)
    ad  = adjoint_model(x, inv)


    assert abs(fm1-fm0 - ad[1].dot(dx)) < 15000, "Finite difference and adjoint are not similar"
