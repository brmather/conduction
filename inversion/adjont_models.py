"""
Copyright 2017 Ben Mather

This file is part of Conduction <https://git.dias.ie/itherc/conduction/>

Conduction is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Conduction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Conduction.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

def linear(x, self, bc='Z'):
    """
    N-dimensional linear model with flux lower BC
    
    Arguments
    ---------
     x  : [k_list, H_list, q0]
     bc : boundary wall to invert

    Returns
    -------
     cost : scalar
    """
    k_list, H_list = np.array_split(x[:-1], 2)
    q0 = x[-1]
    
    # map to mesh
    k0, H = self.map(k_list, H_list)
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    # solve
    T = self.linear_solve(rhs=rhs)
    q = self.heatflux(T, k0)
    delT = self.gradient(T)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, q0=q0) # priors
    return cost

def linear_ad(x, self, bc='Z'):
    """
    N-dimensional linear model with flux lower BC
    
    Arguments
    ---------
     x  : [k_list, H_list, q0]
     bc : boundary wall to invert

    Returns
    -------
     cost : scalar
     grad : [dk_list, dH_list, dq0]

    """
    k_list, H_list = np.array_split(x[:-1], 2)
    q0 = x[-1]
    
    # map to mesh
    k0, H = self.map(k_list, H_list)
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    # solve
    T = self.linear_solve(rhs=rhs)
    q = self.heatflux(T, k0)
    delT = self.gradient(T)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, q0=q0) # priors
    
    ## AD ##
    dk = np.zeros_like(H)
    dH = np.zeros_like(H)
    dT = np.zeros_like(H)
    dq0 = np.array(0.0)
    
    # priors
    dcdk_list = self.objective_routine_ad(k=k_list)
    dcdH_list = self.objective_routine_ad(H=H_list)
    dcdq0 = self.objective_routine_ad(q0=q0)
    # observations
    dT += self.objective_routine_ad(T=T)

    dq = np.zeros_like(q)
    dq[0] = self.objective_routine_ad(q=q[0])
    
    ddelT = np.zeros_like(delT)
    ddelT[0] = self.objective_routine_ad(delT=delT[0])
    

    dTd = self.gradient_ad(ddelT, T)
    dT += dTd
    
    dTq, dkq = self.heatflux_ad(dq, q, T, k0)
    dT += dTq
    dk += dkq
    
    # solve
    dA, db = self.linear_solve_ad(T, dT)
    
    dk += dA
    dH += -db
    dz = self.grid_delta[-1]
    lowerBC_mask = self.mesh.bc["min"+bc]["mask"]
    dq0 += np.sum(-db[lowerBC_mask]/dz/inv.ghost_weights[lowerBC_mask])
    
    # pack to lists
    dk_list, dH_list = inv.map_ad(dk, dH)
    dk_list += dcdk_list
    dH_list += dcdH_list
    dq0 += dcdq0
    
    dx = np.hstack([dk_list, dH_list, [dq0]])
    
    return cost, dx


def nonlinear(x, self, bc='Z'):
    """
    N-dimensional nonlinear model with flux lower BC
    Implements Hofmeister 1999 temperature-dependent
    conductivity law

    Arguments
    ---------
     x : [k_list, H_list, a_list, q0]

    Returns
    -------
     cost : scalar

    """
    def hofmeister1999(k0, T, a=0.25, c=0.0):
        return k0*(298.0/T)**a + c*T**3

    k_list, H_list, a_list = np.array_split(x[:-1], 3)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a = self.map(k_list, H_list, a_list)
    k = k0.copy()
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    while error > tolerance:
        k_last = k.copy()
        self.mesh.diffusivity[:] = k
        T = self.linear_solve(rhs=rhs) # solve
        k = hofmeister1999(k0, T, a)
        error = np.absolute(k - k_last).max()
        i += 1
        
    q = self.heatflux(T, k)
    delT = self.gradient(T)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, q0=q0) # priors
    return cost


def nonlinear_ad(x, self, bc='Z'):
    """
    N-dimensional nonlinear model with flux lower BC
    Implements Hofmeister 1999 temperature-dependent
    conductivity law

    Arguments
    ---------
     x : [k_list, H_list, a_list, q0]

    Returns
    -------
     cost : scalar
     grad : [dk_list, dH_list, da_list, dq0]
    """
    def hofmeister1999(k0, T, a=0.25, c=0.0):
        return k0*(298.0/T)**a + c*T**3
    k_list, H_list, a_list = np.array_split(x[:-1], 3)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a = self.map(k_list, H_list, a_list)
    k = [k0.copy()]
    T = [None]
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    self.mesh.temperature._gdata.set(0.)
    while error > tolerance:
        self.mesh.diffusivity[:] = k[i]
        # solve
        Ti = self.linear_solve(rhs=rhs)
        ki = hofmeister1999(k0, Ti, a)
        T.append(Ti.copy())
        k.append(ki.copy())
        error = np.absolute(k[-1] - k[-2]).max()
        i += 1

    q = self.heatflux(T[-1], k[-1])
    delT = self.gradient(T[-1])
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T[-1], delT=delT[0]) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, q0=q0) # priors
    
    ## AD ##
    dk = np.zeros_like(H)
    dH = np.zeros_like(H)
    dT = np.zeros_like(H)
    da = np.zeros_like(H)
    dk0 = np.zeros_like(H)
    dq0 = np.array(0.0)
    
    # priors
    dcdk_list = self.objective_routine_ad(k=k_list)
    dcdH_list = self.objective_routine_ad(H=H_list)
    dcda_list = self.objective_routine_ad(a=a_list)
    dcdq0 = self.objective_routine_ad(q0=q0)
    # observations
    dT += self.objective_routine_ad(T=T[-1])

    dq = np.zeros_like(q)
    dq[0] = self.objective_routine_ad(q=q[0])
    
    ddelT = np.zeros_like(delT)
    ddelT[0] = self.objective_routine_ad(delT=delT[0])
    

    dTd = self.gradient_ad(ddelT, T[-1])
    dT += dTd
    
    dTq, dkq = self.heatflux_ad(dq, q, T[-1], k[-1])
    dT += dTq
    dk += dkq
    

    # solve
    for j in range(i):
        dkda = np.log(298.0/T[-1-j])*k0*(298.0/T[-1-j])**a
        dkdk0 = (298.0/T[-1-j])**a
        dkdT = -a*k0/T[-1-j]*(298.0/T[-1-j])**a
        
        dk0 += dkdk0*dk
        dT  += dkdT*dk
        da  += dkda*dk
        print dT.min(), dT.max(), dk.min(), dk.max()
        
        dk.fill(0.0)
        
        self.mesh.diffusivity[:] = k[-1-j]
        dA, db = self.linear_solve_ad(T[-1-j], dT)
        dk += dA
        dH += -db
        dz = self.grid_delta[-1]
        lowerBC_mask = self.mesh.bc["min"+bc]["mask"]
        dq0 += np.sum(-db[lowerBC_mask]/dz/inv.ghost_weights[lowerBC_mask])
        
        dT.fill(0.0)
        
    dk0 += dk
        
    # pack to lists
    dk_list, dH_list, da_list = inv.map_ad(dk0, dH, da)
    dk_list += dcdk_list
    dH_list += dcdH_list
    da_list += dcda_list
    dq0 += dcdq0
    
    dx = np.hstack([dk_list, dH_list, da_list, [dq0]])
    
    return cost, dx

def velocity_nonlinear(x, self, bc='Z'):
    k_list, H_list, a_list, psi_list, B_list = np.array_split(x[:-1], 5)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a, psi, B = self.map(k_list, H_list, a_list, psi_list, B_list)
    k = k0.copy()
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    while error > tolerance:
        k_last = k.copy()
        self.mesh.diffusivity[:] = k
        T = self.linear_solve(rhs=rhs) # solve
        k = hofmeister1999(k0, T, a)
        error = np.absolute(k - k_last).max()
        i += 1
    print("{} iterations".format(i))
        
    q = self.heatflux(T, k)
    delT = self.gradient(T)
    rho, Vsp, dVspdT = self.lookup_velocity()
    P = rho*np.abs(self.mesh.coords[:,-1])*9.806*1e-5
    
    # attenuation
    Q = attenuation(Vsp, P, psi, B)
    Vs = Vsp * (1.0 - 0.5*(1.0/np.tan(np.pi*0.26/2.0))*Q)
    self.Vsp = Vsp
    self.Vs = Vs
    self.Q = Q
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T, delT=delT[0], Vs=Vs) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, psi=psi_list, B=B_list, q0=q0) # priors
    return cost

def velocity_nonlinear_ad(x, self, bc='Z'):
    k_list, H_list, a_list, psi_list, B_list = np.array_split(x[:-1], 5)
    q0 = x[-1]
    
    # map to mesh
    k0, H, a, psi, B = self.map(k_list, H_list, a_list, psi_list, B_list)
    k = [k0.copy()]
    T = [None]
    
    self.mesh.update_properties(k0, H)
    self.mesh.boundary_condition("max"+bc, 298.0, flux=False)
    self.mesh.boundary_condition("min"+bc, q0, flux=True)
    rhs = self.mesh.construct_rhs()
    
    error = 10.
    tolerance = 1e-5
    i = 0
    self.mesh.temperature._gdata.set(0.)
    while error > tolerance:
        self.mesh.diffusivity[:] = k[i]
        # solve
        Ti = self.linear_solve(rhs=rhs)
        ki = hofmeister1999(k0, Ti, a)
        T.append(Ti.copy())
        k.append(ki.copy())
        error = np.absolute(k[-1] - k[-2]).max()
        i += 1
    print("{} iterations".format(i))

    q = self.heatflux(T[-1], k[-1])
    delT = self.gradient(T[-1])
    rho, Vsp, dVspdT = self.lookup_velocity()
    P = rho*np.abs(self.mesh.coords[:,-1])*9.806*1e-5
    
    # attenuation
    Q = attenuation(Vsp, P, psi, B)
    Vs = Vsp * (1.0 - 0.5*(1.0/np.tan(np.pi*0.26/2.0))*Q)
    
    cost = 0.0
    cost += self.objective_routine(q=q[0], T=T[-1], delT=delT[0], Vs=Vs) # observations
    cost += self.objective_routine(k=k_list, H=H_list, a=a_list, q0=q0) # priors
    
    ## AD ##
    dk = np.zeros_like(H)
    dH = np.zeros_like(H)
    dT = np.zeros_like(H)
    da = np.zeros_like(H)
    dk0 = np.zeros_like(H)
    dq0 = np.array(0.0)
    dpsi = np.zeros_like(H)
    dB = np.zeros_like(H)
    dQ = np.zeros_like(H)
    
    # priors
    dcdk_list = self.objective_routine_ad(k=k_list)
    dcdH_list = self.objective_routine_ad(H=H_list)
    dcda_list = self.objective_routine_ad(a=a_list)
    dcdpsi_list = self.objective_routine_ad(psi=psi_list)
    dcdB_list = self.objective_routine_ad(B=B_list)
    dcdq0 = self.objective_routine_ad(q0=q0)
    # observations
    dT += self.objective_routine_ad(T=T[-1])

    dq = np.zeros_like(q)
    dq[0] = self.objective_routine_ad(q=q[0])
    
    ddelT = np.zeros_like(delT)
    ddelT[0] = self.objective_routine_ad(delT=delT[0])
    
    dVs = self.objective_routine_ad(Vs=Vs)
    dVsdQ = -Vsp*0.5*(1.0/np.tan(np.pi*0.26/2.0))
    dVsdT = dVspdT*(1.0 - 0.5*(1.0/np.tan(np.pi*0.26/2.0))*Q)
    dQdpsi, dQdB = attenuation_ad(Vsp, P, psi, B)
    
    dQ += dVsdQ*dVs
    dT += dVsdT*dVs
    dpsi += dQdpsi*dQ
    dB += dQdB*dQ
    

    dTd = self.gradient_ad(ddelT, T[-1])
    dT += dTd
    
    dTq, dkq = self.heatflux_ad(dq, q, T[-1], k[-1])
    dT += dTq
    dk += dkq
    

    # solve
    for j in xrange(i):
        dkdT, dkdk0, dkda = hofmeister1999_ad(T[-1-j], k0, a)
        
        dk0 += dkdk0*dk
        dT  += dkdT*dk
        da  += dkda*dk
        
        dk.fill(0.0)
        

        self.mesh.diffusivity[:] = k[-1-j]
        dA, db = self.linear_solve_ad(T[-1-j], dT)

        dk += dA
        dH += -db
        dz = self.grid_delta[-1]
        lowerBC_mask = self.mesh.bc["min"+bc]["mask"]
        dq0 += np.sum(-db[lowerBC_mask]/dz/inv.ghost_weights[lowerBC_mask])
        
        dT.fill(0.0)
        
    dk0 += dk
        
    # pack to lists
    dk_list, dH_list, da_list, dpsi_list, dB_list = inv.map_ad(dk0, dH, da, dpsi, dB)
    dk_list += dcdk_list
    dH_list += dcdH_list
    da_list += dcda_list
    dpsi_list += dcdpsi_list
    dB_list += dcdB_list
    dq0 += dcdq0
    
    dx = np.hstack([dk_list, dH_list, da_list, dpsi_list, dB_list, [dq0]])
    
    return cost, dx