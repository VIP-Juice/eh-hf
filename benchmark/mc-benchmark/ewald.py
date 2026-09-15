"""Classical 2D-periodic Coulomb bilayer, with each layer's G=0 removed.

Cartesian coordinates and cell ROW vectors are in units d=1. Energies are
in E_d=e^2/(4*pi*epsilon*d). This module has no rigid-exciton constraint.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.special import erfc, erfcx
from scipy.optimize import minimize

SQRTPI = math.sqrt(math.pi)


def lattice(phase: str, ratio: float, repeats: int = 1):
    area = np.pi * ratio**2
    if phase == "checkerboard":
        cell = np.sqrt(area) * np.eye(2)
        offset = np.array([0.5, 0.5])
    elif phase == "honeycomb":
        ell = np.sqrt(2 * area / np.sqrt(3))
        cell = ell * np.array([[1., 0.], [0.5, np.sqrt(3)/2]])
        offset = np.array([1/3, 1/3])
    else:
        raise ValueError(phase)
    origins = np.array([[i, j] for i in range(repeats)
                        for j in range(repeats)]) @ cell
    nh = len(origins)
    # The first nh electrons initially sit above the holes.
    positions = np.vstack([origins, origins + offset @ cell, origins])
    layers = np.r_[np.zeros(2*nh, dtype=np.int64), np.ones(nh, dtype=np.int64)]
    return repeats*cell, positions, layers


@njit(cache=True)
def real_pair(delta, dz, cell, invcell, translations, alpha, cutoff2):
    frac = delta @ invcell
    frac -= np.floor(frac + 0.5)
    delta = frac @ cell
    value = 0.
    gx = gy = 0.
    for t in translations:
        x = delta[0] + t[0]
        y = delta[1] + t[1]
        rr = x*x + y*y + dz*dz
        if rr > cutoff2:
            continue
        if rr < 1e-24:
            return 1e100, 0., 0.
        r = math.sqrt(rr)
        c = math.erfc(alpha*r)
        value += c/r
        f = -c/(rr*r) - 2*alpha/SQRTPI*math.exp(-alpha*alpha*rr)/rr
        gx += f*x
        gy += f*y
    return value, gx, gy


@njit(cache=True)
def structure(positions, layers, gs):
    sc = np.zeros((2, len(gs)))
    ss = np.zeros_like(sc)
    for i in range(len(positions)):
        for k in range(len(gs)):
            angle = positions[i, 0]*gs[k, 0] + positions[i, 1]*gs[k, 1]
            sc[layers[i], k] += math.cos(angle)
            ss[layers[i], k] += math.sin(angle)
    return sc, ss


@njit(cache=True)
def energy_gradient(positions, layers, cell, invcell, translations, alpha,
                    cutoff2, gs, c0, cd, area, constant):
    energy = constant
    grad = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i):
            dz = 1.0 if layers[i] != layers[j] else 0.0
            sign = 1. - 2.*dz
            value, gx, gy = real_pair(positions[i]-positions[j], dz, cell,
                                     invcell, translations, alpha, cutoff2)
            energy += sign*value
            grad[i, 0] += sign*gx
            grad[i, 1] += sign*gy
            grad[j, 0] -= sign*gx
            grad[j, 1] -= sign*gy
    sc, ss = structure(positions, layers, gs)
    for k in range(len(gs)):
        energy += (0.5*c0[k]*(sc[0,k]**2+ss[0,k]**2+sc[1,k]**2+ss[1,k]**2)
                   -cd[k]*(sc[0,k]*sc[1,k]+ss[0,k]*ss[1,k]))/area
    for i in range(len(positions)):
        s = layers[i]
        for k in range(len(gs)):
            angle = positions[i, 0]*gs[k, 0] + positions[i, 1]*gs[k, 1]
            co, si = math.cos(angle), math.sin(angle)
            f = (c0[k]*(co*ss[s,k]-si*sc[s,k])
                 -cd[k]*(co*ss[1-s,k]-si*sc[1-s,k]))/area
            grad[i] += f*gs[k]
    return energy, grad


class Ewald:
    def __init__(self, cell, layers, *, split=4.0, accuracy=6.0):
        self.cell = np.asarray(cell, float)
        self.layers = np.asarray(layers, np.int64)
        self.invcell = np.linalg.inv(self.cell)
        self.area = abs(np.linalg.det(self.cell))
        heights = self.area/np.linalg.norm(self.cell, axis=1)
        self.alpha = split/min(heights)
        self.accuracy = accuracy
        self.split = split
        radius = accuracy/self.alpha
        self.cutoff2 = radius**2
        # Wrapped fractional differences lie in [-1/2,1/2]^2. The plane
        # spacing bound encloses every translation within the radial cutoff.
        bounds = np.ceil(radius*np.linalg.norm(self.invcell, axis=0)+.5).astype(int)
        self.translations = np.array([[i, j] for i in range(-bounds[0], bounds[0]+1)
                                      for j in range(-bounds[1], bounds[1]+1)]) @ self.cell
        reciprocal = 2*np.pi*self.invcell.T
        gmax = 2*self.alpha*accuracy
        gb = np.ceil(gmax*np.linalg.norm(self.cell,axis=1)/(2*np.pi)).astype(int)
        gs = np.array([[i,j] for i in range(-gb[0],gb[0]+1)
                       for j in range(-gb[1],gb[1]+1) if i or j]) @ reciprocal
        mag = np.linalg.norm(gs,axis=1)
        self.gs = gs[mag <= gmax]
        mag = mag[mag <= gmax]
        x = mag/(2*self.alpha)
        self.c0 = 2*np.pi/mag*erfc(x)
        # Integral representation of the SAME splitting as supplement S1-S3.
        # erfcx avoids overflow in exp(G*d)*erfc(x+alpha*d).
        plus = np.exp(-x*x-self.alpha**2)*erfcx(x+self.alpha)
        minus = np.exp(-mag)*erfc(x-self.alpha)
        self.cd = np.pi/mag*(plus+minus)
        ne, nh = np.bincount(self.layers, minlength=2)
        v0 = 2*SQRTPI/self.alpha
        vd = 2*np.pi*(np.exp(-self.alpha**2)/(self.alpha*SQRTPI)-erfc(self.alpha))
        background = -(v0*(ne*ne+nh*nh)-2*vd*ne*nh)/(2*self.area)
        self_term = -self.alpha/SQRTPI*(ne+nh)
        # Real interactions of a particle with its nonzero lattice images.
        tr = np.linalg.norm(self.translations,axis=1)
        tr = tr[(tr>1e-12)&(tr<=radius)]
        self_images = .5*(ne+nh)*np.sum(erfc(self.alpha*tr)/tr)
        self.constant = float(background+self_term+self_images)

    def args(self):
        return (self.layers, self.cell, self.invcell, self.translations,
                self.alpha, self.cutoff2, self.gs, self.c0, self.cd,
                self.area, self.constant)

    def evaluate(self, positions):
        return energy_gradient(np.asarray(positions,float), *self.args())

    def wrap(self, positions):
        return (np.asarray(positions) @ self.invcell % 1.) @ self.cell

    def relax(self, positions, *, gtol=2e-10, maxiter=6000):
        shape = positions.shape
        def fun(x):
            e,g = self.evaluate(x.reshape(shape))
            return e,g.ravel()
        result = minimize(fun, np.asarray(positions).ravel(), jac=True,
                          method="L-BFGS-B", options={"gtol":gtol,"ftol":1e-15,
                          "maxiter":maxiter,"maxls":40,"maxcor":30})
        pos = self.wrap(result.x.reshape(shape))
        e,g = self.evaluate(pos)
        return pos, {"energy":float(e), "max_force":float(np.abs(g).max()),
                     "iterations":int(result.nit), "success":bool(result.success),
                     "message":str(result.message)}


def hessian(engine, positions, step=2e-4):
    flat = positions.ravel().copy()
    h = np.empty((len(flat),len(flat)))
    for j in range(len(flat)):
        plus,minus=flat.copy(),flat.copy()
        plus[j]+=step
        minus[j]-=step
        h[:,j]=(engine.evaluate(plus.reshape(-1,2))[1].ravel()
                -engine.evaluate(minus.reshape(-1,2))[1].ravel())/(2*step)
    return (h+h.T)/2


def crystal_energy(phase, ratio, repeats=1, **kwargs):
    cell,pos,layers=lattice(phase,ratio,repeats)
    return Ewald(cell,layers,**kwargs).evaluate(pos)[0]/(len(layers)/3)
