"""Metropolis simulated annealing with individual and fixed-pair translations.

Fixed proposal pairs improve motion of bound e-h pairs. They do not impose
a bond: single-particle proposals move every charge independently. Proposal
partners are held fixed throughout a run, so reverse proposals are symmetric.
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment, minimize
from ewald import real_pair, structure, energy_gradient


def proposal_partners(engine, positions):
    ne = int(np.count_nonzero(engine.layers==0))
    nh = len(positions)-ne
    delta = positions[ne:,None,:]-positions[None,:ne,:]
    f = delta @ engine.invcell
    best = np.full((nh,ne),np.inf)
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            r = (f-np.round(f)+np.array([i,j])) @ engine.cell
            best = np.minimum(best,(r*r).sum(axis=2))
    rows,cols=linear_sum_assignment(best)
    return np.asarray(cols,np.int64)


@njit(cache=True)
def anneal_kernel(positions, partners, temperatures, sweeps_per_temperature,
                  rng_seed, initial_single_step, initial_pair_step,
                  layers, cell, invcell, translations, alpha, cutoff2,
                  gs, c0, cd, area, constant):
    np.random.seed(rng_seed)
    positions=positions.copy()
    ne=2*len(positions)//3
    nh=len(positions)//3
    sc,ss=structure(positions,layers,gs)
    energy=energy_gradient(positions,layers,cell,invcell,translations,alpha,
                           cutoff2,gs,c0,cd,area,constant)[0]
    best_energy=energy
    best=positions.copy()
    initial_energy=energy
    steps=np.array([initial_single_step,initial_pair_step])
    # T, E/Nh, best E/Nh, acceptance single/pair, proposal sizes, drift.
    history=np.zeros((len(temperatures),8))
    dc=np.zeros_like(sc)
    ds=np.zeros_like(ss)
    moved=np.empty(2,np.int64)
    proposed=np.zeros((2,2))
    for it in range(len(temperatures)):
        accepted=np.zeros(2)
        attempted=np.zeros(2)
        for sweep in range(sweeps_per_temperature):
            for attempt in range(len(positions)):
                kind=0 if np.random.random()<.6 else 1
                attempted[kind]+=1
                nm=1
                if kind==0:
                    moved[0]=np.random.randint(len(positions))
                else:
                    h=np.random.randint(nh)
                    moved[0]=partners[h]
                    moved[1]=ne+h
                    nm=2
                displacement=(2*np.random.random(2)-1)*steps[kind]
                for m in range(nm):
                    proposed[m]=positions[moved[m]]+displacement
                de=0.
                for m in range(nm):
                    i=moved[m]
                    for j in range(len(positions)):
                        if j==moved[0] or (nm==2 and j==moved[1]):
                            continue
                        dz=1.0 if layers[i]!=layers[j] else 0.0
                        sign=1.-2.*dz
                        old=real_pair(positions[i]-positions[j],dz,cell,invcell,
                                      translations,alpha,cutoff2)[0]
                        new=real_pair(proposed[m]-positions[j],dz,cell,invcell,
                                      translations,alpha,cutoff2)[0]
                        de+=sign*(new-old)
                dc[:]=0.
                ds[:]=0.
                for m in range(nm):
                    i=moved[m]
                    s=layers[i]
                    for k in range(len(gs)):
                        old=positions[i,0]*gs[k,0]+positions[i,1]*gs[k,1]
                        new=proposed[m,0]*gs[k,0]+proposed[m,1]*gs[k,1]
                        dc[s,k]+=math.cos(new)-math.cos(old)
                        ds[s,k]+=math.sin(new)-math.sin(old)
                for k in range(len(gs)):
                    same=0.
                    for s in range(2):
                        same+=2*(sc[s,k]*dc[s,k]+ss[s,k]*ds[s,k])+dc[s,k]**2+ds[s,k]**2
                    cross=(dc[0,k]*sc[1,k]+sc[0,k]*dc[1,k]+dc[0,k]*dc[1,k]
                           +ds[0,k]*ss[1,k]+ss[0,k]*ds[1,k]+ds[0,k]*ds[1,k])
                    de+=(.5*c0[k]*same-cd[k]*cross)/area
                if de<=0 or np.random.random()<math.exp(-de/temperatures[it]):
                    accepted[kind]+=1
                    for m in range(nm):
                        positions[moved[m]]=proposed[m]
                    sc+=dc
                    ss+=ds
                    energy+=de
                    if energy<best_energy:
                        best_energy=energy
                        best=positions.copy()
        exact=energy_gradient(positions,layers,cell,invcell,translations,alpha,
                              cutoff2,gs,c0,cd,area,constant)[0]
        drift=exact-energy
        energy=exact
        sc,ss=structure(positions,layers,gs)
        rates=accepted/np.maximum(attempted,1.)
        history[it]=np.array([temperatures[it],energy/nh,best_energy/nh,
                              rates[0],rates[1],steps[0],steps[1],drift])
        # Adapt between annealing stages; this is an optimizer, not an
        # equilibrium production chain or a finite-temperature free energy.
        for kind in range(2):
            steps[kind]*=math.exp(1.5*(rates[kind]-.38))
            steps[kind]=min(steps[kind],.3*math.sqrt(area))
    return positions,best,history,initial_energy


def anneal(engine, positions, seed=0, *, stages=30, sweeps=100,
           tmax=.02, tmin=2e-7):
    temperatures=np.geomspace(tmax,tmin,stages)
    partners=proposal_partners(engine,positions)
    a=np.sqrt(engine.area/(np.pi*len(partners)))
    return anneal_kernel(np.asarray(positions,float),partners,temperatures,
                          sweeps,int(seed),.25,.18*a,*engine.args())


def relax_preconditioned(engine,positions,maxiter=20000):
    """An invertible coordinate rescaling for very soft, defective crystals.

    No positions are constrained. Pair COM, relative, and excess-electron
    coordinates have different stiffness scales and are optimized together.
    """
    partners=proposal_partners(engine,positions)
    nh=len(partners); ne=2*nh
    excess=np.setdiff1d(np.arange(ne),partners)
    a=np.sqrt(engine.area/(np.pi*nh))
    com_scale=a**2.5
    electron_scale=a**1.5
    hole=positions[ne:].copy()
    relative=positions[partners]-hole
    f=relative@engine.invcell
    relative=(f-np.round(f))@engine.cell
    y0=np.vstack([(hole+.5*relative)/com_scale,relative,positions[excess]/electron_scale])
    def unpack(y):
        y=y.reshape(-1,2)
        pos=np.empty_like(positions)
        pos[partners]=com_scale*y[:nh]+.5*y[nh:2*nh]
        pos[ne:]=com_scale*y[:nh]-.5*y[nh:2*nh]
        pos[excess]=electron_scale*y[2*nh:]
        return pos
    def fun(y):
        e,g=engine.evaluate(unpack(y))
        grad=np.vstack([com_scale*(g[partners]+g[ne:]),
                        .5*(g[partners]-g[ne:]),electron_scale*g[excess]])
        return e,grad.ravel()
    assert abs(fun(y0.ravel())[0]-engine.evaluate(positions)[0])<1e-8
    opt=minimize(fun,y0.ravel(),jac=True,method="L-BFGS-B",
                 options={"gtol":2e-9,"ftol":1e-15,"maxiter":maxiter,"maxcor":40,"maxls":40})
    pos=engine.wrap(unpack(opt.x))
    e,g=engine.evaluate(pos)
    return pos,dict(energy=float(e),max_force=float(np.abs(g).max()),iterations=int(opt.nit),
                    success=bool(opt.success),message=str(opt.message),
                    preconditioning="invertible COM/relative/excess rescaling",
                    com_scale=float(com_scale),electron_scale=float(electron_scale))
