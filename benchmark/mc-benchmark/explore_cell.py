#!/usr/bin/env python3
"""Allow cell shape and internal coordinates to relax at fixed density.

Added because free-charge MC and Hessians expose a failure of the ideal
two-branch ground-state assumption near the published crossing.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, brentq
from ewald import Ewald, lattice
from run_benchmark import save_json, write_csv, strain_hessian

ROOT=Path(__file__).resolve().parent
LAYERS=np.array([0,0,1],dtype=np.int64)


def reduced_angle(cell):
    """Acute angle of a Gauss-reduced primitive basis (remove basis ambiguity)."""
    a,b=np.array(cell,float)
    for _ in range(100):
        if b@b<a@a:
            a,b=b,a
        multiple=np.rint((a@b)/(a@a))
        if multiple==0:
            break
        b=b-multiple*a
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def unpack(x,ratio):
    u,v=x[:2]
    cell=np.sqrt(np.pi)*ratio*np.array([[np.exp(u),0.],[v,np.exp(-u)]])
    frac=np.array([[0.,0.],x[2:4],x[4:6]])
    return cell,frac@cell


def initial(phase,ratio):
    c,p,l=lattice(phase,ratio)
    scale=np.sqrt(np.pi)*ratio
    f=p@np.linalg.inv(c)
    return np.r_[np.log(c[0,0]/scale),c[1,0]/scale,f[1],f[2]]


def energy(x,ratio):
    cell,p=unpack(x,ratio)
    return Ewald(cell,LAYERS).evaluate(p)[0]


def objective(x,ratio):
    cell,p=unpack(x,ratio)
    e,g=Ewald(cell,LAYERS).evaluate(p)
    derivative=np.zeros(6)
    step=2e-5
    for j in range(2):
        plus,minus=x.copy(),x.copy()
        plus[j]+=step
        minus[j]-=step
        derivative[j]=(energy(plus,ratio)-energy(minus,ratio))/(2*step)
    derivative[2:]= (g@cell.T)[1:].ravel()
    return e,derivative


def relax(x,ratio):
    result=minimize(objective,np.asarray(x),args=(ratio,),jac=True,method="L-BFGS-B",
                    bounds=[(-.6,.6),(-1.5,1.5)]+[(None,None)]*4,
                    options={"ftol":1e-15,"gtol":5e-9,"maxiter":2000,"maxls":40,"maxcor":20})
    cell,p=unpack(result.x,ratio)
    lengths=np.linalg.norm(cell,axis=1)
    angle=np.degrees(np.arccos(np.dot(cell[0],cell[1])/np.prod(lengths)))
    e,g=Ewald(cell,LAYERS).evaluate(p)
    return dict(a_over_d=float(ratio),energy=float(e),x=result.x.tolist(),
                angle_degrees=float(angle),length_ratio=float(lengths[1]/lengths[0]),
                reduced_angle_degrees=reduced_angle(cell),
                max_force=float(abs(g).max()),max_parameter_gradient=float(abs(objective(result.x,ratio)[1]).max()),
                success=bool(result.success),iterations=int(result.nit),message=str(result.message))


def run():
    out=ROOT/"results"/"variable_cell"
    out.mkdir(exist_ok=True)
    rng=np.random.default_rng(97001)
    rows=[]
    checkpoints={}
    for ratio in [4.,4.8,5.,5.1,5.2,5.3,5.35,5.4,5.42,5.44,5.5,5.6,5.8,6.]:
        candidates=[]
        for phase in ["checkerboard","honeycomb"]:
            for j in range(4):
                x=initial(phase,ratio)
                x[:2]+=rng.normal(0,.1,2)
                x[2:]+=rng.normal(0,.025,4)
                r=relax(x,ratio)
                r.update(initial_phase=phase,seed_index=j)
                candidates.append(r)
        candidates.sort(key=lambda x:x["energy"])
        save_json(out/f"a{ratio:g}.json",candidates)
        r=candidates[0]
        rows.append({k:v for k,v in r.items() if k not in ["x","message"]})
        c,p=unpack(np.array(r["x"]),ratio)
        checkpoints[f"cell_a{ratio:g}"]=c
        checkpoints[f"pos_a{ratio:g}"]=p
        print("variable cell",r,flush=True)
    write_csv(out/"scan.csv",rows)
    np.savez_compressed(out/"configurations.npz",**checkpoints)
    root=brentq(lambda x:strain_hessian("checkerboard",x,3e-4)[0],4.8,5.3,xtol=2e-6)
    save_json(out/"square_shear_instability.json",{"a_over_d":root,"finite_difference_step":3e-4})
    print("square shear instability",root,flush=True)


if __name__=="__main__":
    run()
