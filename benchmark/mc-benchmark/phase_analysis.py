#!/usr/bin/env python3
"""Resolve the rhombic distortion exposed by unconstrained stability checks."""
from pathlib import Path
import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import minimize_scalar, brentq
from ewald import Ewald, lattice, crystal_energy, hessian
from run_benchmark import save_json,write_csv

ROOT=Path(__file__).resolve().parent
LAYERS=np.array([0,0,1],dtype=np.int64)


def rhombic(ratio,theta,repeats=1):
    ell=np.sqrt(np.pi*ratio**2/np.sin(theta))
    c=ell*np.array([[1.,0.],[np.cos(theta),np.sin(theta)]])
    origins=np.array([[i,j] for i in range(repeats) for j in range(repeats)])@c
    n=len(origins)
    p=np.vstack([origins,origins+.5*(c[0]+c[1]),origins])
    layers=np.r_[np.zeros(2*n,dtype=np.int64),np.ones(n,dtype=np.int64)]
    return c*repeats,p,layers


def rhombic_energy(ratio,theta,**kwargs):
    c,p,l=rhombic(ratio,theta)
    return Ewald(c,l,**kwargs).evaluate(p)[0]


def optimum(ratio,**kwargs):
    f=lambda theta:rhombic_energy(ratio,theta,**kwargs)
    result=minimize_scalar(f,bounds=(np.pi/3,np.pi/2),method="bounded",options={"xatol":1e-12})
    candidates=[(result.fun,result.x),(f(np.pi/2),np.pi/2),(f(np.pi/3),np.pi/3)]
    return min(candidates)


def square_shear(ratio,step,**kwargs):
    c,p,l=lattice("checkerboard",ratio)
    f=p@np.linalg.inv(c)
    def energy(t):
        d=c@expm(np.array([[0.,t],[t,0.]]))
        return Ewald(d,l,**kwargs).evaluate(f@d)[0]
    # Five-point second derivative; no internal relaxation is needed because
    # inversion makes all forces zero along this midpoint-basis family.
    return (-energy(2*step)+16*energy(step)-30*energy(0)+16*energy(-step)-energy(-2*step))/(12*step**2)


def phase_envelope(ratio):
    return min(optimum(ratio)[0],crystal_energy("honeycomb",ratio))


def run():
    out=ROOT/"results"
    convergence=[]
    for split,accuracy in [(3.,5.),(4.,6.),(5.,7.)]:
        kw=dict(split=split,accuracy=accuracy)
        r=brentq(lambda a:optimum(a,**kw)[0]-crystal_energy("honeycomb",a,**kw),5.44,5.5,xtol=1e-10)
        e,theta=optimum(r,**kw)
        convergence.append(dict(split=split,accuracy=accuracy,a_over_d=r,
                               energy_per_hole=e,rhombic_angle_degrees=float(np.degrees(theta))))
    square=[]
    for step in [.01,.006,.003,.0015]:
        r=brentq(lambda a:square_shear(a,step),5.,5.2,xtol=1e-7)
        square.append(dict(strain_step=step,a_over_d=r))
    transition=dict(ideal_square_honeycomb_crossing=5.42240068000776,
                    square_to_rhombic=square[-1]["a_over_d"],
                    rhombic_to_honeycomb=convergence[1]["a_over_d"],
                    rhombic_honeycomb_convergence=convergence,
                    square_rhombic_convergence=square,
                    scope="Lowest energy among square, optimized midpoint rhombic, and honeycomb; MC and variable-cell searches support these candidates, not a proof excluding all larger cells.")
    save_json(out/"extended_transitions.json",transition)
    print(transition,flush=True)
    rows=[]
    for a in np.linspace(4.8,5.65,171):
        er,theta=optimum(a)
        ec=crystal_energy("checkerboard",a)
        eh=crystal_energy("honeycomb",a)
        rows.append(dict(a_over_d=float(a),square=ec,rhombic=er,honeycomb=eh,
            rhombic_minus_square=er-ec,honeycomb_minus_rhombic=eh-er,
            rhombic_angle_degrees=float(np.degrees(theta))))
    write_csv(out/"extended_phase_scan.csv",rows)
    stability=[]
    data={}
    for a in [5.1,5.3,5.42,5.44,transition["rhombic_to_honeycomb"],5.5]:
        er,theta=optimum(a)
        for n in [4,8]:
            c,p,l=rhombic(a,theta,n)
            engine=Ewald(c,l)
            h=hessian(engine,p)
            translations=np.tile(np.eye(2),(len(p),1))
            basis=null_space(translations.T)
            eigen=np.linalg.eigvalsh(basis.T@h@basis)
            key=f"a{a:.8f}_n{n*n}"
            data["eigen_"+key]=eigen
            data["cell_"+key]=c
            data["pos_"+key]=p
            stability.append(dict(a_over_d=a,Nh=n*n,angle_degrees=float(np.degrees(theta)),
                energy_per_hole=engine.evaluate(p)[0]/n**2,
                min_nontranslation_eigenvalue=float(eigen[0]),negative_modes=int(np.sum(eigen < -1e-8))))
            print("rhombic stability",stability[-1],flush=True)
    write_csv(out/"rhombic_stability.csv",stability)
    np.savez_compressed(out/"rhombic_stability.npz",**data)


if __name__=="__main__":
    run()
