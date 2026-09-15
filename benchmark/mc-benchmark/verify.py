#!/usr/bin/env python3
"""Independent identities, derivatives, convergence, and saved-run checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import null_space
from ewald import Ewald, lattice, crystal_energy, hessian
from monte_carlo import anneal

ROOT=Path(__file__).resolve().parent


def verify_core():
    rng=np.random.default_rng(871)
    report={}
    kernel=[]
    for alpha in [.15,.7,2.5]:
        engine=Ewald(np.eye(2)*8,np.array([0,0,1]),split=8*alpha)
        for j in np.unique(np.linspace(0,len(engine.gs)-1,9,dtype=int)):
            g=np.linalg.norm(engine.gs[j])
            def integral(z):
                return 2*np.sqrt(np.pi)*quad(lambda t:
                    np.exp(-z*z*t*t-g*g/(4*t*t))/(t*t),
                    0,alpha,epsabs=2e-13,epsrel=2e-13)[0]
            kernel.append([abs(integral(0)-engine.c0[j]),abs(integral(1)-engine.cd[j])])
    report["fourier_kernel_quadrature_error"]=float(np.max(kernel))
    assert report["fourier_kernel_quadrature_error"]<1e-11
    cell=np.array([[12.,0.],[7.,9.]])
    positions=rng.random((12,2))@cell
    layers=np.r_[np.zeros(8,dtype=int),np.ones(4,dtype=int)]
    engine=Ewald(cell,layers,split=3.5,accuracy=7)
    e,gradient=engine.evaluate(positions)
    fd=np.zeros_like(positions)
    h=1e-5
    for i in range(len(positions)):
        for j in range(2):
            p,m=positions.copy(),positions.copy()
            p[i,j]+=h
            m[i,j]-=h
            fd[i,j]=(engine.evaluate(p)[0]-engine.evaluate(m)[0])/(2*h)
    report["gradient_fd_error"]=float(np.max(np.abs(fd-gradient)))
    assert report["gradient_fd_error"]<1e-7
    translated=positions+rng.integers(-20,21,size=(12,2))@cell
    et,gt=engine.evaluate(translated)
    report["periodicity_energy_error"]=float(abs(et-e))
    assert report["periodicity_energy_error"]<1e-10
    energies=[Ewald(cell,layers,split=s,accuracy=7).evaluate(positions)[0] for s in [2,3.5,6]]
    report["ewald_split_energy_spread"]=float(np.ptp(energies))
    assert report["ewald_split_energy_spread"]<1e-10
    replication=[]
    for phase in ["checkerboard","honeycomb"]:
        es=[crystal_energy(phase,5.42,repeats=n,accuracy=7) for n in [1,2,4,8]]
        replication.append(dict(phase=phase,Nh=[1,4,16,64],energies=es,spread=float(np.ptp(es))))
        assert np.ptp(es)<1e-10
    report["replication"]=replication
    c,p,l=lattice("honeycomb",5.6,2)
    p+=.15*rng.standard_normal(p.shape)
    engine=Ewald(c,l)
    final,best,history,initial=anneal(engine,p,seed=203,stages=4,sweeps=4,tmax=.03,tmin=.0001)
    report["mc_incremental_energy_drift"]=float(np.max(np.abs(history[:,-1])))
    assert report["mc_incremental_energy_drift"]<1e-10
    assert abs(engine.evaluate(best)[0]/4-history[-1,2])<1e-10
    hs=[]
    for phase in ["checkerboard","honeycomb"]:
        c,p,l=lattice(phase,5.4224,4)
        engine=Ewald(c,l)
        translations=np.tile(np.eye(2),(len(p),1))
        basis=null_space(translations.T)
        eigen=[]
        for step in [4e-4,2e-4,1e-4]:
            matrix=hessian(engine,p,step)
            eigen.append(float(np.linalg.eigvalsh(basis.T@matrix@basis)[0]))
        hs.append(dict(phase=phase,steps=[4e-4,2e-4,1e-4],lowest_eigenvalues=eigen,
                       spread=float(np.ptp(eigen))))
        assert np.ptp(eigen)<1e-8
    report["hessian_step_convergence"]=hs
    from run_benchmark import strain_hessian
    report["strain_step_convergence"]=[]
    for phase in ["checkerboard","honeycomb"]:
        vals=[strain_hessian(phase,5.4224,s).tolist() for s in [2e-3,1e-3,5e-4]]
        report["strain_step_convergence"].append(dict(phase=phase,steps=[2e-3,1e-3,5e-4],eigenvalues=vals))
        assert np.max(np.ptp(vals,axis=0))<1e-4
    return report


def verify_outputs():
    from run_benchmark import cases
    results=ROOT/"results"
    transition=json.loads((results/"transition.json").read_text())
    root=transition["a_over_d_critical"]
    assert 5.42<root<5.43
    assert crystal_energy("honeycomb",5.42)>crystal_energy("checkerboard",5.42)
    assert crystal_energy("honeycomb",5.43)<crystal_energy("checkerboard",5.43)
    records=[json.loads(p.read_text()) for p in sorted((results/"runs").glob("*.json"))]
    assert len(records)==len(cases("all")),(len(records),len(cases("all")))
    discrepancies=[]
    residuals=[]
    below=[]
    below_ideal=[]
    for record in records:
        data=np.load(results/"runs"/(record["case_id"]+".npz"))
        assert np.all(np.isfinite(data["relaxed"]))
        assert len(data["layers"])==3*record["Nh"]
        assert np.count_nonzero(data["layers"]==0)==2*record["Nh"]
        engine=Ewald(data["cell"],data["layers"])
        e,g=engine.evaluate(data["relaxed"])
        discrepancies.append(abs(e/record["Nh"]-record["relaxed_energy_per_hole"]))
        residuals.append(float(np.abs(g).max()))
        assert discrepancies[-1]<1e-11
        assert np.abs(g).max()<3e-6,(record["case_id"],np.abs(g).max())
        assert record["max_mc_energy_drift"]<1e-8
        assert e/record["Nh"]<=min(record["final_mc_energy_per_hole"],record["best_mc_energy_per_hole"])+1e-10
        ideal_lower=min(crystal_energy(p,record["a_over_d"]) for p in ["checkerboard","honeycomb"])
        if e/record["Nh"]<ideal_lower-1e-8:
            below_ideal.append(dict(case_id=record["case_id"],below_ideal=e/record["Nh"]-ideal_lower))
        from phase_analysis import phase_envelope
        lower=phase_envelope(record["a_over_d"])
        if e/record["Nh"]<lower-1e-8:
            below.append(dict(case_id=record["case_id"],below_lower_branch=e/record["Nh"]-lower))
    assert not below,below
    return dict(number_of_runs=len(records),max_saved_energy_error=max(discrepancies),
                max_relaxed_force=max(residuals),states_below_three_branch_envelope=below,
                states_below_ideal_two_branch_envelope=below_ideal)


def verify_extended():
    from phase_analysis import optimum,phase_envelope,rhombic_energy,square_shear
    from explore_cell import unpack
    from run_benchmark import strain_hessian
    result={}
    tr=json.loads((ROOT/"results/extended_transitions.json").read_text())
    assert 5.0610<tr["square_to_rhombic"]<5.0612
    assert 5.4643<tr["rhombic_to_honeycomb"]<5.4645
    assert square_shear(5.06,.003)>0
    assert square_shear(5.07,.003)<0
    assert optimum(5.46)[0]<crystal_energy("honeycomb",5.46)
    assert optimum(5.47)[0]>crystal_energy("honeycomb",5.47)
    assert optimum(5.44)[0]<crystal_energy("honeycomb",5.44)-7e-6
    errors=[]
    for a in [4.8,5.,5.06,5.1,5.3,5.44,5.46436336,5.5]:
        emin,theta=optimum(a)
        grid=[rhombic_energy(a,t) for t in np.linspace(np.pi/3,np.pi/2,151)]
        errors.append(emin-min(grid))
    assert max(errors)<1e-12
    result["scalar_minimum_above_dense_angle_grid"]=float(max(errors))
    runs=list((ROOT/"results/cell_mc").glob("*.json"))
    assert len(runs)==15
    below=[]
    max_error=0.
    max_excess=0.
    for path in runs:
        run=json.loads(path.read_text())
        assert len(run["history"])==80
        for state in [run["best"],*run["history"]]:
            c,p=unpack(np.array(state["x"]),run["ratio"])
            e=Ewald(c,np.array([0,0,1])).evaluate(p)[0]
            max_error=max(max_error,abs(e-state["energy"]))
            if e<phase_envelope(run["ratio"])-1e-8: below.append((path.name,e))
        max_excess=max(max_excess,run["best"]["energy"]-phase_envelope(run["ratio"]))
    assert max_error<1e-10
    assert max_excess<1e-8
    assert not below,below
    result.update(cell_search_runs=len(runs),cell_search_proposals=15*80,
                  max_saved_cell_energy_error=float(max_error),max_best_cell_energy_excess=float(max_excess),
                  cell_search_states_below_envelope=below)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only",action="store_true")
    args=parser.parse_args()
    result={"core":verify_core()}
    if not args.core_only:
        result["outputs"]=verify_outputs()
        result["extended"]=verify_extended()
    result["passed"]=True
    path=ROOT/"results"/("core_verification.json" if args.core_only else "verification.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
