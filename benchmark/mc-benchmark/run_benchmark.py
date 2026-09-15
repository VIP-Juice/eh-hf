#!/usr/bin/env python3
"""Reproduce the classical comparison and execute actual Metropolis annealing.

Run from any directory. Results and checkpoints are written beside this file.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import brentq
from ewald import Ewald, lattice, crystal_energy, hessian
from monte_carlo import anneal, proposal_partners

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"
PHASES=["checkerboard","honeycomb"]


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def periodic_vectors(engine, points):
    frac=(points[:,None,:]-points[None,:,:])@engine.invcell
    frac-=np.round(frac)
    best=np.full(frac.shape[:2],np.inf)
    vectors=np.zeros_like(frac)
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            trial=(frac+np.array([i,j]))@engine.cell
            r2=(trial*trial).sum(axis=2)
            use=r2<best
            vectors[use]=trial[use]
            best[use]=r2[use]
    np.fill_diagonal(best,np.inf)
    return best,vectors


def order_parameters(engine,positions):
    ne=2*len(positions)//3
    partners=proposal_partners(engine,positions)
    excess=positions[np.setdiff1d(np.arange(ne),partners)]
    holes=positions[ne:]
    result={}
    for label,points in [("hole",holes),("excess",excess)]:
        distance,vectors=periodic_vectors(engine,points)
        for m in [4,6]:
            idx=np.argsort(distance,axis=1)[:,:min(m,len(points)-1)]
            bonds=vectors[np.arange(len(points))[:,None],idx]
            theta=np.arctan2(bonds[:,:,1],bonds[:,:,0])
            result[f"{label}_psi{m}"]=float(abs(np.exp(1j*m*theta).mean()))
    f=(positions[partners]-holes)@engine.invcell
    d2=np.full(len(holes),np.inf)
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            r=(f-np.round(f)+np.array([i,j]))@engine.cell
            d2=np.minimum(d2,(r*r).sum(axis=1))
    result["eh_inplane_rms_d"]=float(np.sqrt(d2.mean()))
    result["paired_fraction_within_0p25d"]=float(np.mean(d2<.25**2))
    if min(result["hole_psi4"],result["excess_psi4"])>.9999:
        label="square_order"
    elif min(result["hole_psi6"],result["excess_psi6"])>.9999:
        label="triangular_order"
    elif min(result["hole_psi4"],result["excess_psi4"])>.85:
        label="distorted_square_like"
    elif min(result["hole_psi6"],result["excess_psi6"])>.85:
        label="distorted_triangular_like"
    else:
        label="distorted_or_defective"
    result["final_order"]=label
    return result


def analytic():
    RESULTS.mkdir(exist_ok=True)
    rows=[]
    for ratio in sorted(set(np.r_[np.linspace(1,10,181),np.linspace(5.3,5.55,51)])):
        ec=crystal_energy("checkerboard",ratio)
        eh=crystal_energy("honeycomb",ratio)
        rows.append(dict(a_over_d=float(ratio),checkerboard=ec,honeycomb=eh,
                         honeycomb_minus_checkerboard=eh-ec))
    write_csv(RESULTS/"lattice_energies.csv",rows)
    roots=[]
    for split,accuracy in [(3.,5.),(4.,6.),(5.,7.)]:
        def difference(x):
            return (crystal_energy("honeycomb",x,split=split,accuracy=accuracy)
                    -crystal_energy("checkerboard",x,split=split,accuracy=accuracy))
        root=brentq(difference,5.3,5.6,xtol=2e-11)
        roots.append(dict(split=split,accuracy=accuracy,a_over_d=root))
    root=roots[1]["a_over_d"]
    slope=((crystal_energy("honeycomb",root+1e-3)-crystal_energy("checkerboard",root+1e-3))
           -(crystal_energy("honeycomb",root-1e-3)-crystal_energy("checkerboard",root-1e-3)))/.002
    result={"a_over_d_critical":root,"energy_per_hole_Ed":crystal_energy("checkerboard",root),
            "delta_energy_slope":slope,"ewald_convergence":roots,
            "scope":"T=0 crossing of the two defect-free composite crystals"}
    save_json(RESULTS/"transition.json",result)
    print(json.dumps(result),flush=True)


def strain_hessian(phase,ratio,step):
    cell,pos,layers=lattice(phase,ratio)
    frac=pos@np.linalg.inv(cell)
    def energy(s,t):
        deformed=cell@expm(np.array([[s,t],[t,-s]]))
        engine=Ewald(deformed,layers)
        relaxed,info=engine.relax(frac@deformed)
        return info["energy"]
    e0=energy(0.,0.)
    xx=(energy(step,0)+energy(-step,0)-2*e0)/step**2
    yy=(energy(0,step)+energy(0,-step)-2*e0)/step**2
    xy=(energy(step,step)+energy(-step,-step)-energy(step,-step)-energy(-step,step))/(4*step**2)
    return np.linalg.eigvalsh([[xx,xy],[xy,yy]])


def stability():
    rows=[]
    spectra={}
    for ratio in [2.,4.,5.,5.4,5.422400,5.44,6.,8.]:
        for phase in PHASES:
            for repeats in [4,8]:
                cell,pos,layers=lattice(phase,ratio,repeats)
                engine=Ewald(cell,layers)
                h=hessian(engine,pos)
                # Project out the two EXACT uniform translations, not just
                # the lowest eigenvalues (which could include instabilities).
                translations=np.tile(np.eye(2),(len(pos),1))
                basis=null_space(translations.T)
                eigenvalues=np.linalg.eigvalsh(basis.T@h@basis)
                key=f"{phase}_a{ratio:g}_n{repeats**2}"
                spectra[key]=eigenvalues
                strain=strain_hessian(phase,ratio,1e-3)
                rows.append(dict(phase=phase,a_over_d=ratio,Nh=repeats**2,
                    min_nontranslation_eigenvalue=float(eigenvalues[0]),
                    negative_modes=int(np.sum(eigenvalues < -1e-8)),
                    translation_residual=float(np.abs(h@translations).max()),
                    max_force=float(np.abs(engine.evaluate(pos)[1]).max()),
                    strain_min_eigenvalue=float(strain[0]),strain_max_eigenvalue=float(strain[1])))
                print("stability",key,rows[-1],flush=True)
    write_csv(RESULTS/"stability.csv",rows)
    np.savez_compressed(RESULTS/"hessian_spectra.npz",**spectra)


def run_case(case):
    phase,ratio,repeats,seed,initialization,stages,sweeps,tmax=case
    key=f"{initialization}_{phase}_a{ratio:g}_n{repeats**2}_s{seed}"
    path=RESULTS/"runs"/f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    start=time.monotonic()
    if phase=="rhombic":
        from phase_analysis import optimum,rhombic
        _,theta=optimum(ratio)
        cell,ideal,layers=rhombic(ratio,theta,repeats)
    else:
        cell,ideal,layers=lattice(phase,ratio,repeats)
    engine=Ewald(cell,layers)
    rng=np.random.default_rng(seed)
    if initialization=="perturbed":
        initial=engine.wrap(ideal+rng.normal(0,.04*ratio,ideal.shape))
    elif initialization=="random":
        initial=rng.random(ideal.shape)@cell
    else:
        raise ValueError(initialization)
    final,best,history,initial_energy=anneal(engine,initial,seed,
        stages=stages,sweeps=sweeps,tmax=tmax,tmin=2e-7)
    # Quench the final and best visited configurations; save BOTH outcomes.
    relaxed_best,info_best=engine.relax(best)
    relaxed_final,info_final=engine.relax(final)
    if info_best["energy"]<info_final["energy"]:
        relaxed,info=relaxed_best,info_best
    else:
        relaxed,info=relaxed_final,info_final
    nh=repeats**2
    ideal_e=engine.evaluate(ideal)[0]/nh
    record=dict(case_id=key,initial_lattice=phase,initialization=initialization,
        a_over_d=ratio,Nh=nh,Ne=2*nh,seed=seed,stages=stages,sweeps_per_stage=sweeps,
        attempted_moves=stages*sweeps*len(ideal),tmax=tmax,tmin=2e-7,
        initial_energy_per_hole=initial_energy/nh,
        final_mc_energy_per_hole=engine.evaluate(final)[0]/nh,
        best_mc_energy_per_hole=engine.evaluate(best)[0]/nh,
        relaxed_energy_per_hole=info["energy"]/nh,
        ideal_seed_energy_per_hole=ideal_e,
        relaxed_minus_ideal=info["energy"]/nh-ideal_e,
        max_mc_energy_drift=float(np.abs(history[:,-1]).max()),
        quench_best=info_best,quench_final=info_final,
        max_force=info["max_force"],relax_iterations=info["iterations"],
        seconds=time.monotonic()-start,**order_parameters(engine,relaxed))
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"),cell=cell,layers=layers,
                        initial=initial,final_mc=engine.wrap(final),best_mc=engine.wrap(best),
                        relaxed=relaxed,history=history)
    save_json(path,record)
    return record


def cases(suite):
    jobs=[]
    if suite in ["all","seeded"]:
        for ratio in [2.,4.,5.35,5.40,5.42,5.44,5.50,6.,8.]:
            for phase in PHASES:
                for seed in [1103,2207,3301]:
                    jobs.append((phase,ratio,4,seed,"perturbed",32,100,.006))
        for ratio in [2.,5.40,5.44,8.]:
            for phase in PHASES:
                for seed in [4409,5501]:
                    jobs.append((phase,ratio,8,seed,"perturbed",32,100,.006))
    if suite in ["all","random"]:
        for ratio in [2.,5.40,5.44,8.]:
            for phase in PHASES:
                for seed in [6619,7703]:
                    jobs.append((phase,ratio,4,seed,"random",40,200,.1))
        for ratio in [2.,8.]:
            for phase in PHASES:
                jobs.append((phase,ratio,8,8803,"random",40,200,.1))
    if suite in ["all","extended"]:
        for ratio in [5.1,5.4,5.44,5.46,5.47]:
            for seed in [1103,2207,3301]:
                jobs.append(("rhombic",ratio,4,seed,"perturbed",32,100,.0005))
        for ratio in [5.46,5.47]:
            for seed in [1103,2207,3301]:
                jobs.append(("honeycomb",ratio,4,seed,"perturbed",32,100,.0005))
        for ratio in [5.44,5.46,5.47]:
            for seed in [4409,5501]:
                jobs.append(("rhombic",ratio,8,seed,"perturbed",32,100,.0005))
        for ratio in [5.46,5.47]:
            for seed in [4409,5501]:
                jobs.append(("honeycomb",ratio,8,seed,"perturbed",32,100,.0005))
    return jobs


def mc(suite,workers):
    jobs=cases(suite)
    RESULTS.mkdir(exist_ok=True)
    save_json(RESULTS/f"manifest_{suite}.json",{"case_fields":["phase","a_over_d","repeats","seed",
              "initialization","stages","sweeps","tmax"],"cases":jobs})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(run_case,case) for case in jobs]
        for f in as_completed(fs):
            r=f.result()
            print(f'{r["case_id"]}: E/Nh={r["relaxed_energy_per_hole"]:.12f} '
                  f'order={r["final_order"]} residual={r["max_force"]:.2g} '
                  f'time={r["seconds"]:.1f}s',flush=True)
    records=[json.loads(p.read_text()) for p in sorted((RESULTS/"runs").glob("*.json"))]
    flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in records]
    write_csv(RESULTS/"mc_runs.csv",flat)


def refine_saved():
    """Continue only quenches whose residual force failed the final check."""
    for path in sorted((RESULTS/"runs").glob("*.json")):
        row=json.loads(path.read_text())
        if row["max_force"]<=1e-6:
            continue
        with np.load(path.with_suffix(".npz")) as data:
            payload={k:data[k] for k in data.files}
        engine=Ewald(payload["cell"],payload["layers"])
        before=payload["relaxed"].copy()
        from monte_carlo import relax_preconditioned
        refined,info=relax_preconditioned(engine,before,maxiter=20000)
        if info["energy"]>row["relaxed_energy_per_hole"]*row["Nh"]+1e-10:
            raise RuntimeError("Refinement raised energy")
        payload["relaxed_before_refinement"]=before
        payload["relaxed"]=refined
        row["before_refinement"]={"energy_per_hole":row["relaxed_energy_per_hole"],"max_force":row["max_force"]}
        row["refinement"]=info
        row["relaxed_energy_per_hole"]=info["energy"]/row["Nh"]
        row["relaxed_minus_ideal"]=row["relaxed_energy_per_hole"]-row["ideal_seed_energy_per_hole"]
        row["max_force"]=info["max_force"]
        row["relax_iterations"]+=info["iterations"]
        row.update(order_parameters(engine,refined))
        np.savez_compressed(path.with_suffix(".npz"),**payload)
        save_json(path,row)
        print("refined",row["case_id"],info,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["analytic","stability","mc","refine","all"])
    parser.add_argument("--suite",choices=["all","seeded","random","extended"],default="all")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.stage in ["all","analytic"]: analytic()
    if args.stage in ["all","stability"]: stability()
    if args.stage in ["all","mc"]: mc(args.suite,args.workers)
    if args.stage in ["all","refine"]: refine_saved()


if __name__=="__main__":
    main()
