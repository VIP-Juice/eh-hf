#!/usr/bin/env python3
"""Basin-hopping MC of the full 3-charge primitive cell and its shape.

This is a ground-state search, not finite-temperature thermodynamics. Each
proposal changes all six independent parameters and is locally quenched.
The cell area is fixed, while electron/hole positions and cell shape vary.
"""
import numpy as np
from explore_cell import initial,relax
from run_benchmark import RESULTS,save_json,write_csv
from phase_analysis import optimum
from ewald import crystal_energy


def run():
    rows=[]
    for ratio in [5.05,5.2,5.44,5.46436336,5.5]:
        for seed in [99103,99207,99301]:
            rng=np.random.default_rng(seed)
            x=initial("checkerboard" if seed%2 else "honeycomb",ratio)
            x[:2]=rng.uniform([-.2,-.5],[.2,.8])
            x[2:]=rng.random(4)
            current=relax(x,ratio)
            best=current
            history=[]
            for step in range(80):
                x=np.array(current["x"])
                x+=rng.normal(size=6)*np.array([.08,.22,.16,.16,.05,.05])
                x[:2]=np.clip(x[:2],[-.5,-1.4],[.5,1.4])
                proposed=relax(x,ratio)
                de=proposed["energy"]-current["energy"]
                accepted=de<=0 or rng.random()<np.exp(-de/.0003)
                if accepted: current=proposed
                if proposed["energy"]<best["energy"]: best=proposed
                er,theta=optimum(ratio)
                eh=crystal_energy("honeycomb",ratio)
                branch=("honeycomb" if abs(proposed["energy"]-eh)<1e-8 and
                        abs(proposed["reduced_angle_degrees"]-60)<.1 else
                        "rhombic_or_square" if abs(proposed["energy"]-er)<1e-8 else "other_local_minimum")
                history.append(dict(step=step,accepted=bool(accepted),energy=proposed["energy"],
                                    branch=branch,x=proposed["x"],max_force=proposed["max_force"]))
            result=dict(ratio=ratio,seed=seed,steps=80,temperature=.0003,
                        best=best,history=history)
            save_json(RESULTS/"cell_mc"/f"a{ratio:.8f}_s{seed}.json",result)
            row=dict(a_over_d=ratio,seed=seed,best_energy=best["energy"],
                     predicted_envelope=min(optimum(ratio)[0],crystal_energy("honeycomb",ratio)),
                     max_force=best["max_force"],
                     rhombic_or_square_visits=sum(h["branch"]=="rhombic_or_square" for h in history),
                     honeycomb_visits=sum(h["branch"]=="honeycomb" for h in history),
                     other_visits=sum(h["branch"]=="other_local_minimum" for h in history))
            rows.append(row)
            print("cell MC",row,flush=True)
    write_csv(RESULTS/"cell_mc_summary.csv",rows)


if __name__=="__main__":
    run()
