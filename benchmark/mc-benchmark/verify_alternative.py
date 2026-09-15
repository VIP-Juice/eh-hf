#!/usr/bin/env python3
"""Energy-only check using intralayer Ewald plus UNSPLIT interlayer Fourier.

This deliberately uses the earlier benchmark's independent pair summation.
It is slower but tests the surprising shear instability and rhombic energy.
"""
import numpy as np
from scipy.linalg import expm
from reference_energy import intralayer_ewald_energy,interlayer_energy
from ewald import Ewald,lattice,crystal_energy
from phase_analysis import rhombic,optimum
from run_benchmark import RESULTS,save_json


def alternative(cell,positions):
    return (intralayer_ewald_energy(positions[:2],np.array([-1.,-1.]),*cell,
                                   real_cutoff=10,reciprocal_cutoff=12)
            +intralayer_ewald_energy(positions[2:],np.array([1.]),*cell,
                                    real_cutoff=10,reciprocal_cutoff=12)
            +interlayer_energy(positions[:2],positions[2:],*cell,reciprocal_radius=40.))


def run():
    checks=[]
    for ratio in [5.,5.4,5.4224,5.44,5.46436336,5.5]:
        cell,pos,layers=lattice("checkerboard",ratio)
        frac=pos@np.linalg.inv(cell)
        for t in [0.,.001,-.001]:
            c=cell@expm(np.array([[0.,t],[t,0.]]))
            p=frac@c
            e=alternative(c,p)
            reference=Ewald(c,layers).evaluate(p)[0]
            checks.append(dict(ratio=ratio,structure="sheared_square",strain=t,
                               alternative_energy=e,new_energy=reference,error=abs(e-reference)))
        er,theta=optimum(ratio)
        c,p,l=rhombic(ratio,theta)
        e=alternative(c,p)
        checks.append(dict(ratio=ratio,structure="optimized_rhombic",angle_degrees=float(np.degrees(theta)),
                           alternative_energy=e,new_energy=er,error=abs(e-er)))
        c,p,l=lattice("honeycomb",ratio)
        e=alternative(c,p)
        reference=crystal_energy("honeycomb",ratio)
        checks.append(dict(ratio=ratio,structure="honeycomb",alternative_energy=e,
                           new_energy=reference,error=abs(e-reference)))
    maxerr=max(c["error"] for c in checks)
    assert maxerr<1e-10
    result=dict(max_energy_error=maxerr,checks=checks,passed=True)
    save_json(RESULTS/"alternative_verification.json",result)
    print("alternative formulation max energy error",maxerr,flush=True)


if __name__=="__main__":
    run()
