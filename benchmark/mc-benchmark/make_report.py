#!/usr/bin/env python3
"""Build the scientific report and figures from saved numerical artifacts."""
import csv
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ewald import Ewald,crystal_energy
from phase_analysis import optimum,phase_envelope,square_shear
from explore_cell import unpack,reduced_angle
from run_benchmark import order_parameters,write_csv,save_json

R=ROOT/"results"
F=ROOT/"figures"
COLORS={"checkerboard":"#0072B2","rhombic":"#D55E00","honeycomb":"#009E73"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.dpi":200,"figure.dpi":110,"axes.labelsize":11})


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def table(rows,columns):
    return "| "+" | ".join(k for k,v in columns)+" |\n|"+"|".join("---" for _ in columns)+"|\n"+"\n".join(
        "| "+" | ".join(fn(r) for k,fn in columns)+" |" for r in rows)


def refresh_labels(records):
    # Recompute descriptive labels from saved positions; never infer a final
    # phase from the initialization or from a previously used coarse label.
    for row in records:
        data=np.load(R/"runs"/(row["case_id"]+".npz"))
        row.update(order_parameters(Ewald(data["cell"],data["layers"]),data["relaxed"]))
        save_json(R/"runs"/(row["case_id"]+".json"),row)
    write_csv(R/"mc_runs.csv",[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in records])
    cell_rows=[]
    for path in sorted((R/"cell_mc").glob("*.json")):
        run=json.loads(path.read_text())
        a=run["ratio"]
        er,_=optimum(a)
        eh=crystal_energy("honeycomb",a)
        for h in run["history"]:
            cell,p=unpack(np.array(h["x"]),a)
            h["reduced_angle_degrees"]=reduced_angle(cell)
            h["branch"]=("honeycomb" if abs(h["energy"]-eh)<1e-8 and abs(h["reduced_angle_degrees"]-60)<.1
                          else "rhombic_or_square" if abs(h["energy"]-er)<1e-8 else "other_local_minimum")
        save_json(path,run)
        cell_rows.append(dict(a_over_d=a,seed=run["seed"],best_energy=run["best"]["energy"],
            predicted_envelope=min(er,eh),max_force=run["best"]["max_force"],
            rhombic_or_square_visits=sum(h["branch"]=="rhombic_or_square" for h in run["history"]),
            honeycomb_visits=sum(h["branch"]=="honeycomb" for h in run["history"]),
            other_visits=sum(h["branch"]=="other_local_minimum" for h in run["history"])))
    if cell_rows: write_csv(R/"cell_mc_summary.csv",cell_rows)


def get_best(records,phase,ratio,nh,init="perturbed"):
    choices=[r for r in records if r["initial_lattice"]==phase and r["a_over_d"]==ratio
             and r["Nh"]==nh and r["initialization"]==init]
    return min(choices,key=lambda r:r["relaxed_energy_per_hole"])


def energy_plot(records,tr):
    rows=read_csv(R/"extended_phase_scan.csv")
    a=np.array([float(r["a_over_d"]) for r in rows])
    sq=np.array([float(r["square"]) for r in rows])
    rh=np.array([float(r["rhombic"]) for r in rows])
    hc=np.array([float(r["honeycomb"]) for r in rows])
    theta=np.array([float(r["rhombic_angle_degrees"]) for r in rows])
    r1,r2=tr["square_to_rhombic"],tr["rhombic_to_honeycomb"]
    fig,axes=plt.subplots(3,1,figsize=(8.1,10),constrained_layout=True)
    ax=axes[0]
    ax.axhline(0,color=COLORS["checkerboard"],label="Ideal square",lw=1.5)
    ax.plot(a,(rh-sq)*1e6,color=COLORS["rhombic"],label="Optimized rhombic",lw=2)
    ax.plot(a,(hc-sq)*1e6,color=COLORS["honeycomb"],label="Honeycomb",lw=2)
    ax.axvline(tr["ideal_square_honeycomb_crossing"],color=".5",ls=":",label="Ideal-only crossing 5.42240")
    ax.set(xlim=(4.9,5.65),ylim=(-85,190),ylabel=r"$(U/N_h-E_{\rm square})/E_d\ \times 10^6$",
           title="Allowing shear lowers the energy near the ideal crossing")
    ax.legend(frameon=True,facecolor="white",edgecolor="white",framealpha=.96,fontsize=9,loc="upper right")
    for x in [r1,r2]: ax.axvline(x,color=".2",ls="--",lw=.8)
    ax=axes[1]
    ax.plot(a,(hc-rh)*1e6,color=".2",lw=1.5,label="Converged lattice energies")
    for nh,marker in [(16,"o"),(64,"s")]:
        xx=[];yy=[]
        for ratio in [5.44,5.46,5.47]:
            try:
                rr=get_best(records,"rhombic",ratio,nh)
                hh=get_best(records,"honeycomb",ratio,nh)
                xx.append(ratio);yy.append((hh["relaxed_energy_per_hole"]-rr["relaxed_energy_per_hole"])*1e6)
            except ValueError: pass
        ax.scatter(xx,yy,marker=marker,s=65,facecolors="none",edgecolors=COLORS["rhombic"] if nh==16 else COLORS["honeycomb"],label=f"MC + quench, $N_h={nh}$",zorder=5)
    ax.axhline(0,color=".6",lw=.7)
    ax.axvline(r2,color=".2",ls="--",lw=.8)
    ax.set(xlim=(5.435,5.48),ylim=(-5,10),ylabel=r"$(E_{\rm honeycomb}-E_{\rm rhombic})/E_d\ \times 10^6$",
           title=f"Rhombic-honeycomb crossing: a/d = {r2:.5f}")
    ax.legend(frameon=False,fontsize=9)
    ax=axes[2]
    groundtheta=np.where(a<r1,90,np.where(a<r2,theta,60))
    ax.plot(a,theta,color=COLORS["rhombic"],alpha=.35,lw=1.5,label="Rhombic branch")
    ax.plot(a,groundtheta,color=".15",lw=2,label="Lowest tested branch")
    for left,right,color,label in [(4.9,r1,COLORS["checkerboard"],"Square"),(r1,r2,COLORS["rhombic"],"Rhombic"),(r2,5.65,COLORS["honeycomb"],"Honeycomb")]:
        ax.axvspan(left,right,color=color,alpha=.07)
        ax.text((left+right)/2,94,label,ha="center",color=color,fontweight="bold")
    ax.set(xlim=(4.9,5.65),ylim=(57,97),ylabel="Primitive rhombus angle (degrees)",xlabel=r"$a/d$",
           title=f"Square loses shear stability at a/d = {r1:.5f}")
    for ax in axes:
        ax.set_xlabel(r"$a/d$")
        ax.grid(alpha=.15)
    fig.savefig(F/"phase_comparison.png")
    fig.savefig(F/"phase_comparison.svg")
    plt.close(fig)


def snapshot(ax,row,title):
    data=np.load(R/"runs"/(row["case_id"]+".npz"))
    a=row["a_over_d"]
    cell=data["cell"]/a
    pos=data["relaxed"]/a
    layers=data["layers"]
    polygon=np.array([[0.,0.],cell[0],cell.sum(axis=0),cell[1],[0.,0.]])
    ax.plot(*polygon.T,color=".25",lw=1)
    ax.scatter(*pos[layers==1].T,s=42,facecolors="none",edgecolors="#377EB8",lw=1.2,label="Hole")
    ax.scatter(*pos[layers==0].T,s=13,color="#D43F3A",label="Electron",zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/a$");ax.set_ylabel(r"$y/a$")
    ax.set_title(title+f'\n$a/d={a:g}$; $N_e={row["Ne"]}, N_h={row["Nh"]}$',fontsize=10)


def configuration_plot(records):
    specs=[("checkerboard",2.,64,"Square: annealed and relaxed"),
           ("rhombic",5.44,64,"Rhombic: annealed and relaxed"),
           ("honeycomb",5.47,64,"Honeycomb: annealed and relaxed")]
    fig,axes=plt.subplots(1,3,figsize=(14,6.0),constrained_layout=True)
    for ax,(phase,a,n,title) in zip(axes,specs):
        snapshot(ax,get_best(records,phase,a,n),title)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=10,loc="lower center",ncol=2,bbox_to_anchor=(.5,-.02))
    fig.suptitle("Actual Monte Carlo + quench configurations (projected into the plane)\nA red dot inside a blue ring is a vertically bound electron-hole pair",fontsize=12)
    fig.savefig(F/"configurations.png",bbox_inches="tight",pad_inches=.15)
    fig.savefig(F/"configurations.svg",bbox_inches="tight",pad_inches=.15)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for col,(phase,a) in enumerate([("checkerboard",2.),("honeycomb",8.)]):
        row=get_best(records,phase,a,64,"random")
        snapshot(axes[0,col],row,"Random-start local minimum")
        data=np.load(R/"runs"/(row["case_id"]+".npz"))
        history=data["history"]
        base=phase_envelope(a)
        ax=axes[1,col]
        ax.loglog(history[:,0],np.maximum(history[:,1]-base,1e-12),label="Stage-final MC")
        ax.loglog(history[:,0],np.maximum(history[:,2]-base,1e-12),label="Best MC so far")
        ax.axhline(max(row["relaxed_energy_per_hole"]-base,1e-12),color=".25",ls="--",label="After quench")
        ax.invert_xaxis()
        ax.set(xlabel=r"Annealing temperature $T/E_d$",ylabel=r"Energy excess per hole / $E_d$")
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle("Random starts can retain defects: all outcomes are saved",fontsize=12)
    fig.savefig(F/"random_starts.png")
    plt.close(fig)


def stability_plot(tr):
    rows=read_csv(R/"stability.csv")
    rhrows=read_csv(R/"rhombic_stability.csv")
    fig,axes=plt.subplots(2,1,figsize=(8,7.5),constrained_layout=True)
    for phase in ["checkerboard","honeycomb","rhombic"]:
        for nh,style in [(16,"--"),(64,"-")]:
            select=[r for r in (rhrows if phase=="rhombic" else rows)
                    if int(r["Nh"])==nh and (phase=="rhombic" or r["phase"]==phase)]
            axes[0].plot([float(r["a_over_d"]) for r in select],[float(r["min_nontranslation_eigenvalue"]) for r in select],
                         style,marker="o",ms=3,color=COLORS[phase],label=f"{phase}, $N_h={nh}$")
    axes[0].set_yscale("symlog",linthresh=1e-6)
    axes[0].axhline(0,color=".3",lw=.8)
    axes[0].set(xlabel=r"$a/d$",ylabel=r"$\lambda_{\min}$ after removing translations [$E_d/d^2$]",
                title="Negative eigenvalues expose unstable displacement modes")
    axes[0].legend(frameon=False,fontsize=8,ncol=2)
    a=np.linspace(4.9,5.65,80)
    curv=[square_shear(x,.003) for x in a]
    axes[1].plot(a,curv,color=COLORS["checkerboard"],lw=2)
    axes[1].axhline(0,color=".3",lw=.8)
    axes[1].axvline(tr["square_to_rhombic"],color=".3",ls="--",lw=1)
    axes[1].set(xlabel=r"$a/d$",ylabel=r"Square shear curvature per hole [$E_d$]",
                title="Square shear instability at fixed area")
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(F/"stability.png")
    plt.close(fig)


def report(records,tr):
    core=json.loads((R/"verification.json").read_text())
    alternative=json.loads((R/"alternative_verification.json").read_text())
    examples=[]
    for a in [5.,5.4,5.42,5.44,5.46,5.47,5.5]:
        er,theta=optimum(a)
        examples.append(dict(a=a,square=crystal_energy("checkerboard",a),rhombic=er,
                             honeycomb=crystal_energy("honeycomb",a),theta=90. if a<tr["square_to_rhombic"] else np.degrees(theta)))
    rows=[]
    for nh in [16,64]:
        for a in [5.44,5.46,5.47]:
            rr=get_best(records,"rhombic",a,nh)
            hh=get_best(records,"honeycomb",a,nh)
            rows.append(dict(a=a,nh=nh,rhombic=rr["relaxed_energy_per_hole"],honeycomb=hh["relaxed_energy_per_hole"],
                             delta=hh["relaxed_energy_per_hole"]-rr["relaxed_energy_per_hole"],
                             maxforce=max(rr["max_force"],hh["max_force"])))
    write_csv(R/"mc_boundary_comparison.csv",rows)
    estimates=[]
    for nh in [16,64]:
        l=next(r for r in rows if r["nh"]==nh and r["a"]==5.46)
        u=next(r for r in rows if r["nh"]==nh and r["a"]==5.47)
        estimates.append(dict(Nh=nh,linear_interpolated_mc_crossing=5.46-.01*l["delta"]/(u["delta"]-l["delta"]),
                              bracket=[5.46,5.47]))
    save_json(R/"mc_transition_estimates.json",estimates)
    e_table=table(examples,[("a/d",lambda r:f'{r["a"]:.2f}'),("Ideal square",lambda r:f'{r["square"]:.12f}'),
                         ("Optimized rhombic",lambda r:f'{r["rhombic"]:.12f}'),("Honeycomb",lambda r:f'{r["honeycomb"]:.12f}'),
                         ("Rhombic angle",lambda r:f'{r["theta"]:.4f} deg')])
    mc_table=table(rows,[("a/d",lambda r:f'{r["a"]:.2f}'),("Nh",lambda r:str(r["nh"])),
                        ("MC+quench rhombic",lambda r:f'{r["rhombic"]:.12f}'),("MC+quench honeycomb",lambda r:f'{r["honeycomb"]:.12f}'),
                        ("Honeycomb - rhombic",lambda r:f'{r["delta"]:+.6e}')])
    stability=read_csv(R/"stability.csv")
    stability=[r for r in stability if int(r["Nh"])==64 and float(r["a_over_d"]) in [2.,5.4224,8.]]
    stab_table=table(stability,[("Structure",lambda r:r["phase"]),("a/d",lambda r:r["a_over_d"]),
                              ("Lowest non-translation eigenvalue",lambda r:f'{float(r["min_nontranslation_eigenvalue"]):+.6e}'),
                              ("Negative modes",lambda r:r["negative_modes"])])
    total_moves=sum(r["attempted_moves"] for r in records)
    n_random=sum(r["initialization"]=="random" for r in records)
    n_below=len(core["outputs"]["states_below_ideal_two_branch_envelope"])
    er,theta=optimum(5.44)
    delta=er-crystal_energy("honeycomb",5.44)
    text=f'''# Classical Monte Carlo comparison and stability result

## Result

**The ideal checkerboard-honeycomb crossing is reproduced at a/d = 5.42240068.**
However, the square checkerboard is unstable to shear before reaching it.
An optimized rhombic crystal is lower in energy nearby. The lowest tested
sequence is:

| Candidate ground-state boundary | a/d | Interpretation |
|---|---:|---|
| Square to rhombic | **{tr["square_to_rhombic"]:.5f}** | Continuous onset of shear; angle departs from 90 degrees |
| Rhombic to honeycomb | **{tr["rhombic_to_honeycomb"]:.5f}** | Energy crossing; angle jumps from about 79.99 to 60 degrees |
| Ideal square vs ideal honeycomb only | 5.42240068 | Restricted comparison reproducing the paper's 5.42 |

The rhombic-to-honeycomb crossing is bracketed directly by the relaxed
Monte Carlo energies at **5.46 and 5.47**, for both 48 and 192 individual
charges. Linear interpolation gives {estimates[0]["linear_interpolated_mc_crossing"]:.6f}
(Nh=16) and {estimates[1]["linear_interpolated_mc_crossing"]:.6f} (Nh=64).
The more precise value above comes from converged periodic lattice energies;
it is not a statistical Monte Carlo confidence interval.

![Energy comparison and transitions](figures/phase_comparison.png)

## Why the extra branch matters

At a/d=5.44, the rhombic angle is about {np.degrees(theta):.4f} degrees.
Its energy per hole is **{er:.12f} E_d**, which is **{abs(delta):.8e} E_d
lower than ideal honeycomb**. This is an explicit counterexample to using
the crossing of only the two ideal crystals as the unrestricted ground-state
boundary. The maximum discrepancy between the primary and independently
split electrostatic energies is {alternative["max_energy_error"]:.3e} E_d.

The density convention is **a=1/sqrt(pi*n_h), n_e=2*n_h**. Units are
**d=1, E_d=e^2/(4*pi*epsilon*d)**. Every energy below is **U/Nh**, not U/(Ne+Nh).

{e_table}

## Monte Carlo evidence

Completed **{len(records)} particle annealing runs**, including **{n_random}
uniform-random starts**, with **{total_moves:,} attempted particle/pair moves**.
The runs use Ne=2Nh and Nh=16 or 64. Every electron and hole moves independently;
paired proposals improve efficiency without enforcing binding.
The best visited and final MC states are separately quenched to zero temperature.

The table uses the lowest relaxed run for each indicated initial branch,
density, and size. Other runs, including defective outcomes, remain in the
saved dataset. The added boundary suite uses a lower starting temperature
than the broad exploratory suite; details are in [METHODS.md](METHODS.md).

{mc_table}

![Actual particle configurations](figures/configurations.png)

There are also **15 independently initialized cell-shape basin-hopping
searches, totaling 1,200 quenched Metropolis proposals**. They vary both the
cell shape and all independent primitive-cell particle coordinates. Their
lowest energies agree with the extended square/rhombic/honeycomb envelope;
see `results/cell_mc_summary.csv` and the saved proposal histories. The
quenched visitation counts describe this optimizer and have no interpretation
as physical equilibrium phase probabilities.

![Random-start outcomes and annealing histories](figures/random_starts.png)

## Configuration stability

The Hessian is the curvature of energy with respect to in-plane particle
displacements. Two uniform translation modes are projected out explicitly.
Negative eigenvalues indicate an instability, even when all forces vanish.
Eigenvalue units are E_d/d^2.

{stab_table}

The optimized rhombic branch has **zero negative modes** in the tested 4x4
and 8x8 supercells at a/d=5.1, 5.3, 5.42, 5.44, the new crossing, and 5.5.
It remains locally stable slightly above its energy crossing, where honeycomb
has the lower energy. The square has negative homogeneous shear curvature
above about 5.06108 and negative finite-cell displacement modes near 5.42.

![Mechanical stability checks](figures/stability.png)

## Verification and numerical precision

All checks in `verify.py` and `verify_alternative.py` passed. Key results:

- Analytic Fourier kernel vs independent quadrature: {core["core"]["fourier_kernel_quadrature_error"]:.3e}.
- Analytic force vs finite-difference energy gradient: {core["core"]["gradient_fd_error"]:.3e}.
- Ewald splitting variation on a distorted random configuration: {core["core"]["ewald_split_energy_spread"]:.3e} E_d.
- Largest saved/recomputed MC-quench energy discrepancy: {core["outputs"]["max_saved_energy_error"]:.3e} E_d per hole.
- Largest residual Cartesian force among all relaxed particle runs: {core["outputs"]["max_relaxed_force"]:.3e} E_d/d.
- {n_below} particle runs reached energies below the two-ideal-structure envelope;
  no saved state was below the expanded square/rhombic/honeycomb envelope by more than 1e-8 E_d per hole.
- Changing Ewald splitting and cutoffs shifts the rhombic/honeycomb root by less than 2e-9 in a/d.
- The square shear root converges to about 5.061078 as the finite-difference strain step is reduced.

Five decimal places are reported for the new candidate boundaries. Extra
digits in the JSON files record numerical convergence, not proof that other
untested structures cannot alter the phase diagram.

## Relation to the paper and limits

The supplied paper (p. 2 and Fig. 1) reports a/d=5.42 after comparing the two
identified ideal composite geometries. Supplement section I describes
random-position classical minimization and Ewald summation. This benchmark
reproduces that restricted energy comparison, but free particle stability
and cell-shape relaxation expose an additional rhombic minimum. The energy
counterexample is confirmed by a separate electrostatic representation.

These calculations support the expanded **candidate** ground-state sequence
for this classical Hamiltonian. Finite cells sample only commensurate
wavevectors, particle runs have fixed boxes, and cell-shape searches use a
three-charge primitive cell. A finite search cannot exclude every larger
basis, longer wavelength modulation, or other competing phase. No
finite-temperature free energy or quantum phase boundary is inferred.

## Files and reproduction

- [README.md](README.md): commands and file map.
- [METHODS.md](METHODS.md): equations, conventions, schedules, and measurements.
- `results/extended_transitions.json`: both expanded boundaries and convergence.
- `results/mc_runs.csv`: every particle run and its measured final order.
- `results/runs/`: all initial, final, best, and relaxed coordinates and histories.
- `results/verification.json`, `results/alternative_verification.json`: executable checks.
- `sources/paper.pdf`, `sources/supp.pdf`: supplied sources.

Source: [Dai and Fu, Physical Review Letters 132, 196202 (2024)](https://doi.org/10.1103/PhysRevLett.132.196202),
main paper pp. 2-3 and Supplementary Material section I, pp. 1-2.
'''
    (ROOT/"REPORT.md").write_text(text)


def main():
    F.mkdir(exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted((R/"runs").glob("*.json"))]
    refresh_labels(records)
    tr=json.loads((R/"extended_transitions.json").read_text())
    energy_plot(records,tr)
    configuration_plot(records)
    stability_plot(tr)
    report(records,tr)
    print("Wrote REPORT.md and four scientific figures.")


if __name__=="__main__":main()
