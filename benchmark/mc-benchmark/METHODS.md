# Model, numerical method, and interpretation

## 1. Source and classical limit

David D. Dai and Liang Fu, *Strong-Coupling Phases of Trions and Excitons in
Electron-Hole Bilayers at Commensurate Densities*, Physical Review Letters
132, 196202 (2024), DOI [10.1103/PhysRevLett.132.196202](https://doi.org/10.1103/PhysRevLett.132.196202).
The supplied main paper is `sources/paper.pdf`; the supplied supplement is
`sources/supp.pdf` (dated January 29, 2024).

The main paper p. 2 defines the classical limit, density, and the two named
composite crystals. Its Fig. 1 on p. 3 gives their energy comparison. The
supplement section I, pp. 1-2, gives Ewald Eqs. S1-S4 and describes random
initializations with 128 electrons and 64 holes, followed by precise
defect-free lattice energies. **That source method is classical energy
minimization; our Metropolis annealing schedules are new implementation
choices, not settings supplied by the authors.** Quantum Hartree-Fock and
quantum Monte Carlo sections are outside this classical calculation.

We take the kinetic energy to zero at fixed layer separation and density.
All particles are classical points, with immutable species and layer. Spin,
Fermi statistics, masses, zero-point motion, and exchange do not enter.
The interactions are the paper's unscreened Coulomb interactions.

## 2. Units, density, and energy

Let `d=1` and measure energy in `E_d=e^2/(4*pi*epsilon*d)`.
The in-plane coordinates of electrons and holes remain independent.

\[
U=\frac12\sum_{i\ne j}^{N_e}\frac1{|\mathbf r_i^e-\mathbf r_j^e|}
 +\frac12\sum_{i\ne j}^{N_h}\frac1{|\mathbf r_i^h-\mathbf r_j^h|}
 -\sum_{i,j}\frac1{\sqrt{|\mathbf r_i^e-\mathbf r_j^h|^2+1}},
\]

with periodic image sums and the background corrections below.
The density convention is

\[
N_e=2N_h,\qquad n_h=(\pi a^2)^{-1},\qquad A=N_h\pi a^2.
\]

All energies in the tables and main plots are **U/N_h**, equivalently energy
per 2-electron/1-hole formula unit, in `E_d`. Divide by three to obtain energy
per individual charge. A positive `honeycomb - square` favors square.

## 3. Periodic Ewald energy

Rows of the 2 by 2 matrix `C` are real-space cell vectors. A fractional row
coordinate `f` gives `r=f C`. Reciprocal rows are `B=2*pi*C^(-T)`.
Periodicity is in the plane only; there is no artificial repetition in z.
Each layer has a neutralizing background, implemented by removing its
`G=0` Fourier mode, consistently with supplement S4 and S24.
No additional capacitor constant is included.

Set `R=sqrt(r^2+z^2)`, where `z` is 0 or 1. The short-range potential is

\[
v_s(R)=\frac{\operatorname{erfc}(\alpha R)}{R}.
\]

The **full separation R**, not the in-plane distance alone, enters erfc.
Using the Gaussian integral representation, the Fourier transform of the
long-range piece can be evaluated analytically:

\[
C_z(G)=\frac{\pi}{G}\left[
 e^{Gz}\operatorname{erfc}\!\left(\frac{G}{2\alpha}+\alpha z\right)
 +e^{-Gz}\operatorname{erfc}\!\left(\frac{G}{2\alpha}-\alpha z\right)\right].
\]

Equivalently,

\[
C_z(G)=2\sqrt\pi\int_0^\alpha
 t^{-2}\exp[-z^2t^2-G^2/(4t^2)]\,dt.
\]

For z=0 this is `2*pi*erfc(G/(2*alpha))/G`. This analytic form implements
the same split as supplement S1-S3; it replaces the supplement's numerical
Bessel transform. Independent quadrature checks it. The potentially large
`exp(G*z)` product is evaluated using the scaled complementary error function.

With unsigned structure factors `S_e=sum_e exp(iG.r_e)` and `S_h` likewise,

\[
U_{\rm recip}=\frac1{2A}\sum_{G\ne0}\left[
 C_0(G)(|S_e|^2+|S_h|^2)-2C_1(G)\Re(S_eS_h^*)\right].
\]

The real-space part is the signed pair sum of `v_s`; self interaction at
zero displacement is excluded, and each particle's interactions with its
nonzero lattice images are included. The other terms are

\[
U_{\rm self}=-\frac{\alpha}{\sqrt\pi}(N_e+N_h),
\]

\[
V_s(z)=2\pi\left[\frac{e^{-\alpha^2z^2}}{\alpha\sqrt\pi}
 -z\operatorname{erfc}(\alpha z)\right],
\]

\[
U_{\rm back}=-\frac{V_s(0)(N_e^2+N_h^2)-2V_s(1)N_eN_h}{2A}.
\]

These include point-background and background-background contributions
without double counting. Individual intermediate terms depend on alpha;
the converged total does not. Independent uniform-background conventions
can add a common density-dependent constant, which does not change a
comparison at the same A and particle numbers.

The production choice is `alpha=4/min(cell heights)`, analogous to the
supplement's `r0=1/alpha` near one quarter of the smallest dimension. We
retain full separation `R <= 6/alpha` and reciprocal magnitude
`G <= 12*alpha`. Index bounds enclose these circles even in a skew cell.
We also use `(split,accuracy)=(3,5)` and `(5,7)` for convergence, and compare
primitive cells with their 2 by 2, 4 by 4, and 8 by 8 replications.

The alternative energy implementation uses a separate intralayer Ewald
sum and the **unsplit** interlayer kernel `2*pi*exp(-G)/G` through G=40.
This avoids sharing the bilayer split with the primary implementation.

## 4. Candidate geometries and the extra degree of freedom

Every primitive cell has an electron and a hole at the origin (a vertical
dipole) and one excess electron. With primitive area `A0=pi*a^2`:

* Square: orthogonal vectors of length sqrt(A0), excess electron at
  `(a1+a2)/2`.
* Honeycomb: equal vectors at 60 degrees, length sqrt(2*A0/sqrt(3)), excess
  electron at `(a1+a2)/3`.
* Rhombic: equal vectors at angle theta, length sqrt(A0/sin(theta)), excess
  electron at `(a1+a2)/2`. Minimize theta on [60,90] degrees.

The square is the theta=90 endpoint of the rhombic family. The honeycomb
has a different basis, so reaching theta=60 within the midpoint family
does not create the honeycomb.

The two-ideal-structure crossing is found by a sign-bracketed root solve.
The rhombic/honeycomb crossing uses the minimized rhombic energy in the
same root solve. Bisection/root tolerance is distinguished from uncertainty
about additional possible phases.

`explore_cell.py` also relaxes all six independent primitive-cell parameters:
two fixed-area cell-shape parameters plus four relative fractional particle
coordinates (one electron fixes the arbitrary origin). It does not enforce
vertical dipoles, a midpoint excess electron, equal vector lengths, or a
named lattice. Randomly perturbed starts from both named lattices find the
same square/rhombic/honeycomb branches, including equivalent cell bases.
The shape parametrization is `C=sqrt(A)*[[exp(u),0],[v,exp(-u)]]`, with
search bounds `-0.6 <= u <= 0.6` and `-1.5 <= v <= 1.5`; the four internal
fractional coordinates are unbounded and periodic. These finite shape bounds
are an additional search restriction, not a proof about more extreme cells.
`cell_mc.py` adds 15 random-start basin-hopping searches, each with 80
quenched Metropolis proposals at an optimization temperature 0.0003 E_d.
Every proposed minimum and accept/reject outcome is saved. This sampling
operates over primitive-cell structures; its frequencies are not physical
phase probabilities or finite-temperature free energies.

## 5. Particle Monte Carlo

The particle simulations use 4 by 4 and 8 by 8 primitive-cell repetitions:
`(Ne,Nh)=(32,16)` or `(128,64)`, hence 48 or 192 individual charges.
Their cell shape and area stay fixed during a particle run.

Each attempted Metropolis move is either:

1. With probability 0.6, translate one uniformly chosen charge by a uniform
   displacement in a centered square.
2. With probability 0.4, translate a hole and its assigned electron by the
   same displacement. Proposal partners are obtained by an initial
   minimum-distance one-to-one assignment and **held fixed for the run**.

Both proposals are symmetric. The acceptance probability is
`min(1,exp(-Delta U/T))`. Single moves act on all particles, so paired
proposals do not constrain binding, colocation, partner exchange, or motion.
The paired move's internal real-space pair energy is unchanged; its full
reciprocal contribution, including joint quadratic terms, is retained.
Structure factors are updated on accepted moves. At every temperature stage,
the full energy and structure factors are recomputed to measure numerical drift.

One sweep is `Ne+Nh` attempted moves, with replacement. Schedules are geometric:

| Suite | Initialization | Stages x sweeps | T_max | T_min |
|---|---|---:|---:|---:|
| Original seeded comparison | independent Gaussian displacement, sigma=0.04*a | 32 x 100 | 0.006 | 2e-7 |
| Random search | uniform independent charge positions in cell | 40 x 200 | 0.1 | 2e-7 |
| Rhombic/new-boundary check | independent Gaussian displacement, sigma=0.04*a | 32 x 100 | 0.0005 | 2e-7 |

Move sizes adapt between stages toward acceptance 0.38. Initial single and
paired half-widths are 0.25*d and 0.18*a. The lower-temperature added suite
tests recovery of the closely competing crystals while the hotter runs
probe escape and defect formation. These are annealing optimizers, not
stationary equilibrium chains, so their histories are not averaged to
estimate a free-energy transition.

The exact run list and random seeds are generated by `cases('all')` and
saved in manifests. Each trajectory saves its initial state, lowest visited
MC state, final MC state, and temperature-stage history.

## 6. Zero-temperature relaxation and configuration classification

Both the best visited and final MC configurations are relaxed separately
with L-BFGS-B and analytic position gradients. The lower result is reported.
All positions move independently. The stopping settings are `gtol=2e-10`,
`ftol=1e-15`, maximum 6000 iterations; the actual maximum Cartesian force is
saved because the optimizer can terminate on energy tolerance before gtol.
The verification checks force residuals and recomputed saved energies.
If the residual force exceeds 1e-6 E_d/d, `run_benchmark.py refine` continues
that saved quench for up to 20,000 further iterations. It preserves the
pre-refinement positions, energy, and residual; it does not rerun MC or
replace the saved trajectory.
For the remaining soft defective states, this continuation uses an invertible
change of optimization coordinates: pair center of mass is scaled by a^(5/2),
relative displacement is unscaled, and excess-electron position is scaled by
a^(3/2), consistent with the different stiffness scales. All six coordinates
per electron-electron-hole formula unit remain free; this is a numerical
preconditioner, not a rigid-dipole approximation. Actual Cartesian residual
forces are still the verification criterion. The rescaling, iteration count,
status, energy, and actual force residual are recorded with each refinement.
The scaled-coordinate L-BFGS-B settings are `gtol=2e-9`, `ftol=1e-15`,
`maxcor=40`, `maxls=40`, and at most 20,000 iterations.

We pair holes to electrons one-to-one for **measurement only**, report the
in-plane electron-hole RMS distance, and measure bond order on the hole and
excess-electron sublattices:

\[
\psi_m=\left|\frac1{mN_h}\sum_i\sum_{j\in m\ {\rm nearest}(i)}e^{im\theta_{ij}}\right|,
\quad m=4,6.
\]

The final `final_order` labels are descriptive threshold classifications,
not proof of a thermodynamic phase. A rhombus and a distorted square can
both have strong fourfold order. Therefore final phase assignment in the
report additionally uses cell geometry, energy relative to each branch,
direct snapshots, and force/stability checks. Initial lattice labels alone
are never treated as final phases. Random starts can retain defects; their
energy excess is reported and they are not substituted for precise lattice
energies.

## 7. Mechanical stability and the important distinction from an energy crossing

We form the Cartesian Hessian by centered differences of analytic gradients,
with step 0.0002*d. A positive Hessian means small displacements cost energy.
We explicitly project out the two uniform translations with a null-space
basis before taking eigenvalues; deleting the lowest two eigenvalues would
incorrectly hide unstable modes. Step variation and translational residuals
are checked. The 4 by 4 and 8 by 8 cells probe their respective discrete
commensurate wavevectors, not every wavevector of an infinite system.

Separately, area-preserving homogeneous strain is

\[
C(s,t)=C_0\exp\begin{pmatrix}s&t\\t&-s\end{pmatrix}.
\]

This includes shape variations unavailable in a fixed simulation box. For
square and midpoint-rhombic structures, inversion symmetry leaves every
particle force zero at fixed fractional coordinates along the strain path.
The square shear curvature changes sign near a/d=5.06108. Its root is checked
with five-point second derivatives and several strain steps. The rhombic
angle then departs continuously from 90 degrees. Near a/d=5.46436, rhombic
and honeycomb are both local minima but their energies cross, with an angle
jump from about 79.99 degrees to the honeycomb's 60 degrees.

Thus a force-balanced ideal crystal can be a saddle rather than a local
minimum, and persistence in a fixed box does not establish stability to
homogeneous shear. This is the mechanism behind the discrepancy with the
two-ideal-structure comparison.

## 8. Scope of the result

The converged ideal crossing is reproduced; the additional rhombic structure
is a quantitative counterexample to identifying it with the unrestricted
ground-state boundary for the stated classical model. Particle MC, primitive
cell-shape searches, alternative electrostatics, and finite-cell stability
support the extended candidate sequence. A finite search cannot exclude
all larger bases, longer wavelength modulations, phase coexistence, or
other parameters. No quantum or finite-temperature phase boundary is claimed.
