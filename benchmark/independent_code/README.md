# Independent Dai-Fu 2024 benchmark

This directory reproduces the two numerical results requested from D. D. Dai
and L. Fu, *Phys. Rev. Lett.* **132**, 196202 (2024):

1. the classical checkerboard-versus-honeycomb energy comparison in Fig. 1;
2. the restricted Hartree-Fock electron and hole densities in Fig. 3.

`classical_fig1.py` and `hartree_fock_fig3.py` are independent of the two
notebooks in the repository and do not change their asymmetric-bilayer
defaults.

## Run

From the repository root:

```bash
python3 benchmark/independent_code/classical_fig1.py
python3 benchmark/independent_code/hartree_fock_fig3.py
python3 benchmark/independent_code/verify.py
```

The classical calculation takes about a second.  A clean run of the published
`45 x 39` plane-wave HF calculation takes roughly two minutes on the
development machine; the saved result reached the paper's maximum-gradient
criterion in 182 direct-minimization steps.  For a fast algorithmic smoke test, use:

```bash
python3 benchmark/independent_code/hartree_fock_fig3.py --quick --iterations 5 --output-dir /tmp/eh-hf-smoke
```

`--resume` continues from `benchmark/independent_code/results/fig3_checkpoint.npz`.
The default run
starts cleanly from translated Gaussian orbitals on the expected 6 x 6
triangular trion lattice.

## Fig. 1: classical electrostatics

The length unit is `d`, the Coulomb-energy unit is
`e^2/(4*pi*epsilon*d)`, and `a = 1/sqrt(pi*n_h)`.  Each primitive cell contains
one hole, an electron at the same in-plane coordinate, and one excess electron.

- Checkerboard: a square Bravais cell of area `pi*a^2`, with the excess
  electron displaced by `(a1+a2)/2` from the dipole.
- Honeycomb: a triangular Bravais cell of the same area, with the excess
  electron displaced by `(a1+a2)/3`.

`classical_fig1.py` evaluates the intralayer terms with the 2D Ewald split in
Supplemental Eqs. S1-S4.  The split length is one quarter of the shortest cell
vector.  It evaluates the finite-separation interlayer term with
`v(q) = 2*pi*exp(-q*d)/q`.  The `q=0` component is removed, which is the
uniform neutralizing-background convention used in the paper.  No number is
fit to the published crossover.

The result is

```text
a/d = 5.422401
```

compared with `5.42` in the paper.

## Fig. 3: restricted Hartree-Fock

`hartree_fock_fig3.py` implements Supplemental Eqs. S13-S25 in a rectangular
plane-wave basis.  Products and Coulomb convolutions use a `(2*Nx-1) x
(2*Ny-1)` auxiliary FFT grid, so every difference of retained plane waves is
represented without wraparound aliasing.  The two electron-spin species share
one occupied subspace (restricted HF); the hole species is spin polarized.
Exchange is included only within each identical fermion species.

The defaults are exactly the published calculation parameters:

| parameter | value |
| --- | ---: |
| `m_e` | `0.8 m0 = 2 m_h` |
| `m_h` | `0.4 m0` |
| `d` | `4.83 a_B,h` |
| hole `r_s` | `8` |
| `N_h` | `36` |
| `N_e,up = N_e,down` | `36` |
| plane-wave basis | `45 x 39` |
| box aspect `Ly/Lx` | `sqrt(3)/2` |

The converged benchmark gives

```text
E/N = -0.07652262 E_h,h
```

versus `-0.07652 E_h,h` in the Supplemental Material.  The largest restricted
real-coordinate gradient component is `9.88e-6`, below the paper's `1e-5`
stopping threshold.  The density integrals are `N_e=72` and `N_h=36`; both
density maxima are near `0.016/a_B,h^2`, on the same scale as Fig. 3.  The
broader electron peaks and narrower colocated hole peaks form the published
6 x 6 triangular spin-singlet-trion crystal.

## Outputs

- `results/fig1_classical.png`: classical energies and their difference.
- `results/fig1_classical.csv`: numerical Fig. 1 curve.
- `results/fig3_density.png`: common-scale electron/hole density panels.
- `results/fig3_density_data.npz`: densities and converged occupied subspaces.
- `results/fig3_metrics.json`: parameters, energy decomposition, convergence
  history, density normalization, and comparison value from the paper.
- `results/fig3_checkpoint.npz`: resumable HF checkpoint.

The benchmark deliberately reproduces the defect-free, trion-initialized RHF
minimum shown in the main text.  As discussed by the authors, random starts can
instead become trapped in slightly higher-energy defective local minima.
