# Fig. 3 benchmark using the repository HF notebook

This benchmark executes the HF implementation directly from
`../../Asymmetric-Bilayers----Hartree-Fock.ipynb`.  It does not copy the
notebook's `CSD!`, `Fock!`, `energy_calc!`, `grad_calc!`, or coefficient-vector
conversion functions.  The Python launcher extracts and evaluates notebook
cells 0, 2, 3, 4, 6, and 7 in a fresh Julia process and records SHA-256 hashes
of both the notebook and the HF-engine cell in every result.

## What the adapter changes

The notebook is configured for a different problem, so the launcher replaces
only its physical setup with the Dai--Fu Fig. 3 values:

- `numbers = [36, 36, 36]` for electron up, electron down, and polarized holes;
- `masses = [2, 2, 1]` in hole-mass units;
- `rsM = 8` and `d = 4.83` in hole-Bohr-radius units;
- locked electron-spin coefficient matrices;
- a deterministic 6 by 6 triangular, colocated trion initialization.

The notebook's native oblique triangular cell and hexagonal momentum cutoff
are retained.  Its default `Rk=7` basis has only 169 plane waves, compared with
the paper's 45 by 39 rectangular basis (1755 plane waves).  The saved cutoff
sweep therefore distinguishes agreement of the HF engine from basis-cutoff
agreement with the published number.

Notebook cells 8 and 9 are not executed.  Cell 8 initializes the notebook's
different physical problem, while cell 9 is not executable in a fresh kernel
because it assigns `long_vec = sol.minimizer` before `sol` has been created.
The driver supplies the Fig. 3 initialization and then calls the same
`Optim.ConjugateGradient()` minimization used in cell 9.

## Run

From the repository root:

```bash
julia --project=benchmark/existing_code -e 'using Pkg; Pkg.instantiate()'
python3 -B benchmark/existing_code/run_existing_notebook_fig3.py
```

For a cutoff run, use for example:

```bash
python3 -B benchmark/existing_code/run_existing_notebook_fig3.py \
  --rk 10 --output-dir benchmark/existing_code/results/basis_sweep/rk10
```

After producing `Rk=7,8,9,10,11,12`, verify and summarize them with:

```bash
python3 -B benchmark/existing_code/verify.py
```

The default result directory contains the density CSV/NPZ/PNG, Julia
checkpoint, source hashes, optimizer diagnostics, density integrals, gradient
finite-difference check, and the generated basis-convergence CSV/PNG.

## Saved result

All saved runs meet the supplement's `1e-5` maximum-gradient threshold, contain
36 colocated electron and hole density peaks, and integrate to 72 electrons and
36 holes.  The energy decreases toward the paper value over the tested native
hexagonal cutoffs:

| `Rk` | plane waves | energy/particle | difference from `-0.07652` |
| ---: | ---: | ---: | ---: |
| 7 | 169 | -0.07589830 | 6.217e-4 |
| 8 | 217 | -0.07613509 | 3.849e-4 |
| 9 | 271 | -0.07627939 | 2.406e-4 |
| 10 | 331 | -0.07632911 | 1.909e-4 |
| 11 | 397 | -0.07633933 | 1.807e-4 |
| 12 | 469 | -0.07634106 | 1.789e-4 |

This is a successful benchmark of the repository HF engine, but it is not an
exact rerun of the paper's finite basis, and this limited sweep is not an
infinite-cutoff extrapolation.  `Rk=12` is the first tested cutoff that includes
the second primitive reciprocal shell of the 6 by 6 trion lattice.  The
notebook explicitly allocates an
`Nk x Nk x Nk` `Int16` momentum-lookup array and evaluates cubic nested sums.
At the paper's 1755 plane waves that lookup alone would occupy about 10.1 GiB,
before the HF optimization begins.  The remaining energy offset can therefore
contain both momentum-cutoff error and the finite-cell-shape difference between
the notebook's oblique cell and the paper's rectangular cell.
