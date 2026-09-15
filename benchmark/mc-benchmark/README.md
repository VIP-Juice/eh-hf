# Classical Monte Carlo benchmark: Dai-Fu electron-hole bilayer

## Main result

The ideal square checkerboard and ideal honeycomb energies cross at
**a/d = 5.42240068**, reproducing Fig. 1 of Dai and Fu, PRL **132**, 196202 (2024).

However, a free stability calculation finds that the square checkerboard
has a shear instability before that crossing. An optimized **rhombic
composite crystal** has lower energy than both ideal structures nearby.
For the tested candidates, the resulting sequence is

**square -- 5.06108 --> rhombic -- 5.46436 --> honeycomb.**

These are zero-temperature energy comparisons, supported by particle Monte
Carlo, cell-shape basin hopping, and finite-supercell Hessians. They are not
a mathematical proof excluding every possible larger-unit-cell structure.
The paper's 5.42 value must be described as the crossing of the two ideal
geometries; it is not the unrestricted boundary found by this benchmark.

Read **[REPORT.md](REPORT.md)** for numerical evidence, plots, Monte Carlo
results, convergence, and limitations, and **[METHODS.md](METHODS.md)** for
the complete Hamiltonian, Ewald conventions, moves, and stability tests.

## Reproduce

Python 3.11 was used. The scripts are independent of the working directory
and of the sibling `../independent_code/` implementation.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -B run_benchmark.py analytic
.venv/bin/python -B run_benchmark.py stability
.venv/bin/python -B explore_cell.py
.venv/bin/python -B phase_analysis.py
.venv/bin/python -B run_benchmark.py mc --suite all --workers 4
.venv/bin/python -B run_benchmark.py refine
.venv/bin/python -B cell_mc.py
.venv/bin/python -B verify_alternative.py
.venv/bin/python -B verify.py
.venv/bin/python -B make_report.py
```

Run `verify.py --core-only` for a short engine check before starting the
Monte Carlo jobs. Particle Monte Carlo jobs save each completed run and skip
existing case IDs on restart. Existing run records must be removed or moved
to a separate directory before deliberately rerunning changed algorithms or
schedules. Do not mix outputs from modified code with an earlier dataset.

## Contents

| File | Role |
|---|---|
| `ewald.py` | Full 2D-periodic bilayer energy, forces, local relaxation, Hessian |
| `monte_carlo.py` | Metropolis annealing of individual charges and paired translations |
| `run_benchmark.py` | Ideal branches, seeded/random particle runs, order measurements |
| `explore_cell.py` | Six-parameter primitive-cell relaxation without square/honeycomb constraints |
| `phase_analysis.py` | Optimized rhombic branch and both corrected candidate boundaries |
| `cell_mc.py` | Randomly initialized cell-shape basin-hopping Monte Carlo |
| `reference_energy.py` | Standalone copy of the earlier independent energy-only method |
| `verify_alternative.py` | Countercheck with unsplit interlayer Fourier summation |
| `verify.py` | Formula, derivative, convergence, and stored-output checks |
| `make_report.py` | Regenerate figures, comparison tables, and the report |
| `results/runs/*.npz` | Initial, MC final/best, and relaxed coordinates; cell, species, history |
| `results/runs/*.json` | Parameters, energies, force residuals, structure measurements |
| `results/cell_mc/*.json` | Every proposed cell-search minimum and acceptance decision |
| `sources/` | Supplied PDFs, extracted text, and provenance |

The original research repository files and previous benchmark are unchanged.
