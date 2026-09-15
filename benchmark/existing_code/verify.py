#!/usr/bin/env python3
"""Verify the Dai--Fu Fig. 3 benchmark driven by the repository HF notebook."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import tomllib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[1] / "Asymmetric-Bilayers----Hartree-Fock.ipynb"
RESULTS = HERE / "results"
PAPER_ENERGY = -0.07652
EXPECTED_RUNS = ((7, 169), (8, 217), (9, 271), (10, 331), (11, 397), (12, 469))
RUNNER = HERE / "run_existing_notebook_fig3.py"
DRIVER = HERE / "existing_notebook_driver.jl"
ARCHIVED_RUNNER = RESULTS / "provenance" / "run_existing_notebook_fig3.py"
ARCHIVED_DRIVER = RESULTS / "provenance" / "existing_notebook_driver.jl"


def periodic_peak_count(density: np.ndarray) -> int:
    is_maximum = np.ones(density.shape, dtype=bool)
    for shift_v in (-1, 0, 1):
        for shift_u in (-1, 0, 1):
            if shift_u == 0 and shift_v == 0:
                continue
            neighbor = np.roll(np.roll(density, shift_v, axis=0), shift_u, axis=1)
            is_maximum &= density >= neighbor
    is_maximum &= density > 0.8 * density.max()
    return int(is_maximum.sum())


def run_directory(rk: int) -> Path:
    return RESULTS if rk == 7 else RESULTS / "basis_sweep" / f"rk{rk}"


def normalized_adapter_source(path: Path, kind: str) -> str:
    source = path.read_text()
    if kind == "runner":
        source, count = re.subn(
            r'^\s*parser\.add_argument\("--output-dir"[^\n]+$',
            '    parser.add_argument("--output-dir", type=Path, default=<OUTPUT_DIR>)',
            source,
            flags=re.MULTILINE,
        )
    elif kind == "driver":
        source, count = re.subn(
            r'^output_dir = abspath\(argument_value\("--output-dir", "[^"]+"\)\)$',
            'output_dir = abspath(argument_value("--output-dir", "<OUTPUT_DIR>"))',
            source,
            flags=re.MULTILINE,
        )
    else:
        raise ValueError(f"unknown adapter source kind: {kind}")
    if count != 1:
        raise AssertionError(f"could not normalize {kind} output path in {path}")
    return source


def load_run(
    rk: int,
    expected_nk: int,
    notebook_hash: str,
    engine_hash: str,
    runner_hash: str,
    driver_hash: str,
) -> dict[str, float]:
    directory = run_directory(rk)
    with (directory / "artifact_hashes.json").open() as stream:
        artifact_hashes = json.load(stream)
    for name, expected_hash in artifact_hashes.items():
        actual_hash = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"Rk={rk} saved artifact changed after generation: {name}")
    with (directory / "metrics.toml").open("rb") as stream:
        metrics = tomllib.load(stream)
    parameters = metrics["parameters"]
    result = metrics["result"]
    source = metrics["source"]

    if source["notebook_sha256"] != notebook_hash:
        raise AssertionError(f"Rk={rk} result did not come from the current notebook")
    if source["hf_engine_cell_sha256"] != engine_hash:
        raise AssertionError(f"Rk={rk} result did not use the current notebook HF cell")
    if source["runner_sha256"] != runner_hash or source["driver_sha256"] != driver_hash:
        raise AssertionError(f"Rk={rk} result did not use the current benchmark adapter")
    if parameters["momentum_cutoff_Rk"] != rk or parameters["number_plane_waves"] != expected_nk:
        raise AssertionError(f"Rk={rk} basis metadata is inconsistent")
    expected_parameters = {
        "n_electrons_up": 36,
        "n_electrons_down": 36,
        "n_holes": 36,
        "species": ["e↑", "e↓", "h"],
        "layers": [1, 1, 2],
        "spin_locked": True,
        "electron_mass_in_hole_units": 2.0,
        "hole_mass_in_hole_units": 1.0,
        "d_over_aB_h": 4.83,
        "hole_rs": 8.0,
    }
    for key, expected in expected_parameters.items():
        if parameters[key] != expected:
            raise AssertionError(f"Rk={rk} has {key}={parameters[key]}, expected {expected}")
    if not result["converged"] or result["max_gradient_component"] >= 1.0e-5:
        raise AssertionError(f"Rk={rk} did not meet the paper's gradient tolerance")
    if abs(result["integrated_electron_density"] - 72.0) > 5.0e-5:
        raise AssertionError(f"Rk={rk} electron density is not normalized to 72")
    if abs(result["integrated_hole_density"] - 36.0) > 5.0e-5:
        raise AssertionError(f"Rk={rk} hole density is not normalized to 36")

    density = np.load(directory / "density.npz")
    electron_density = density["electron_density"]
    hole_density = density["hole_density"]
    density_weight = parameters["area"] / electron_density.size
    integrated_electrons = density_weight * float(electron_density.sum())
    integrated_holes = density_weight * float(hole_density.sum())
    if abs(integrated_electrons - 72.0) > 5.0e-5:
        raise AssertionError(f"Rk={rk} saved electron array is not normalized to 72")
    if abs(integrated_holes - 36.0) > 5.0e-5:
        raise AssertionError(f"Rk={rk} saved hole array is not normalized to 36")
    if abs(integrated_electrons - result["integrated_electron_density"]) > 1.0e-10:
        raise AssertionError(f"Rk={rk} saved and reported electron integrals disagree")
    if abs(integrated_holes - result["integrated_hole_density"]) > 1.0e-10:
        raise AssertionError(f"Rk={rk} saved and reported hole integrals disagree")
    electron_peaks = periodic_peak_count(electron_density)
    hole_peaks = periodic_peak_count(hole_density)
    if electron_peaks != 36 or hole_peaks != 36:
        raise AssertionError(
            f"Rk={rk} density has {electron_peaks} electron and {hole_peaks} hole peaks"
        )
    correlation = float(np.corrcoef(electron_density.ravel(), hole_density.ravel())[0, 1])
    if correlation < 0.95:
        raise AssertionError(f"Rk={rk} electron and hole peaks are not colocated")
    electron_modulation = float(electron_density.std() / electron_density.mean())
    hole_modulation = float(hole_density.std() / hole_density.mean())
    modulation_ratio = hole_modulation / electron_modulation
    if modulation_ratio < 2.0:
        raise AssertionError(f"Rk={rk} holes are not appreciably narrower than electrons")
    if rk == EXPECTED_RUNS[-1][0]:
        if not (0.014 < electron_density.max() < 0.018):
            raise AssertionError("highest-cutoff electron density scale disagrees with Fig. 3")
        if not (0.014 < hole_density.max() < 0.018):
            raise AssertionError("highest-cutoff hole density scale disagrees with Fig. 3")

    return {
        "Rk": rk,
        "plane_waves": expected_nk,
        "energy_per_particle": float(result["energy_per_particle_hartree_h"]),
        "paper_error": abs(float(result["energy_per_particle_hartree_h"]) - PAPER_ENERGY),
        "max_gradient_component": float(result["max_gradient_component"]),
        "electron_hole_correlation": correlation,
        "hole_electron_modulation_ratio": modulation_ratio,
        "electron_peaks": electron_peaks,
        "hole_peaks": hole_peaks,
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def main() -> None:
    raw = NOTEBOOK.read_bytes()
    notebook = json.loads(raw)
    engine_source = "".join(notebook["cells"][7]["source"])
    notebook_hash = hashlib.sha256(raw).hexdigest()
    engine_hash = hashlib.sha256(engine_source.encode()).hexdigest()
    runner_hash = hashlib.sha256(ARCHIVED_RUNNER.read_bytes()).hexdigest()
    driver_hash = hashlib.sha256(ARCHIVED_DRIVER.read_bytes()).hexdigest()
    if normalized_adapter_source(RUNNER, "runner") != normalized_adapter_source(
        ARCHIVED_RUNNER, "runner"
    ):
        raise AssertionError("current runner differs numerically from the result-producing source")
    if normalized_adapter_source(DRIVER, "driver") != normalized_adapter_source(
        ARCHIVED_DRIVER, "driver"
    ):
        raise AssertionError("current driver differs numerically from the result-producing source")
    rows = [
        load_run(rk, nk, notebook_hash, engine_hash, runner_hash, driver_hash)
        for rk, nk in EXPECTED_RUNS
    ]

    energies = np.array([row["energy_per_particle"] for row in rows])
    if np.any(np.diff(energies) >= 0.0):
        raise AssertionError("energy is not strictly decreasing with the notebook cutoff")
    if rows[-1]["paper_error"] >= rows[0]["paper_error"]:
        raise AssertionError("the limited cutoff study does not approach the paper energy")

    with (RESULTS / "metrics.toml").open("rb") as stream:
        default_metrics = tomllib.load(stream)
    derivative_relative_error = default_metrics["result"]["finite_difference_relative_error"]
    if derivative_relative_error > 5.0e-3:
        raise AssertionError(
            f"notebook analytic gradient relative error is {derivative_relative_error:.3e}"
        )

    columns = tuple(rows[0])
    with (RESULTS / "basis_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    plane_waves = np.array([row["plane_waves"] for row in rows])
    figure, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    axis.plot(plane_waves, energies, "o-", label="repository notebook engine")
    axis.axhline(PAPER_ENERGY, color="black", linestyle="--", label="paper")
    axis.set_xlabel("number of plane waves")
    axis.set_ylabel(r"energy per particle [$\mathrm{Ha}_h$]")
    axis.legend()
    figure.savefig(RESULTS / "basis_convergence.png", dpi=220)
    plt.close(figure)

    for row in rows:
        print(
            f"Rk={row['Rk']:2d}, Nk={row['plane_waves']:3d}: "
            f"E/N={row['energy_per_particle']:+.8f}, "
            f"paper error={row['paper_error']:.3e}, "
            f"max|gradient|={row['max_gradient_component']:.3e}"
        )
    print(f"default-run gradient finite-difference relative error: {derivative_relative_error:.3e}")
    print("all existing-notebook benchmark checks passed")


if __name__ == "__main__":
    main()
