#!/usr/bin/env python3
"""Run the repository HF notebook engine at the Dai--Fu Fig. 3 parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
NOTEBOOK = REPOSITORY / "Asymmetric-Bilayers----Hartree-Fock.ipynb"
CORE_CELLS = (0, 2, 3, 4, 6, 7)


def notebook_source() -> tuple[str, str, str]:
    """Extract the setup and HF-engine cells without copying them into the benchmark."""
    raw = NOTEBOOK.read_bytes()
    notebook = json.loads(raw)
    cells = notebook["cells"]
    sources = {index: "".join(cells[index]["source"]) for index in CORE_CELLS}

    required = ("function CSD!", "function Fock!", "function energy_calc!", "function grad_calc!")
    missing = [marker for marker in required if marker not in sources[7]]
    if missing:
        raise RuntimeError(f"notebook HF cell is missing expected definitions: {missing}")

    # Rk is a setup parameter, not part of the HF engine.  Keep the notebook's
    # value as the default while allowing basis-convergence runs from the CLI.
    basis_source, count = re.subn(
        r"(?m)^Rk\s*=\s*7\s*$",
        'Rk = parse(Int, get(ENV, "EH_HF_RK", "7"))',
        sources[3],
    )
    if count != 1:
        raise RuntimeError("could not locate the notebook's Rk setup line")

    fig3_override = r'''
# Parameter-only adapter for Dai--Fu Fig. 3.  The notebook definitions above
# are intentionally evaluated first, then their physical setup is replaced.
numbers = [36, 36, 36]
species = ["e\u2191", "e\u2193", "h"]
masses = [2.0, 2.0, 1.0]
layers = [1, 1, 2]
spin_locked = true
rsM = 8.0
d = 4.83
T = [1.0 0.5; 0.0 sqrt(3.0)/2]
L = rsM * sqrt(pi * numbers[3] / det(T))
A = L^2 * det(T)
'''

    generated = "\n\n".join(
        (
            sources[0],
            sources[2],
            fig3_override,
            basis_source,
            sources[4],
            sources[6],
            sources[7],
            f'include({json.dumps(str(HERE / "existing_notebook_driver.jl"))})',
        )
    )
    notebook_hash = hashlib.sha256(raw).hexdigest()
    engine_hash = hashlib.sha256(sources[7].encode()).hexdigest()
    return generated, notebook_hash, engine_hash


def plot_density(csv_path: Path, output_path: Path, nu: int, nv: int) -> None:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    u, v, x, y, electron, hole = (data[:, column].reshape(nv, nu) for column in range(6))
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        u=u,
        v=v,
        x=x,
        y=y,
        electron_density=electron,
        hole_density=hole,
    )

    vmax = max(float(electron.max()), float(hole.max()))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    images = []
    for axis, density, title in zip(
        axes, (electron, hole), ("Electron density", "Hole density"), strict=True
    ):
        image = axis.pcolormesh(x, y, density, cmap="hot", vmin=0.0, vmax=vmax, shading="nearest")
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel(r"$x/a_{B,h}$")
        axis.set_ylabel(r"$y/a_{B,h}$")
        axis.set_aspect("equal")
    colorbar = figure.colorbar(images[0], ax=axes, shrink=0.88)
    colorbar.set_label(r"density $[1/a_{B,h}^2]$")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--rk", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--fine-nu", type=int, default=120)
    parser.add_argument("--fine-nv", type=int, default=104)
    parser.add_argument("--seed", type=int, default=20240904)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated, notebook_hash, engine_hash = notebook_source()
    runner_hash = sha256(Path(__file__).resolve())
    driver_hash = sha256(HERE / "existing_notebook_driver.jl")
    environment = os.environ.copy()
    environment.update(
        {
            "EH_HF_RK": str(args.rk),
            "EH_HF_NOTEBOOK": str(NOTEBOOK),
            "EH_HF_NOTEBOOK_SHA256": notebook_hash,
            "EH_HF_ENGINE_SHA256": engine_hash,
            "EH_HF_RUNNER_SHA256": runner_hash,
            "EH_HF_DRIVER_SHA256": driver_hash,
            "JULIA_NUM_THREADS": environment.get("JULIA_NUM_THREADS", "4"),
            "OPENBLAS_NUM_THREADS": "1",
        }
    )

    command_arguments = (
        "--output-dir", str(args.output_dir),
        "--iterations", str(args.iterations),
        "--time-limit", str(args.time_limit),
        "--tolerance", str(args.tolerance),
        "--fine-nu", str(args.fine_nu),
        "--fine-nv", str(args.fine_nv),
        "--seed", str(args.seed),
    )
    with tempfile.TemporaryDirectory(prefix="eh_hf_notebook_") as temporary:
        generated_path = Path(temporary) / "notebook_fig3_generated.jl"
        generated_path.write_text(generated)
        subprocess.run(
            (
                "julia",
                f"--project={HERE}",
                str(generated_path),
                *command_arguments,
            ),
            cwd=REPOSITORY,
            env=environment,
            check=True,
        )

    plot_density(
        args.output_dir / "density.csv",
        args.output_dir / "density.png",
        args.fine_nu,
        args.fine_nv,
    )
    artifact_hashes = {
        name: sha256(args.output_dir / name)
        for name in ("checkpoint.jls", "density.csv", "density.npz", "density.png", "metrics.toml")
    }
    with (args.output_dir / "artifact_hashes.json").open("w") as stream:
        json.dump(artifact_hashes, stream, indent=2)
    print(f"wrote existing-code benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
