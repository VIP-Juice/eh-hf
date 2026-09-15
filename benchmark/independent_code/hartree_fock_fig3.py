#!/usr/bin/env python3
"""Restricted plane-wave Hartree-Fock reproduction of Fig. 3.

The implementation follows Eqs. S13-S25 of Dai and Fu's Supplemental
Material.  It uses the published 45 x 39 rectangular plane-wave basis,
removes the q=0 Coulomb component, constrains the two electron-spin Slater
determinants to be identical, and directly minimizes on the occupied
subspaces.  Units are the hole Bohr radius a_B,h and hole Hartree energy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Parameters:
    nx: int = 45
    ny: int = 39
    n_holes: int = 36
    hole_rs: float = 8.0
    layer_distance: float = 4.83
    electron_mass: float = 2.0  # me/mh = 0.8/0.4
    hole_mass: float = 1.0
    electron_width: float = 5.5
    hole_width: float = 4.0

    @property
    def area(self) -> float:
        return math.pi * self.n_holes * self.hole_rs**2

    @property
    def aspect_ratio(self) -> float:
        return math.sqrt(3.0) / 2.0

    @property
    def lx(self) -> float:
        return math.sqrt(self.area / self.aspect_ratio)

    @property
    def ly(self) -> float:
        return self.aspect_ratio * self.lx


class PlaneWaveGrid:
    """Odd rectangular basis with an alias-free product/convolution grid."""

    def __init__(self, parameters: Parameters):
        if parameters.nx % 2 != 1 or parameters.ny % 2 != 1:
            raise ValueError("nx and ny must be odd")
        self.parameters = parameters
        self.mode_x = np.arange(-(parameters.nx // 2), parameters.nx // 2 + 1)
        self.mode_y = np.arange(-(parameters.ny // 2), parameters.ny // 2 + 1)
        mx, my = np.meshgrid(self.mode_x, self.mode_y, indexing="ij")
        self.basis_modes = np.column_stack((mx.ravel(), my.ravel()))
        self.nbasis = len(self.basis_modes)
        self.kx = 2.0 * math.pi * self.basis_modes[:, 0] / parameters.lx
        self.ky = 2.0 * math.pi * self.basis_modes[:, 1] / parameters.ly
        self.k_squared = self.kx**2 + self.ky**2

        # Products of basis functions span -(N-1), ..., N-1.  These odd
        # dimensions represent every momentum transfer without aliasing.
        self.grid_nx = 2 * parameters.nx - 1
        self.grid_ny = 2 * parameters.ny - 1
        self.grid_size = self.grid_nx * self.grid_ny
        self.embed_x = self.basis_modes[:, 0] % self.grid_nx
        self.embed_y = self.basis_modes[:, 1] % self.grid_ny

        q_mode_x = np.rint(np.fft.fftfreq(self.grid_nx) * self.grid_nx)
        q_mode_y = np.rint(np.fft.fftfreq(self.grid_ny) * self.grid_ny)
        qx, qy = np.meshgrid(
            2.0 * math.pi * q_mode_x / parameters.lx,
            2.0 * math.pi * q_mode_y / parameters.ly,
            indexing="ij",
        )
        q = np.sqrt(qx**2 + qy**2)
        self.intralayer_kernel = np.zeros_like(q)
        nonzero = q > 0.0
        self.intralayer_kernel[nonzero] = 2.0 * math.pi / q[nonzero]
        self.interlayer_kernel = self.intralayer_kernel * np.exp(-q * parameters.layer_distance)
        self.real_space_weight = parameters.area / self.grid_size

    def coefficients_to_real(self, coefficients: np.ndarray) -> np.ndarray:
        embedded = np.zeros(
            (self.grid_nx, self.grid_ny, coefficients.shape[1]), dtype=np.complex128
        )
        embedded[self.embed_x, self.embed_y, :] = coefficients
        embedded *= self.grid_size / math.sqrt(self.parameters.area)
        return np.fft.ifft2(embedded, axes=(0, 1))

    def real_to_coefficients(self, values: np.ndarray) -> np.ndarray:
        transformed = np.fft.fft2(values, axes=(0, 1))
        transformed *= math.sqrt(self.parameters.area) / self.grid_size
        return transformed[self.embed_x, self.embed_y, :]

    @staticmethod
    def convolve(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(kernel * np.fft.fft2(field)).real

    def exchange_action(
        self,
        orbitals_real: np.ndarray,
        kernel: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Return K C using K(r,r') = v(r-r') D(r,r')."""
        noccupied = orbitals_real.shape[2]
        output = np.empty_like(orbitals_real)
        conjugate_orbitals = np.conj(orbitals_real)
        kernel_broadcast = kernel[:, :, None, None]
        for first in range(0, noccupied, batch_size):
            last = min(first + batch_size, noccupied)
            pair_density = (
                conjugate_orbitals[:, :, :, None]
                * orbitals_real[:, :, None, first:last]
            )
            pair_potential = np.fft.ifft2(
                kernel_broadcast * np.fft.fft2(pair_density, axes=(0, 1)),
                axes=(0, 1),
            )
            output[:, :, first:last] = np.sum(
                orbitals_real[:, :, :, None] * pair_potential, axis=2
            )
        return self.real_to_coefficients(output)


def triangular_centers(parameters: Parameters) -> np.ndarray:
    side = round(math.sqrt(parameters.n_holes))
    if side * side != parameters.n_holes:
        raise ValueError("This benchmark expects a square number of holes")
    lattice_constant = parameters.lx / side
    centers = []
    for row in range(side):
        for column in range(side):
            x = lattice_constant * (column + 0.5 * row + 0.17)
            y = parameters.ly * (row + 0.17) / side
            centers.append((x % parameters.lx, y % parameters.ly))
    return np.asarray(centers)


def localized_orbitals(grid: PlaneWaveGrid, centers: np.ndarray, width: float) -> np.ndarray:
    phases = np.exp(
        -1j
        * (
            grid.kx[:, None] * centers[None, :, 0]
            + grid.ky[:, None] * centers[None, :, 1]
        )
    )
    coefficients = np.exp(-0.5 * width**2 * grid.k_squared)[:, None] * phases
    return orthonormalize(coefficients)


def orthonormalize(coefficients: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(coefficients, mode="reduced")
    phases = np.ones(r.shape[0], dtype=np.complex128)
    diagonal = np.diag(r)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases[None, :]


def tangent_residual(coefficients: np.ndarray, fock_action: np.ndarray) -> np.ndarray:
    return fock_action - coefficients @ (coefficients.conj().T @ fock_action)


def precondition(
    residual: np.ndarray,
    coefficients: np.ndarray,
    kinetic: np.ndarray,
) -> np.ndarray:
    direction = residual / (1.0 + 3.0 * kinetic[:, None])
    return direction - coefficients @ (coefficients.conj().T @ direction)


def evaluate(
    grid: PlaneWaveGrid,
    electron_coefficients: np.ndarray,
    hole_coefficients: np.ndarray,
    *,
    exchange_batch: int,
) -> dict[str, object]:
    p = grid.parameters
    electron_real = grid.coefficients_to_real(electron_coefficients)
    hole_real = grid.coefficients_to_real(hole_coefficients)
    electron_density = np.sum(np.abs(electron_real) ** 2, axis=2)
    hole_density = np.sum(np.abs(hole_real) ** 2, axis=2)
    total_electron_density = 2.0 * electron_density

    electron_hartree = grid.convolve(total_electron_density, grid.intralayer_kernel)
    electron_hartree -= grid.convolve(hole_density, grid.interlayer_kernel)
    hole_hartree = grid.convolve(hole_density, grid.intralayer_kernel)
    hole_hartree -= grid.convolve(total_electron_density, grid.interlayer_kernel)

    electron_exchange = grid.exchange_action(
        electron_real, grid.intralayer_kernel, batch_size=exchange_batch
    )
    hole_exchange = grid.exchange_action(
        hole_real, grid.intralayer_kernel, batch_size=exchange_batch
    )
    electron_local = grid.real_to_coefficients(electron_hartree[:, :, None] * electron_real)
    hole_local = grid.real_to_coefficients(hole_hartree[:, :, None] * hole_real)

    electron_kinetic = grid.k_squared / (2.0 * p.electron_mass)
    hole_kinetic = grid.k_squared / (2.0 * p.hole_mass)
    electron_fock = electron_kinetic[:, None] * electron_coefficients
    electron_fock += electron_local - electron_exchange
    hole_fock = hole_kinetic[:, None] * hole_coefficients
    hole_fock += hole_local - hole_exchange

    kinetic_energy_e = float(
        np.sum(electron_kinetic[:, None] * np.abs(electron_coefficients) ** 2).real
    )
    kinetic_energy_h = float(
        np.sum(hole_kinetic[:, None] * np.abs(hole_coefficients) ** 2).real
    )
    hartree_energy = 0.5 * grid.real_space_weight * float(
        np.sum(total_electron_density * electron_hartree + hole_density * hole_hartree)
    )
    exchange_energy_e = -0.5 * float(
        np.vdot(electron_coefficients, electron_exchange).real
    )
    exchange_energy_h = -0.5 * float(np.vdot(hole_coefficients, hole_exchange).real)
    total_energy = (
        2.0 * kinetic_energy_e
        + kinetic_energy_h
        + hartree_energy
        + 2.0 * exchange_energy_e
        + exchange_energy_h
    )

    electron_residual = tangent_residual(electron_coefficients, electron_fock)
    hole_residual = tangent_residual(hole_coefficients, hole_fock)
    max_residual = max(
        float(np.max(np.abs(electron_residual))),
        float(np.max(np.abs(hole_residual))),
    )
    # For a real-coordinate variation, dE/dC is twice the Fock action.  The
    # shared restricted-electron variable contributes to both spin species.
    max_gradient_component = max(
        4.0 * float(np.max(np.abs(electron_residual.real))),
        4.0 * float(np.max(np.abs(electron_residual.imag))),
        2.0 * float(np.max(np.abs(hole_residual.real))),
        2.0 * float(np.max(np.abs(hole_residual.imag))),
    )
    return {
        "energy": total_energy,
        "components": {
            "kinetic_electrons": 2.0 * kinetic_energy_e,
            "kinetic_holes": kinetic_energy_h,
            "hartree": hartree_energy,
            "exchange_electrons": 2.0 * exchange_energy_e,
            "exchange_holes": exchange_energy_h,
        },
        "electron_fock": electron_fock,
        "hole_fock": hole_fock,
        "electron_residual": electron_residual,
        "hole_residual": hole_residual,
        "max_residual": max_residual,
        "max_gradient_component": max_gradient_component,
        "electron_density": total_electron_density,
        "hole_density": hole_density,
    }


def save_checkpoint(
    path: Path,
    electron_coefficients: np.ndarray,
    hole_coefficients: np.ndarray,
    iteration: int,
    step_size: float,
    history: list[dict[str, float]],
) -> None:
    np.savez_compressed(
        path,
        electron_coefficients=electron_coefficients,
        hole_coefficients=hole_coefficients,
        iteration=np.array(iteration),
        step_size=np.array(step_size),
        history_iteration=np.array([item["iteration"] for item in history]),
        history_energy=np.array([item["energy"] for item in history]),
        history_residual=np.array([item["max_residual"] for item in history]),
        history_gradient=np.array([item["max_gradient_component"] for item in history]),
    )


def minimize(
    grid: PlaneWaveGrid,
    electron_coefficients: np.ndarray,
    hole_coefficients: np.ndarray,
    *,
    iterations: int,
    tolerance: float,
    initial_step: float,
    exchange_batch: int,
    checkpoint: Path,
    checkpoint_every: int,
    log_every: int,
    start_iteration: int = 0,
    history: list[dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict[str, float]], int, float]:
    history = [] if history is None else history
    step_size = initial_step
    state = evaluate(
        grid, electron_coefficients, hole_coefficients, exchange_batch=exchange_batch
    )
    accepted_iteration = start_iteration
    for iteration in range(start_iteration, start_iteration + iterations + 1):
        accepted_iteration = iteration
        energy = float(state["energy"])
        residual = float(state["max_residual"])
        gradient = float(state["max_gradient_component"])
        history.append(
            {
                "iteration": float(iteration),
                "energy": energy,
                "max_residual": residual,
                "max_gradient_component": gradient,
            }
        )
        if iteration == start_iteration or iteration % log_every == 0:
            per_particle = energy / (3 * grid.parameters.n_holes)
            print(
                f"iteration {iteration:4d}: E/N = {per_particle:+.8f}, "
                f"max|gradient| = {gradient:.3e}, step = {step_size:.3e}",
                flush=True,
            )
        if gradient < tolerance or iteration == start_iteration + iterations:
            break

        electron_kinetic = grid.k_squared / (2.0 * grid.parameters.electron_mass)
        hole_kinetic = grid.k_squared / (2.0 * grid.parameters.hole_mass)
        electron_direction = precondition(
            state["electron_residual"], electron_coefficients, electron_kinetic
        )
        hole_direction = precondition(state["hole_residual"], hole_coefficients, hole_kinetic)

        accepted = False
        trial_step = step_size
        for _ in range(8):
            trial_electron = orthonormalize(
                electron_coefficients - trial_step * electron_direction
            )
            trial_hole = orthonormalize(hole_coefficients - trial_step * hole_direction)
            trial_state = evaluate(
                grid, trial_electron, trial_hole, exchange_batch=exchange_batch
            )
            if float(trial_state["energy"]) <= energy + 1.0e-11:
                electron_coefficients = trial_electron
                hole_coefficients = trial_hole
                state = trial_state
                step_size = min(1.05 * trial_step, 1.0)
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            print("line search could not lower the energy; stopping", flush=True)
            break
        if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint,
                electron_coefficients,
                hole_coefficients,
                iteration + 1,
                step_size,
                history,
            )
    return electron_coefficients, hole_coefficients, state, history, accepted_iteration, step_size


def density_on_fine_grid(
    coefficients: np.ndarray,
    parameters: Parameters,
    *,
    fine_nx: int = 360,
    fine_ny: int = 312,
) -> np.ndarray:
    mode_x = np.arange(-(parameters.nx // 2), parameters.nx // 2 + 1)
    mode_y = np.arange(-(parameters.ny // 2), parameters.ny // 2 + 1)
    mx, my = np.meshgrid(mode_x, mode_y, indexing="ij")
    modes = np.column_stack((mx.ravel(), my.ravel()))
    embedded = np.zeros((fine_nx, fine_ny, coefficients.shape[1]), dtype=np.complex128)
    embedded[modes[:, 0] % fine_nx, modes[:, 1] % fine_ny, :] = coefficients
    embedded *= fine_nx * fine_ny / math.sqrt(parameters.area)
    orbitals = np.fft.ifft2(embedded, axes=(0, 1))
    return np.sum(np.abs(orbitals) ** 2, axis=2)


def write_outputs(
    output_dir: Path,
    grid: PlaneWaveGrid,
    electron_coefficients: np.ndarray,
    hole_coefficients: np.ndarray,
    state: dict[str, object],
    history: list[dict[str, float]],
    iteration: int,
    elapsed_seconds: float,
    tolerance: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = grid.parameters
    electron_density = 2.0 * density_on_fine_grid(electron_coefficients, p)
    hole_density = density_on_fine_grid(hole_coefficients, p)
    x = np.linspace(0.0, p.lx, electron_density.shape[0], endpoint=False)
    y = np.linspace(0.0, p.ly, electron_density.shape[1], endpoint=False)
    np.savez_compressed(
        output_dir / "fig3_density_data.npz",
        x=x,
        y=y,
        electron_density=electron_density,
        hole_density=hole_density,
        electron_coefficients=electron_coefficients,
        hole_coefficients=hole_coefficients,
    )

    vmax = max(float(electron_density.max()), float(hole_density.max()))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    images = []
    for axis, density, title in zip(
        axes, (electron_density, hole_density), ("Electron density", "Hole density")
    ):
        image = axis.imshow(
            density.T,
            origin="lower",
            extent=(0.0, p.lx, 0.0, p.ly),
            cmap="hot",
            vmin=0.0,
            vmax=vmax,
            interpolation="bicubic",
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel(r"$x/a_{B,h}$")
        axis.set_ylabel(r"$y/a_{B,h}$")
        axis.set_aspect("equal")
    colorbar = figure.colorbar(images[0], ax=axes, shrink=0.88)
    colorbar.set_label(r"density $[1/a_{B,h}^2]$")
    figure.savefig(output_dir / "fig3_density.png", dpi=220)
    plt.close(figure)

    weight = p.area / electron_density.size
    final_energy = float(state["energy"])
    metrics = {
        "parameters": {
            "electron_mass_m0": 0.8,
            "hole_mass_m0": 0.4,
            "electron_mass_in_hole_units": p.electron_mass,
            "hole_mass_in_hole_units": p.hole_mass,
            "d_over_aB_h": p.layer_distance,
            "hole_rs": p.hole_rs,
            "n_holes": p.n_holes,
            "n_electrons_up": p.n_holes,
            "n_electrons_down": p.n_holes,
            "basis": [p.nx, p.ny],
            "box": [p.lx, p.ly],
        },
        "result": {
            "iteration": iteration,
            "converged": float(state["max_gradient_component"]) < tolerance,
            "max_residual": float(state["max_residual"]),
            "max_gradient_component": float(state["max_gradient_component"]),
            "total_energy_hartree_h": final_energy,
            "energy_per_particle_hartree_h": final_energy / (3 * p.n_holes),
            "paper_energy_per_particle_hartree_h": -0.07652,
            "energy_components_hartree_h": state["components"],
            "integrated_electron_density": weight * float(electron_density.sum()),
            "integrated_hole_density": weight * float(hole_density.sum()),
            "max_electron_density": float(electron_density.max()),
            "max_hole_density": float(hole_density.max()),
            "elapsed_seconds": elapsed_seconds,
        },
        "history": history,
    }
    with (output_dir / "fig3_metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, int, float, list[dict[str, float]]]:
    data = np.load(path)
    residuals = data["history_residual"]
    gradients = data["history_gradient"] if "history_gradient" in data else 4.0 * residuals
    history = [
        {
            "iteration": float(i),
            "energy": float(e),
            "max_residual": float(r),
            "max_gradient_component": float(g),
        }
        for i, e, r, g in zip(
            data["history_iteration"], data["history_energy"], residuals, gradients
        )
    ]
    return (
        data["electron_coefficients"],
        data["hole_coefficients"],
        int(data["iteration"]),
        float(data["step_size"]),
        history,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--step", type=float, default=0.35)
    parser.add_argument("--exchange-batch", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="use a 25 x 21 basis for a fast smoke test")
    args = parser.parse_args()

    parameters = Parameters(nx=25, ny=21) if args.quick else Parameters()
    grid = PlaneWaveGrid(parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "fig3_checkpoint.npz"
    centers = triangular_centers(parameters)
    start_iteration = 0
    history: list[dict[str, float]] = []
    step_size = args.step
    if args.resume and checkpoint.exists():
        electron_coefficients, hole_coefficients, start_iteration, step_size, history = load_checkpoint(checkpoint)
        print(f"resuming checkpoint at iteration {start_iteration}", flush=True)
    else:
        electron_coefficients = localized_orbitals(grid, centers, parameters.electron_width)
        hole_coefficients = localized_orbitals(grid, centers, parameters.hole_width)

    start_time = time.monotonic()
    electron_coefficients, hole_coefficients, state, history, iteration, step_size = minimize(
        grid,
        electron_coefficients,
        hole_coefficients,
        iterations=args.iterations,
        tolerance=args.tolerance,
        initial_step=step_size,
        exchange_batch=args.exchange_batch,
        checkpoint=checkpoint,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        start_iteration=start_iteration,
        history=history,
    )
    elapsed_seconds = time.monotonic() - start_time
    save_checkpoint(
        checkpoint,
        electron_coefficients,
        hole_coefficients,
        iteration,
        step_size,
        history,
    )
    write_outputs(
        args.output_dir,
        grid,
        electron_coefficients,
        hole_coefficients,
        state,
        history,
        iteration,
        elapsed_seconds,
        args.tolerance,
    )
    print(
        f"wrote Fig. 3 outputs; E/N = {float(state['energy'])/(3*parameters.n_holes):+.8f}, "
        f"paper = -0.07652000, elapsed = {elapsed_seconds:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
