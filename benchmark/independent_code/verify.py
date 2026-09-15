#!/usr/bin/env python3
"""Verify the saved Fig. 1 and Fig. 3 reproduction outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from classical_fig1 import (
    geometry,
    interlayer_energy,
    intralayer_ewald_energy,
    transition_point,
)
from hartree_fock_fig3 import (
    Parameters,
    PlaneWaveGrid,
    evaluate,
    localized_orbitals,
    orthonormalize,
    triangular_centers,
)


def classical_energy_with_cutoffs(phase: str, a_over_d: float, cutoff: int, qmax: float) -> float:
    a1, a2, excess_electron = geometry(phase, a_over_d)
    electrons = np.array([[0.0, 0.0], excess_electron])
    holes = np.array([[0.0, 0.0]])
    energy = intralayer_ewald_energy(
        electrons,
        np.array([-1.0, -1.0]),
        a1,
        a2,
        real_cutoff=cutoff,
        reciprocal_cutoff=cutoff,
    )
    energy += intralayer_ewald_energy(
        holes,
        np.array([1.0]),
        a1,
        a2,
        real_cutoff=cutoff,
        reciprocal_cutoff=cutoff,
    )
    energy += interlayer_energy(electrons, holes, a1, a2, reciprocal_radius=qmax)
    return energy


def verify_hf_derivative() -> float:
    """Compare the implemented Fock action with a finite energy derivative."""
    parameters = Parameters(
        nx=9,
        ny=7,
        n_holes=4,
        hole_rs=5.0,
        electron_width=3.0,
        hole_width=2.5,
    )
    grid = PlaneWaveGrid(parameters)
    centers = triangular_centers(parameters)
    electrons = localized_orbitals(grid, centers, parameters.electron_width)
    holes = localized_orbitals(grid, centers, parameters.hole_width)
    state = evaluate(grid, electrons, holes, exchange_batch=2)

    generator = np.random.default_rng(123)
    electron_direction = generator.normal(size=electrons.shape) + 1j * generator.normal(
        size=electrons.shape
    )
    electron_direction -= electrons @ (electrons.conj().T @ electron_direction)
    electron_direction /= np.linalg.norm(electron_direction)
    hole_direction = generator.normal(size=holes.shape) + 1j * generator.normal(size=holes.shape)
    hole_direction -= holes @ (holes.conj().T @ hole_direction)
    hole_direction /= np.linalg.norm(hole_direction)

    # A real variation contributes 2*Re<dC|FC> per species; the restricted
    # electron subspace occurs twice, once for each spin.
    analytic = 4.0 * float(np.vdot(electron_direction, state["electron_fock"]).real)
    analytic += 2.0 * float(np.vdot(hole_direction, state["hole_fock"]).real)
    epsilon = 1.0e-4
    plus = evaluate(
        grid,
        orthonormalize(electrons + epsilon * electron_direction),
        orthonormalize(holes + epsilon * hole_direction),
        exchange_batch=2,
    )["energy"]
    minus = evaluate(
        grid,
        orthonormalize(electrons - epsilon * electron_direction),
        orthonormalize(holes - epsilon * hole_direction),
        exchange_batch=2,
    )["energy"]
    finite_difference = (float(plus) - float(minus)) / (2.0 * epsilon)
    return abs(analytic - finite_difference)


def periodic_peak_count(density: np.ndarray) -> int:
    """Count prominent local maxima with periodic boundaries."""
    is_maximum = np.ones(density.shape, dtype=bool)
    for shift_x in (-1, 0, 1):
        for shift_y in (-1, 0, 1):
            if shift_x == 0 and shift_y == 0:
                continue
            neighbor = np.roll(np.roll(density, shift_x, axis=0), shift_y, axis=1)
            is_maximum &= density >= neighbor
    is_maximum &= density > 0.8 * density.max()
    return int(is_maximum.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    crossover = transition_point()
    if abs(crossover - 5.42) > 0.01:
        raise AssertionError(f"classical crossover {crossover:.6f} does not reproduce 5.42")
    # These are internal high-precision regression anchors obtained from the
    # converged Ewald implementation, not additional tabulated values from the
    # paper.  Paper agreement is independently tested by the 5.42 crossover.
    absolute_references = {
        ("checkerboard", 1.0): -4.235418516834,
        ("honeycomb", 1.0): -4.168366613957,
        ("checkerboard", 10.0): -1.147736276442,
        ("honeycomb", 10.0): -1.148144750733,
    }
    for phase in ("checkerboard", "honeycomb"):
        for ratio in (1.0, 5.42, 10.0):
            standard = classical_energy_with_cutoffs(phase, ratio, 7, 32.0)
            tighter = classical_energy_with_cutoffs(phase, ratio, 9, 40.0)
            if abs(standard - tighter) > 1.0e-10:
                raise AssertionError(f"classical Ewald sum is not converged for {phase} at {ratio}")
            reference = absolute_references.get((phase, ratio))
            if reference is not None and abs(standard - reference) > 1.0e-10:
                raise AssertionError(f"absolute classical energy changed for {phase} at {ratio}")

    derivative_error = verify_hf_derivative()
    if derivative_error > 1.0e-7:
        raise AssertionError(f"HF Fock-action derivative error is {derivative_error:.3e}")

    with (args.results / "fig3_metrics.json").open() as stream:
        metrics = json.load(stream)
    result = metrics["result"]
    parameters = metrics["parameters"]
    if parameters["basis"] != [45, 39]:
        raise AssertionError(f"expected the published 45 x 39 basis, got {parameters['basis']}")
    if not result["converged"]:
        raise AssertionError(
            f"HF gradient did not converge: {result['max_gradient_component']:.3e}"
        )
    energy_error = abs(
        result["energy_per_particle_hartree_h"] - result["paper_energy_per_particle_hartree_h"]
    )
    if energy_error > 1.0e-4:
        raise AssertionError(f"HF energy-per-particle error is {energy_error:.3e}")
    density = np.load(args.results / "fig3_density_data.npz")
    electron_density = density["electron_density"]
    hole_density = density["hole_density"]
    if electron_density.shape != hole_density.shape:
        raise AssertionError("electron and hole density grids differ")
    if electron_density.min() < -1.0e-14 or hole_density.min() < -1.0e-14:
        raise AssertionError("a density contains negative values")
    if not (0.014 < electron_density.max() < 0.018):
        raise AssertionError("electron density maximum is inconsistent with Fig. 3")
    if not (0.014 < hole_density.max() < 0.018):
        raise AssertionError("hole density maximum is inconsistent with Fig. 3")
    if periodic_peak_count(electron_density) != 36:
        raise AssertionError("electron density does not contain the expected 6 x 6 peaks")
    if periodic_peak_count(hole_density) != 36:
        raise AssertionError("hole density does not contain the expected 6 x 6 peaks")
    density_correlation = float(
        np.corrcoef(electron_density.ravel(), hole_density.ravel())[0, 1]
    )
    if density_correlation < 0.95:
        raise AssertionError("electron and hole density peaks are not colocated")
    electron_modulation = float(electron_density.std() / electron_density.mean())
    hole_modulation = float(hole_density.std() / hole_density.mean())
    if hole_modulation < 2.0 * electron_modulation:
        raise AssertionError("hole peaks are not narrower than electron peaks")

    area = math.pi * parameters["n_holes"] * parameters["hole_rs"] ** 2
    density_weight = area / electron_density.size
    integrated_electrons = density_weight * float(electron_density.sum())
    integrated_holes = density_weight * float(hole_density.sum())
    if abs(integrated_electrons - 72.0) > 1.0e-8:
        raise AssertionError("electron density does not integrate to 72")
    if abs(integrated_holes - 36.0) > 1.0e-8:
        raise AssertionError("hole density does not integrate to 36")
    if abs(integrated_electrons - result["integrated_electron_density"]) > 1.0e-12:
        raise AssertionError("electron density NPZ and metrics JSON disagree")
    if abs(integrated_holes - result["integrated_hole_density"]) > 1.0e-12:
        raise AssertionError("hole density NPZ and metrics JSON disagree")

    benchmark_parameters = Parameters(
        nx=parameters["basis"][0],
        ny=parameters["basis"][1],
        n_holes=parameters["n_holes"],
        hole_rs=parameters["hole_rs"],
        layer_distance=parameters["d_over_aB_h"],
        electron_mass=parameters["electron_mass_in_hole_units"],
        hole_mass=parameters["hole_mass_in_hole_units"],
    )
    grid = PlaneWaveGrid(benchmark_parameters)
    recomputed = evaluate(
        grid,
        density["electron_coefficients"],
        density["hole_coefficients"],
        exchange_batch=2,
    )
    if abs(float(recomputed["energy"]) - result["total_energy_hartree_h"]) > 1.0e-10:
        raise AssertionError("stored coefficients and metrics JSON energies disagree")
    if abs(float(recomputed["max_residual"]) - result["max_residual"]) > 1.0e-12:
        raise AssertionError("stored coefficients and metrics JSON residuals disagree")
    if (
        abs(
            float(recomputed["max_gradient_component"])
            - result["max_gradient_component"]
        )
        > 1.0e-12
    ):
        raise AssertionError("stored coefficients and metrics JSON gradients disagree")

    checkpoint = np.load(args.results / "fig3_checkpoint.npz")
    if int(checkpoint["iteration"]) != result["iteration"]:
        raise AssertionError("checkpoint and metrics JSON iterations disagree")
    if not np.array_equal(
        checkpoint["electron_coefficients"], density["electron_coefficients"]
    ):
        raise AssertionError("electron coefficients differ between saved artifacts")
    if not np.array_equal(checkpoint["hole_coefficients"], density["hole_coefficients"]):
        raise AssertionError("hole coefficients differ between saved artifacts")

    energies = np.array([item["energy"] for item in metrics["history"]])
    if np.any(np.diff(energies) > 1.0e-9):
        raise AssertionError("accepted HF steps are not energy-monotone")

    print(f"Fig. 1 crossover: {crossover:.6f} (paper: 5.42)")
    print(
        "Fig. 3 energy per particle: "
        f"{result['energy_per_particle_hartree_h']:+.8f} "
        f"(paper: {result['paper_energy_per_particle_hartree_h']:+.8f})"
    )
    print(
        "Fig. 3 density integrals: "
        f"Ne={integrated_electrons:.12f}, Nh={integrated_holes:.12f}"
    )
    print(f"Fig. 3 max restricted gradient component: {result['max_gradient_component']:.3e}")
    print(
        "Fig. 3 structure: 36 colocated peaks, "
        f"correlation={density_correlation:.6f}, "
        f"modulation ratio={hole_modulation/electron_modulation:.3f}"
    )
    print(f"HF analytic/finite-difference derivative error: {derivative_error:.3e}")
    print("all benchmark checks passed")


if __name__ == "__main__":
    main()
