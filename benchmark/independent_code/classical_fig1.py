#!/usr/bin/env python3
"""Reproduce the classical checkerboard/honeycomb comparison in Fig. 1.

Lengths are measured in units of the interlayer spacing d, charges in units of
e, and energies in e^2/(4*pi*epsilon*d).  Each primitive cell contains one
hole, one electron directly above it (a dipole), and one excess electron.
The q=0 Fourier component is removed, exactly as for the neutralizing
background used in the paper and its Supplemental Material.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SQRT_PI = math.sqrt(math.pi)


def reciprocal_lattice(a1: np.ndarray, a2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return cell area and reciprocal vectors satisfying ai dot bj = 2*pi delta_ij."""
    cell = np.column_stack((a1, a2))
    area = abs(float(np.linalg.det(cell)))
    reciprocal = 2.0 * math.pi * np.linalg.inv(cell).T
    return area, reciprocal[:, 0], reciprocal[:, 1]


def geometry(phase: str, a_over_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return primitive vectors and the excess-electron basis position."""
    a = float(a_over_d)  # d = 1; one hole per area pi*a^2.
    if phase == "checkerboard":
        lattice_constant = math.sqrt(math.pi) * a
        a1 = np.array([lattice_constant, 0.0])
        a2 = np.array([0.0, lattice_constant])
        excess_electron = 0.5 * (a1 + a2)
    elif phase == "honeycomb":
        lattice_constant = a * math.sqrt(2.0 * math.pi / math.sqrt(3.0))
        a1 = np.array([lattice_constant, 0.0])
        a2 = np.array([0.5 * lattice_constant, 0.5 * math.sqrt(3.0) * lattice_constant])
        excess_electron = (a1 + a2) / 3.0
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return a1, a2, excess_electron


def intralayer_ewald_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    real_cutoff: int = 7,
    reciprocal_cutoff: int = 7,
) -> float:
    """2D-periodic 1/r energy with a uniform neutralizing background.

    The short-range kernel is erfc(r/r0)/r with r0 equal to one quarter of
    the shortest cell vector, matching the Supplemental Material.  The four
    terms below are respectively real space, reciprocal space, point self,
    and short-range background contributions.
    """
    area, b1, b2 = reciprocal_lattice(a1, a2)
    r0 = 0.25 * min(np.linalg.norm(a1), np.linalg.norm(a2))
    alpha = 1.0 / r0

    ns = np.arange(-real_cutoff, real_cutoff + 1)
    n1, n2 = np.meshgrid(ns, ns, indexing="ij")
    translations = n1[..., None] * a1 + n2[..., None] * a2
    real_energy = 0.0
    for i, ri in enumerate(positions):
        for j, rj in enumerate(positions):
            vectors = translations + ri - rj
            radii = np.linalg.norm(vectors, axis=-1)
            if i == j:
                radii[real_cutoff, real_cutoff] = np.inf
            terms = np.fromiter(
                (math.erfc(alpha * radius) / radius for radius in radii.ravel()),
                dtype=float,
                count=radii.size,
            )
            real_energy += 0.5 * charges[i] * charges[j] * float(terms.sum())

    ns = np.arange(-reciprocal_cutoff, reciprocal_cutoff + 1)
    n1, n2 = np.meshgrid(ns, ns, indexing="ij")
    reciprocal_vectors = (n1[..., None] * b1 + n2[..., None] * b2).reshape(-1, 2)
    reciprocal_vectors = reciprocal_vectors[np.any(reciprocal_vectors != 0.0, axis=1)]
    magnitudes = np.linalg.norm(reciprocal_vectors, axis=1)
    phases = reciprocal_vectors @ positions.T
    structure = np.exp(1j * phases) @ charges
    reciprocal_energy = np.sum(
        np.fromiter(
            (math.erfc(g / (2.0 * alpha)) for g in magnitudes),
            dtype=float,
            count=magnitudes.size,
        )
        * np.abs(structure) ** 2
        / magnitudes
    )
    reciprocal_energy *= math.pi / area

    self_energy = -alpha / SQRT_PI * float(charges @ charges)
    background_energy = -math.pi * float(charges.sum()) ** 2 / (area * alpha * SQRT_PI)
    return float(real_energy + reciprocal_energy + self_energy + background_energy)


def interlayer_energy(
    electron_positions: np.ndarray,
    hole_positions: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    *,
    d: float = 1.0,
    reciprocal_radius: float = 32.0,
) -> float:
    """Attractive e-h energy from 2*pi*exp(-q*d)/q, with q=0 removed."""
    area, b1, b2 = reciprocal_lattice(a1, a2)
    # This conservative index range encloses the requested circular q cutoff.
    smallest_b = min(np.linalg.norm(b1), np.linalg.norm(b2))
    nmax = int(math.ceil(reciprocal_radius / smallest_b)) + 3
    ns = np.arange(-nmax, nmax + 1)
    n1, n2 = np.meshgrid(ns, ns, indexing="ij")
    vectors = (n1[..., None] * b1 + n2[..., None] * b2).reshape(-1, 2)
    magnitudes = np.linalg.norm(vectors, axis=1)
    keep = (magnitudes > 0.0) & (magnitudes <= reciprocal_radius)
    vectors = vectors[keep]
    magnitudes = magnitudes[keep]

    # Unit charges are included here: electrons are -1 and holes are +1.
    electron_structure = -np.exp(1j * (vectors @ electron_positions.T)).sum(axis=1)
    hole_structure = np.exp(1j * (vectors @ hole_positions.T)).sum(axis=1)
    summand = (
        np.exp(-magnitudes * d)
        * np.real(electron_structure * np.conj(hole_structure))
        / magnitudes
    )
    return float(2.0 * math.pi / area * summand.sum())


def classical_energy(phase: str, a_over_d: float) -> float:
    """Electrostatic energy per 2e-1h primitive cell in units of 1/d."""
    a1, a2, excess_electron = geometry(phase, a_over_d)
    origin = np.zeros(2)
    electron_positions = np.array([origin, excess_electron])
    hole_positions = np.array([origin])
    energy = intralayer_ewald_energy(
        electron_positions, np.array([-1.0, -1.0]), a1, a2
    )
    energy += intralayer_ewald_energy(hole_positions, np.array([1.0]), a1, a2)
    energy += interlayer_energy(electron_positions, hole_positions, a1, a2)
    return energy


def transition_point(left: float = 4.5, right: float = 6.5) -> float:
    """Locate E_honeycomb - E_checkerboard = 0 by bisection."""
    def difference(x: float) -> float:
        return classical_energy("honeycomb", x) - classical_energy("checkerboard", x)

    f_left = difference(left)
    f_right = difference(right)
    if f_left * f_right >= 0.0:
        raise RuntimeError("Transition is not bracketed")
    for _ in range(45):
        midpoint = 0.5 * (left + right)
        f_midpoint = difference(midpoint)
        if f_left * f_midpoint <= 0.0:
            right = midpoint
            f_right = f_midpoint
        else:
            left = midpoint
            f_left = f_midpoint
    return 0.5 * (left + right)


def write_results(output_dir: Path, samples: int) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = np.linspace(1.0, 10.0, samples)
    honeycomb = np.array([classical_energy("honeycomb", x) for x in ratios])
    checkerboard = np.array([classical_energy("checkerboard", x) for x in ratios])
    difference = honeycomb - checkerboard
    crossover = transition_point()

    with (output_dir / "fig1_classical.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["a_over_d", "honeycomb_energy_1_over_d", "checkerboard_energy_1_over_d", "honeycomb_minus_checkerboard_1_over_d"])
        writer.writerows(zip(ratios, honeycomb, checkerboard, difference))

    figure, axes = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(ratios, honeycomb, label="Honeycomb", linewidth=2.0)
    axes[0].plot(ratios, checkerboard, label="Checkerboard", linewidth=1.7, linestyle="--")
    axes[0].set_ylabel(r"$E\;[1/d]$")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="0.25", linewidth=0.9, linestyle="--")
    axes[1].plot(ratios, difference, color="tab:green", linewidth=2.0)
    axes[1].axvline(crossover, color="tab:red", linewidth=1.1, linestyle=":")
    axes[1].annotate(
        rf"$a/d={crossover:.3f}$",
        xy=(crossover, 0.0),
        xytext=(crossover + 0.45, 0.018),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        color="tab:red",
    )
    axes[1].set_xlabel(r"$a/d$")
    axes[1].set_ylabel(r"$E_{\rm honeycomb}-E_{\rm checkerboard}\;[1/d]$")
    figure.savefig(output_dir / "fig1_classical.png", dpi=220)
    plt.close(figure)
    return crossover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--samples", type=int, default=121)
    args = parser.parse_args()
    crossover = write_results(args.output_dir, args.samples)
    print(f"checkerboard/honeycomb crossover a/d = {crossover:.6f} (paper: 5.42)")


if __name__ == "__main__":
    main()
