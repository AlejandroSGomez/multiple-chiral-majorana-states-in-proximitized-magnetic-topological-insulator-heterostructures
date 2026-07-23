#!/usr/bin/env python3
"""Low-energy bands of the superconducting B=0 quantum-well slab.

The calculation uses the standing-wave momenta of the B=0 slab, so the
Hamiltonian separates into one 8x8 BdG block for each confined z mode.
All quantities are expressed directly in the dimensionless units used in the
manuscript: energies in Delta and lengths/momenta in c.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "generated"
CACHE_DIR = ROOT / "data" / "processed"
FIG_BASENAME = "reference_slab_bands_Lz40"
CACHE_PATH = CACHE_DIR / "reference_slab_bands_Lz40.npz"


# Parameters used for the B=0 multi-Majorana calculations.  Delta=c=1.
PARAMS = {
    "Lz_c": 40,
    "ky_c": 0.0,
    "a_c": 2.0,
    "mu_tilde_Delta": 1.0,
    "C2_Delta_c2": 0.0,
    "M0_Delta": -2.8,
    "M1_Delta_c2": 25.0,
    "M2_Delta_c2": 141.475,
    "A_Delta_c": 20.5,
    "G_Delta": -4.6,
    "T_Delta": 4.5,
    "B_Delta_c": 0.0,
    "Delta_Delta": 1.0,
}


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def gamma_matrices() -> dict[str, np.ndarray]:
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return {
        "z00": kron3(sz, s0, s0),
        "z0z": kron3(sz, s0, sz),
        "00x": kron3(s0, s0, sx),
        "zzy": kron3(sz, sz, sy),
        "yyz": kron3(sy, sy, sz),
        "zyy": kron3(sz, sy, sy),
        "zzz": kron3(sz, sz, sz),
    }


def calculate_bands(
    params: dict[str, float],
    *,
    kx_c: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the eight BdG branches for every confined z mode."""
    g = gamma_matrices()
    n_z = int(params["Lz_c"])
    a_c = float(params["a_c"])
    ky_c = float(params["ky_c"])
    kz_c = np.arange(1, n_z + 1, dtype=float) * np.pi / (n_z + 1)

    energies = np.empty((kx_c.size, n_z, 8), dtype=float)
    for ik, kxc in enumerate(kx_c):
        ck_xy = 2.0 - np.cos(kxc * a_c) - np.cos(ky_c * a_c)
        epsilon = (
            params["mu_tilde_Delta"]
            + 2.0 * params["C2_Delta_c2"] / a_c**2 * ck_xy
        )
        in_plane = (
            params["A_Delta_c"] / a_c
            * (
                np.sin(kxc * a_c) * g["00x"]
                + np.sin(ky_c * a_c) * g["zzy"]
            )
        )
        for im, kzc in enumerate(kz_c):
            mass = (
                params["M0_Delta"]
                + 2.0 * params["M2_Delta_c2"] / a_c**2 * ck_xy
                + 2.0 * params["M1_Delta_c2"] * (1.0 - np.cos(kzc))
            )
            hamiltonian = (
                epsilon * g["z00"]
                + mass * g["z0z"]
                + in_plane
                + params["Delta_Delta"] * g["yyz"]
                + params["T_Delta"] * g["zyy"]
                + params["G_Delta"] * g["zzz"]
            )
            energies[ik, im] = np.linalg.eigvalsh(hamiltonian)

    return {"kx_c": np.asarray(kx_c), "kz_c": kz_c, "energies": energies}


def load_or_calculate(*, force: bool = False) -> dict[str, np.ndarray]:
    if CACHE_PATH.exists() and not force:
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    kx_c = np.linspace(-0.12, 0.12, 401)
    data = calculate_bands(PARAMS, kx_c=kx_c)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **data, **{key: np.array(value) for key, value in PARAMS.items()})
    return data


def paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.dpi": 600,
    }


def plot_panel(
    ax: mpl.axes.Axes,
    data: dict[str, np.ndarray],
    *,
    energy_window: tuple[float, float] = (-2.0, 2.0),
    show_xlabel: bool = True,
) -> dict[str, object]:
    """Plot the slab bands on an explicit axes and return the artists."""
    x = np.asarray(data["kx_c"])
    energies = np.asarray(data["energies"])
    lines = []
    for mode in range(energies.shape[1]):
        for branch in range(energies.shape[2]):
            y = energies[:, mode, branch]
            if np.any((y >= energy_window[0]) & (y <= energy_window[1])):
                line, = ax.plot(x, y, color="#262626", lw=0.85, alpha=0.82)
                lines.append(line)

    ax.axhline(0.0, color="0.48", lw=0.65, zorder=0)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(*energy_window)
    ax.set_xlabel(r"$k_x c$" if show_xlabel else "")
    ax.set_ylabel(r"$E/\Delta$")
    ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)
    return {"ax": ax, "lines": lines}


def make_figure(data: dict[str, np.ndarray]) -> mpl.figure.Figure:
    """
    FIGURE_HANDOFF
    kind: panel
    entrypoint: plot_panel(ax, data, *, energy_window=(-2, 2), show_xlabel=True)
    target: single-column
    size_in: 3.6 x 3.6
    dependencies: matplotlib, numpy
    inputs: kx_c and energies[kx, confined-mode, BdG-branch]
    returns: axes and line handles
    labels: x = k_x c; y = E/Delta; panel letter owned by figure
    colorbar: none
    exports: figures/generated/reference_slab_bands_Lz40.{pdf,png,svg}
    """
    with mpl.rc_context(paper_style()):
        fig, ax = plt.subplots(figsize=(3.6, 3.6), constrained_layout=True)
        plot_panel(ax, data)
        return fig


def save_figure(fig: mpl.figure.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "B=0 superconducting slab bands at ky=0 and Lz/c=40",
        "Subject": "BdG bands in units of Delta versus kx*c",
    }
    fig.savefig(FIG_DIR / f"{FIG_BASENAME}.pdf", metadata=metadata)
    fig.savefig(
        FIG_DIR / f"{FIG_BASENAME}.png",
        dpi=600,
        metadata={"Description": metadata["Subject"]},
    )
    fig.savefig(
        FIG_DIR / f"{FIG_BASENAME}.svg",
        metadata={"Title": metadata["Title"], "Description": metadata["Subject"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="ignore and replace the cached bands")
    args = parser.parse_args()
    data = load_or_calculate(force=args.recompute)
    fig = make_figure(data)
    save_figure(fig)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / (FIG_BASENAME + '.pdf')}")
    print(f"Wrote {FIG_DIR / (FIG_BASENAME + '.png')}")
    print(f"Wrote {FIG_DIR / (FIG_BASENAME + '.svg')}")


if __name__ == "__main__":
    main()
