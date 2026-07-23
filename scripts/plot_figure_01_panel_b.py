#!/usr/bin/env python3
"""Bands of the B=0 superconducting ribbon, finite in y and z."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compute"))

from ribbon_model import PARAMS as BASE_PARAMS
from ribbon_model import gamma_matrices, paper_style


FIG_DIR = ROOT / "figures" / "generated"
CACHE_DIR = ROOT / "data" / "processed"
FIG_BASENAME = "Fig1_panel_b"
CACHE_PATH = CACHE_DIR / "fig01_bands.npz"

PARAMS = dict(BASE_PARAMS)
PARAMS.update({"Lz_c": 20, "Ly_c": 150})

N_K = 401
N_EIGS_PER_MODE = 20
# Modes 1--4 cover the full |E/Delta| <= 2 window; 5 and 6 are retained as
# an explicit numerical guard and are discarded automatically by the plot.
Z_MODES = np.arange(1, 7, dtype=int)
TEXT_SCALE = 3.0


def large_text_style() -> dict[str, object]:
    """Paper style with all typography and tick geometry scaled by 3."""
    style = paper_style()
    style.update({
        "font.size": 10 * TEXT_SCALE,
        "axes.labelsize": 12 * TEXT_SCALE,
        "axes.titlesize": 10 * TEXT_SCALE,
        "xtick.labelsize": 9 * TEXT_SCALE,
        "ytick.labelsize": 9 * TEXT_SCALE,
        "legend.fontsize": 8 * TEXT_SCALE,
        "axes.linewidth": 0.8 * TEXT_SCALE,
        "xtick.major.size": 3.5 * TEXT_SCALE,
        "ytick.major.size": 3.5 * TEXT_SCALE,
        "xtick.major.width": 0.8 * TEXT_SCALE,
        "ytick.major.width": 0.8 * TEXT_SCALE,
        "xtick.minor.size": 2.0 * TEXT_SCALE,
        "ytick.minor.size": 2.0 * TEXT_SCALE,
        "xtick.minor.width": 0.6 * TEXT_SCALE,
        "ytick.minor.width": 0.6 * TEXT_SCALE,
        "xtick.major.pad": 3.5 * TEXT_SCALE,
        "ytick.major.pad": 3.5 * TEXT_SCALE,
    })
    return style


def y_ribbon_hamiltonian(
    kx_c: float,
    z_mode: int,
    params: dict[str, float],
) -> sp.csc_matrix:
    """Return one confined-z block of the finite-y BdG Hamiltonian."""
    g = gamma_matrices()
    a_c = float(params["a_c"])
    n_y = int(round(params["Ly_c"] / a_c))
    n_z = int(params["Lz_c"])
    kz_c = float(z_mode) * np.pi / (n_z + 1)

    ck_x = 2.0 - np.cos(kx_c * a_c)
    epsilon = (
        params["mu_tilde_Delta"]
        + 2.0 * params["C2_Delta_c2"] / a_c**2 * ck_x
    )
    mass = (
        params["M0_Delta"]
        + 2.0 * params["M2_Delta_c2"] / a_c**2 * ck_x
        + 2.0 * params["M1_Delta_c2"] * (1.0 - np.cos(kz_c))
    )
    onsite = (
        epsilon * g["z00"]
        + mass * g["z0z"]
        + params["A_Delta_c"] / a_c * np.sin(kx_c * a_c) * g["00x"]
        + params["Delta_Delta"] * g["yyz"]
        + params["T_Delta"] * g["zyy"]
        + params["G_Delta"] * g["zzz"]
    )
    hopping_y = (
        -params["C2_Delta_c2"] / a_c**2 * g["z00"]
        - params["M2_Delta_c2"] / a_c**2 * g["z0z"]
        - 1j * params["A_Delta_c"] / (2.0 * a_c) * g["zzy"]
    )

    eye_y = sp.eye(n_y, format="csc")
    upper = sp.diags(np.ones(n_y - 1), 1, format="csc")
    lower = sp.diags(np.ones(n_y - 1), -1, format="csc")
    return (
        sp.kron(eye_y, sp.csc_matrix(onsite), format="csc")
        + sp.kron(upper, sp.csc_matrix(hopping_y), format="csc")
        + sp.kron(lower, sp.csc_matrix(hopping_y.conj().T), format="csc")
    )


def calculate_bands(
    params: dict[str, float],
    *,
    kx_c: np.ndarray,
    z_modes: np.ndarray = Z_MODES,
    n_eigs: int = N_EIGS_PER_MODE,
) -> dict[str, np.ndarray]:
    """Compute low-energy bands independently in each B=0 z subband."""
    energies = np.empty((kx_c.size, z_modes.size, n_eigs), dtype=float)
    for ik, kxc in enumerate(kx_c):
        for im, mode in enumerate(z_modes):
            hamiltonian = y_ribbon_hamiltonian(float(kxc), int(mode), params)
            eigvals = sla.eigsh(
                hamiltonian,
                k=int(n_eigs),
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
            )
            energies[ik, im] = np.sort(eigvals.real)
        if ik % 40 == 0 or ik == kx_c.size - 1:
            print(f"Computed k point {ik + 1}/{kx_c.size}", flush=True)

    return {
        "kx_c": np.asarray(kx_c),
        "z_modes": np.asarray(z_modes),
        "energies": energies,
    }


def load_or_calculate(*, force: bool = False) -> dict[str, np.ndarray]:
    if CACHE_PATH.exists() and not force:
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    kx_c = np.linspace(-0.12, 0.12, N_K)
    data = calculate_bands(PARAMS, kx_c=kx_c)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        **data,
        **{key: np.array(value) for key, value in PARAMS.items()},
    )
    return data


def plot_panel(
    ax: mpl.axes.Axes,
    data: dict[str, np.ndarray],
    *,
    energy_window: tuple[float, float] = (-2.0, 2.0),
    show_xlabel: bool = True,
) -> dict[str, object]:
    """Plot finite-y ribbon bands on an explicit axes."""
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
    ax.xaxis.labelpad = 4.0 * TEXT_SCALE
    ax.yaxis.labelpad = 4.0 * TEXT_SCALE
    ax.set_xticks(
        [-0.1, -0.05, 0.0, 0.05, 0.1],
        labels=[r"$-0.10$", r"$-0.05$", r"$0$", r"$0.05$", r"$0.10$"],
    )
    ax.set_yticks(
        [-2, -1, 0, 1, 2],
        labels=[r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$"],
    )
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)
    return {"ax": ax, "lines": lines}


def make_figure(data: dict[str, np.ndarray]) -> mpl.figure.Figure:
    """Build the enlarged band panel used in the manual Fig. 1 assembly."""
    with mpl.rc_context(large_text_style()):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
        plot_panel(ax, data)
        return fig


def save_figure(fig: mpl.figure.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = "B=0 superconducting ribbon bands at Ly/c=150 and Lz/c=20"
    subject = "BdG bands in units of Delta versus kx*c"
    fig.savefig(FIG_DIR / f"{FIG_BASENAME}.pdf", metadata={"Title": title, "Subject": subject})
    fig.savefig(
        FIG_DIR / f"{FIG_BASENAME}.png",
        dpi=600,
        metadata={"Description": subject},
    )
    fig.savefig(
        FIG_DIR / f"{FIG_BASENAME}.svg",
        metadata={"Title": title, "Description": subject},
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
