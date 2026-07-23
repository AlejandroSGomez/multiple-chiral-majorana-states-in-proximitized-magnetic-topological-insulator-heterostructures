#!/usr/bin/env python3
"""Build the APS-style main-text Figure 7 from cached Josephson data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "figures" / "generated"
SPECTRUM_FILE = DATA_DIR / "fig07_spectrum.npz"
MP_FILE = DATA_DIR / "fig07_spatial_mp.npz"
DELTA0 = 0.1
C0 = 2.0
PANEL_LABEL_SIZE = 14.1  # 12 pt after inclusion at the 153 mm text width
MP_ARROW_COLOR = "#00D5FF"
MP_ARROW_EDGE_COLOR = "#003847"

STATE_STYLE = {
    0: {"marker": "*", "color": "#D55E00", "size": 63, "label": r"state $n=0$"},
    2: {"marker": "D", "color": "#0072B2", "size": 29, "label": r"state $n=2$"},
}


def aps_style() -> dict:
    """Local APS-like typography and line styling for a dense double-column figure."""
    return {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 8.2,
        "axes.labelsize": 9.2,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "axes.formatter.use_mathtext": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "savefig.dpi": 600,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
    }


def load_data() -> tuple[dict, dict]:
    """Load the cached spectrum and spatial Majorana-polarization data."""
    with np.load(SPECTRUM_FILE, allow_pickle=False) as src:
        spectrum = {key: src[key] for key in src.files}
    with np.load(MP_FILE, allow_pickle=False) as src:
        mp_data = {key: src[key] for key in src.files}
    return spectrum, mp_data


def _edges(values: np.ndarray, step: float) -> np.ndarray:
    return np.concatenate([values - step / 2.0, [values[-1] + step / 2.0]])


def _selected_cases(spectrum: dict, mp_data: dict) -> list[dict]:
    """Map each spatial panel to the identical eigenstate in the spectrum cache."""
    delta = float(spectrum["Delta"])
    cases = []
    for phi_index, phi in enumerate(mp_data["phis"]):
        spectrum_index = int(np.argmin(np.abs(spectrum["phi"] - phi)))
        for state in STATE_STYLE:
            energy = float(mp_data["E"][phi_index, state])
            band = int(np.argmin(np.abs(spectrum["emat"][:, spectrum_index] - energy)))
            residual = abs(float(spectrum["emat"][band, spectrum_index]) - energy)
            if residual > 1e-10:
                raise ValueError(f"Could not match phi={phi}, state={state}: residual={residual}")
            cases.append({
                "phi_index": phi_index,
                "spectrum_index": spectrum_index,
                "phi_over_pi": float(phi / np.pi),
                "state": state,
                "band": band,
                "energy_over_delta": energy / delta,
            })
    return cases


def plot_spectrum_panel(
    ax: mpl.axes.Axes,
    spectrum: dict,
    cases: list[dict],
    *,
    cax: mpl.axes.Axes,
) -> dict:
    """Plot the low-energy spectrum and mark the six spatially shown states."""
    phi = np.asarray(spectrum["phi"], dtype=float) / np.pi
    energies = np.asarray(spectrum["emat"], dtype=float) / float(spectrum["Delta"])
    chi = np.asarray(spectrum["maxchi_mat"], dtype=float)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    x = np.repeat(phi[None, :], energies.shape[0], axis=0).ravel()
    y = energies.ravel()
    colors = chi.ravel()
    draw_order = np.argsort(colors)
    spectrum_artist = ax.scatter(
        x[draw_order],
        y[draw_order],
        c=colors[draw_order],
        cmap=cmap,
        norm=norm,
        s=3.6,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    ax.axhline(0.0, color="0.52", lw=0.65, zorder=0)
    for selected_phi in (0.5, 1.0):
        ax.axvline(selected_phi, color="0.62", lw=0.65, ls=(0, (1.5, 2.0)), zorder=0)

    marker_handles = []
    for state, style in STATE_STYLE.items():
        selected = [case for case in cases if case["state"] == state]
        ax.scatter(
            [case["phi_over_pi"] for case in selected],
            [case["energy_over_delta"] for case in selected],
            marker=style["marker"],
            s=style["size"],
            facecolor=style["color"] if state == 0 else "none",
            edgecolor="black" if state == 0 else style["color"],
            linewidth=0.75 if state == 0 else 1.25,
            zorder=8 if state == 0 else 9,
            clip_on=False,
        )
        marker_handles.append(Line2D(
            [], [], linestyle="none", marker=style["marker"],
            markerfacecolor=style["color"] if state == 0 else "none",
            markeredgecolor="black" if state == 0 else style["color"],
            markeredgewidth=0.75 if state == 0 else 1.1,
            markersize=7.2 if state == 0 else 5.2,
            label=style["label"],
        ))

    ax.set_xlim(-0.015, 2.0)
    ax.set_ylim(-0.06, 0.06)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-0.06, -0.03, 0.0, 0.03, 0.06])
    ax.set_xlabel(r"$\phi/\pi$", labelpad=1.5)
    ax.set_ylabel(r"$E_n(k_x=0,\phi)/\Delta$", labelpad=2.0)
    ax.legend(
        handles=marker_handles,
        loc="upper left",
        borderaxespad=0.45,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="0.75",
        handletextpad=0.35,
        borderpad=0.28,
        labelspacing=0.25,
    )

    colorbar = ax.figure.colorbar(spectrum_artist, cax=cax, ticks=[0.0, 0.5, 1.0])
    cax.set_title(r"$\hat{\chi}_\nu$", fontsize=9.0, pad=3.0)
    cax.tick_params(labelsize=7.4, pad=1.5)
    return {
        "ax": ax,
        "artist": spectrum_artist,
        "markers": marker_handles,
        "colorbar": colorbar,
    }


def plot_mp_panel(
    ax: mpl.axes.Axes,
    mp_data: dict,
    *,
    phi_index: int,
    state: int,
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    show_right_endpoint_tick: bool,
    rho_threshold: float = 0.15,
    y_stride: int = 3,
) -> dict:
    """Plot one normalized density map with local MP vectors overlaid."""
    rho = np.asarray(mp_data["rho"][phi_index, state], dtype=float)
    mp = (
        np.asarray(mp_data["MP_re"][phi_index, state], dtype=float)
        + 1j * np.asarray(mp_data["MP_im"][phi_index, state], dtype=float)
    )
    a_over_c = float(mp_data["a"]) / C0
    z_step = float(mp_data["c"]) / C0
    y = np.arange(int(mp_data["ny"])) * a_over_c
    z = np.arange(int(mp_data["nz"])) * z_step
    yy, zz = np.meshgrid(y, z, indexing="ij")
    rho_norm = rho / (float(rho.max()) + 1e-300)

    mesh = ax.pcolormesh(
        _edges(y, a_over_c),
        _edges(z, z_step),
        rho_norm.T,
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        zorder=0,
    )
    for boundary in (
        float(mp_data["Lz_bot"]) / C0,
        (float(mp_data["Lz_bot"]) + float(mp_data["Lz_mid"])) / C0,
    ):
        ax.axhline(boundary, color="0.20", lw=0.65, ls=(0, (3.0, 2.0)), alpha=0.9, zorder=2)

    selected = rho_norm > rho_threshold
    subsample = np.zeros_like(selected, dtype=bool)
    subsample[::y_stride, :] = True
    selected &= subsample
    u = mp.real[selected]
    v = mp.imag[selected]
    magnitude_max = float(np.hypot(u, v).max()) + 1e-300
    quiver = ax.quiver(
        yy[selected],
        zz[selected],
        u / magnitude_max,
        v / magnitude_max,
        angles="uv",
        pivot="mid",
        scale=14.5,
        width=0.009,
        headwidth=3.4,
        headlength=4.2,
        headaxislength=3.7,
        color=MP_ARROW_COLOR,
        edgecolor=MP_ARROW_EDGE_COLOR,
        linewidth=0.22,
        alpha=1.0,
        zorder=3,
    )

    chi = max(float(mp_data["chiL"][phi_index, state]), float(mp_data["chiR"][phi_index, state]))
    annotation = ax.text(
        0.50,
        0.91,
        rf"$\hat{{\chi}}_\nu={chi:.2f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        zorder=5,
    )

    style = STATE_STYLE[state]
    state_marker = ax.scatter(
        [0.93],
        [0.90],
        transform=ax.transAxes,
        marker=style["marker"],
        s=34 if state == 0 else 21,
        facecolor=style["color"] if state == 0 else "white",
        edgecolor="black" if state == 0 else style["color"],
        linewidth=0.65 if state == 0 else 1.0,
        zorder=6,
        clip_on=False,
    )

    ax.set_xlim(0.0, 150.0)
    ax.set_ylim(0.0, 23.0)
    ax.set_yticks([0, 10, 20])
    if show_xlabel:
        ax.set_xticks([0, 75, 150] if show_right_endpoint_tick else [0, 75])
        ax.set_xlabel(r"$y/c$", labelpad=1.0)
    else:
        ax.set_xticks([0, 75, 150])
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(rf"state $n={state}$" + "\n" + r"$z/c$", labelpad=2.0)
    else:
        ax.tick_params(labelleft=False)
    if show_title:
        phi_fraction = float(mp_data["phis"][phi_index] / np.pi)
        title = {0.0: r"$\phi=0$", 0.5: r"$\phi=\pi/2$", 1.0: r"$\phi=\pi$"}[phi_fraction]
        ax.set_title(title, pad=3.2)

    return {
        "ax": ax,
        "mesh": mesh,
        "quiver": quiver,
        "annotation": annotation,
        "state_marker": state_marker,
    }


def make_layout(*, width_in: float = 7.05, height_in: float = 3.72):
    """Create the named APS double-column layout with dedicated spectrum colorbar."""
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.58, 3.42],
        left=0.075,
        right=0.982,
        bottom=0.125,
        top=0.895,
        wspace=0.27,
    )
    left = outer[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.09)
    right = outer[0, 1].subgridspec(2, 3, wspace=0.085, hspace=0.38)
    axes = {"a": fig.add_subplot(left[0, 0])}
    caxes = {"chi": fig.add_subplot(left[0, 1])}
    for row in range(2):
        for col in range(3):
            letter = chr(ord("b") + 3 * row + col)
            axes[letter] = fig.add_subplot(right[row, col])
    return fig, axes, caxes


def _add_panel_letters(axes: dict[str, mpl.axes.Axes]) -> None:
    fig = next(iter(axes.values())).figure
    left_nudges = {letter: -0.012 for letter in ("c", "d", "f", "g")}
    for letter, ax in axes.items():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = ax.get_tightbbox(renderer)
        target_offset_px = (tight_bbox.x0 - ax.bbox.x0) if tight_bbox is not None else 0.0
        target_offset_px += left_nudges.get(letter, 0.0) * fig.bbox.width
        annotation = ax.annotate(
            rf"$\rm{{({letter})}}$",
            xy=(0.0, 1.010),
            xycoords="axes fraction",
            xytext=(0.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            annotation_clip=False,
            clip_on=False,
            zorder=20,
        )
        annotation.set_gid("panel-letter")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        target_left_px = ax.bbox.x0 + target_offset_px
        glyph_bbox = annotation.get_window_extent(renderer)
        annotation.set_position(((target_left_px - glyph_bbox.x0) * 72.0 / fig.dpi, 0.0))


def build_figure(*, width_in: float = 7.05, height_in: float = 3.72):
    """Compose the spectrum and six spatial Majorana-polarization panels."""
    spectrum, mp_data = load_data()
    cases = _selected_cases(spectrum, mp_data)
    fig, axes, caxes = make_layout(width_in=width_in, height_in=height_in)
    handles = {"spectrum": plot_spectrum_panel(axes["a"], spectrum, cases, cax=caxes["chi"])}

    spatial_handles = {}
    for row, state in enumerate((0, 2)):
        for col in range(3):
            letter = chr(ord("b") + 3 * row + col)
            spatial_handles[letter] = plot_mp_panel(
                axes[letter],
                mp_data,
                phi_index=col,
                state=state,
                show_xlabel=row == 1,
                show_ylabel=col == 0,
                show_title=row == 0,
                show_right_endpoint_tick=col == 2,
            )
    handles["spatial"] = spatial_handles
    _add_panel_letters(axes)
    return fig, axes, caxes, handles


def validate_layout(fig: mpl.figure.Figure, axes: dict, caxes: dict) -> None:
    """Reject clipped or overlapping content after a full canvas draw."""
    panel_letters = [
        text
        for ax in axes.values()
        for text in ax.texts
        if text.get_gid() == "panel-letter"
    ]
    # Panel letters deliberately occupy the inter-panel margin.  Exclude them
    # while checking the data/axis regions, then validate their own extents.
    for text in panel_letters:
        text.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tolerance = 1.0
    content_bboxes = {}
    for name, artist in {**axes, **{f"cbar_{key}": value for key, value in caxes.items()}}.items():
        bbox = artist.get_tightbbox(renderer)
        if bbox is None:
            continue
        content_bboxes[name] = bbox
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            raise RuntimeError(f"Clipped layout region {name}: {bbox.bounds} outside {canvas.bounds}")

    spectrum_right = content_bboxes["cbar_chi"].x1
    spatial_left = min(content_bboxes[key].x0 for key in "bdef")
    if spectrum_right >= spatial_left:
        raise RuntimeError("Spectrum colorbar overlaps the spatial-panel region")

    for row in ("bcd", "efg"):
        for left_name, right_name in zip(row[:-1], row[1:]):
            left_bbox = content_bboxes[left_name]
            right_bbox = content_bboxes[right_name]
            if left_bbox.x1 >= right_bbox.x0:
                raise RuntimeError(f"Neighboring panels {left_name} and {right_name} overlap")
    for top_name, bottom_name in zip("bcd", "efg"):
        top_bbox = content_bboxes[top_name]
        bottom_bbox = content_bboxes[bottom_name]
        if bottom_bbox.y1 >= top_bbox.y0:
            raise RuntimeError(f"Neighboring panels {top_name} and {bottom_name} overlap")

    for text in panel_letters:
        text.set_visible(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in panel_letters:
        bbox = text.get_window_extent(renderer)
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            raise RuntimeError(f"Clipped panel letter: {bbox.bounds} outside {canvas.bounds}")


def export_figure(fig: mpl.figure.Figure, output_pdf: Path) -> tuple[Path, Path]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    fig.savefig(output_pdf, dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(output_png, dpi=600, bbox_inches=None, pad_inches=0)
    return output_pdf, output_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "Fig7.pdf")
    parser.add_argument("--width", type=float, default=7.05)
    parser.add_argument("--height", type=float, default=3.72)
    args = parser.parse_args()

    with mpl.rc_context(aps_style()):
        fig, axes, caxes, _handles = build_figure(width_in=args.width, height_in=args.height)
        validate_layout(fig, axes, caxes)
        pdf, png = export_figure(fig, args.output)
        plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
