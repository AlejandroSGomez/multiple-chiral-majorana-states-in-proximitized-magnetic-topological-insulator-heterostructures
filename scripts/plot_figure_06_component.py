#!/usr/bin/env python3
"""Build the computational part of Fig. 6 with panel (a) left blank."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter

try:
    from .plot_figures import (
        CACHE,
        C0,
        DELTA0,
        DPI,
        PANEL_LABEL_SIZE,
        ROOT,
        _LETTERS,
        _bin_edges,
        _phi_fmt,
        group_letter,
        load_npz,
    )
except ImportError:
    from plot_figures import (
        CACHE,
        C0,
        DELTA0,
        DPI,
        PANEL_LABEL_SIZE,
        ROOT,
        _LETTERS,
        _bin_edges,
        _phi_fmt,
        group_letter,
        load_npz,
    )


PAIR_COLORS = [
    "#30345f",
    "#5b62b5",
    "#8fa96a",
    "#d5c36a",
    "#e79d53",
    "#ad4f56",
]
OUTPUT_DIR = ROOT / "figures" / "generated"
PREVIEW = OUTPUT_DIR / "Fig6_component.png"


def load_extended_map():
    data = load_npz(CACHE / "fig06_pair_map.npz")
    phi = np.asarray(data["phi"], dtype=float)
    thickness = np.asarray(data["thickness"], dtype=float)
    pair_grid = np.asarray(data["pair_grid"], dtype=int)
    if not np.array_equal(pair_grid[:, 0], pair_grid[:, -1]):
        raise ValueError("Fig. 6 endpoint columns are not 2pi-periodic")
    return phi, thickness, pair_grid


def build_figure_06_component():
    """Compose the updated panels (b)--(f) on the established Fig. 6 canvas."""
    f_label, f_tick, f_title = 28, 24, 26
    f_cbt, f_cbtitle = 22, 26

    phi_v, lz, pair_grid = load_extended_map()
    lz_c = lz / C0
    spec = load_npz(CACHE / "fig06_spectra.npz")
    spec4 = load_npz(CACHE / "fig06_spectrum_4pi.npz")

    values = np.arange(0, int(pair_grid.max()) + 1)
    bounds = np.arange(values[0] - 0.5, values[-1] + 1.5, 1.0)
    pair_cmap = ListedColormap(PAIR_COLORS[: values.size], name="pair_count")
    pair_norm = BoundaryNorm(bounds, pair_cmap.N, clip=True)
    chi_cmap, chi_norm = plt.get_cmap("viridis"), Normalize(0.0, 1.0)

    fig = plt.figure(figsize=(17.35, 11.6), dpi=DPI, constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1.45, 1.0],
        left=0.060,
        right=0.945,
        bottom=0.080,
        top=0.945,
        wspace=0.12,
        hspace=0.32,
    )
    spectra_grid = outer[1, :].subgridspec(
        1,
        4,
        wspace=0.16,
        width_ratios=[2, 1, 1, 1],
    )
    ax_placeholder = fig.add_subplot(outer[0, 0])
    ax_map = fig.add_subplot(outer[0, 1])
    ax_specs = [fig.add_subplot(spectra_grid[0, index]) for index in range(4)]
    ax_placeholder.set_axis_off()

    phi_edges = _bin_edges(phi_v)
    phi_edges[[0, -1]] = [phi_v[0], phi_v[-1]]
    lz_edges = _bin_edges(lz_c)
    pair_mesh = ax_map.pcolormesh(
        phi_edges,
        lz_edges,
        pair_grid,
        cmap=pair_cmap,
        norm=pair_norm,
        shading="flat",
        edgecolors="none",
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )

    # Repeat the endpoint samples outside the physical phase interval.
    phi_strip = 0.04 * (phi_v[-1] - phi_v[0])
    endpoint_strips = [
        ax_map.pcolormesh(
            [phi_v[0] - phi_strip, phi_v[0]],
            lz_edges,
            pair_grid[:, [0]],
            cmap=pair_cmap,
            norm=pair_norm,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=True,
            zorder=2,
        ),
        ax_map.pcolormesh(
            [phi_v[-1], phi_v[-1] + phi_strip],
            lz_edges,
            pair_grid[:, [-1]],
            cmap=pair_cmap,
            norm=pair_norm,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=True,
            zorder=2,
        ),
    ]
    endpoint_boundaries = [
        ax_map.axvline(
            boundary,
            color="white",
            lw=0.55,
            alpha=0.85,
            zorder=3,
        )
        for boundary in (
            phi_v[0],
            phi_v[-1],
        )
    ]
    phase_midline = ax_map.axvline(
        np.pi,
        color="white",
        ls=":",
        lw=0.9,
        zorder=3,
    )

    ax_map.set_xlabel(r"$\phi$", labelpad=2, fontsize=f_label)
    ax_map.set_ylabel(r"$L_z/c$", labelpad=4, fontsize=f_label)
    ax_map.set_xticks([phi_v[0], np.pi, phi_v[-1]])
    ax_map.xaxis.set_major_formatter(FuncFormatter(_phi_fmt))
    ax_map.set_yticks([5, 10, 15, 20, 25, 30, 35])
    ax_map.set_xlim(
        float(phi_v[0] - phi_strip),
        float(phi_v[-1] + phi_strip),
    )
    ax_map.set_ylim(float(lz_edges[0]), float(lz_edges[-1]))
    ax_map.spines["bottom"].set_bounds(float(phi_v[0]), float(phi_v[-1]))
    ax_map.grid(color="white", alpha=0.12, lw=0.4)
    ax_map.tick_params(axis="both", labelsize=f_tick, pad=2)

    fig.canvas.draw()
    map_position = ax_map.get_position()
    cax_pairs = fig.add_axes(
        [map_position.x1 + 0.008, map_position.y0, 0.011, map_position.height]
    )
    pair_colorbar = fig.colorbar(
        pair_mesh,
        cax=cax_pairs,
        boundaries=bounds,
        ticks=values,
        spacing="uniform",
    )
    pair_colorbar.ax.tick_params(labelsize=f_cbt, pad=2)
    pair_colorbar.ax.set_title(r"$N_{\rm pairs}$", fontsize=f_cbtitle, pad=10)

    spectrum_handles = []
    lz_list = [20, 30, 40, 50]
    for index, (axis, lz_value) in enumerate(zip(ax_specs, lz_list)):
        if index == 0:
            phis = spec4["phi"]
            energies = spec4["energy"]
            chis = spec4["chi"]
        else:
            key = f"{lz_value:g}"
            phis = spec[f"phi_{key}"]
            energies = spec[f"energy_{key}"]
            chis = spec[f"chi_{key}"]

        order = np.argsort(phis)
        phis = phis[order]
        energies = energies[order] / DELTA0
        chis = chis[order]
        axis_handles = []
        for band in range(energies.shape[1]):
            points = np.column_stack([phis, energies[:, band]])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(
                segments,
                cmap=chi_cmap,
                norm=chi_norm,
                linewidths=0.95,
                alpha=0.98,
            )
            collection.set_array(0.5 * (chis[:-1, band] + chis[1:, band]))
            axis.add_collection(collection)
            axis_handles.append(collection)
        spectrum_handles.append(axis_handles)

        axis.axhline(0.0, color="0.45", lw=0.7, ls="--", zorder=0)
        if index == 0:
            axis.axvline(2.0 * np.pi, color="0.55", lw=0.8, ls=":", zorder=0)
        axis.set_xlim(float(phis.min()), float(phis.max()))
        axis.set_ylim(-1.0, 1.0)
        axis.set_xticks(
            [0.0, np.pi, 2.0 * np.pi]
            if phis.max() <= 2.2 * np.pi
            else [0.0, 2.0 * np.pi, 4.0 * np.pi]
        )
        axis.xaxis.set_major_formatter(FuncFormatter(_phi_fmt))
        axis.set_yticks([-1.0, 0.0, 1.0])
        if index == 0:
            axis.set_ylabel(r"$E/\Delta$", labelpad=0, fontsize=f_label)
            axis.yaxis.set_label_coords(-0.22, 0.5)
        else:
            axis.set_yticklabels([])
        axis.set_xlabel(r"$\phi$", labelpad=2, fontsize=f_label)
        axis.set_title(
            rf"$L_z/c={lz_value / C0:g}$",
            fontsize=f_title,
            pad=8,
        )
        axis.grid(color="0.84", lw=0.5, alpha=0.8)
        axis.tick_params(axis="both", labelsize=f_tick, pad=2)

    fig.canvas.draw()
    spectrum_positions = [axis.get_position() for axis in ax_specs]
    cax_chi = fig.add_axes(
        [
            max(position.x1 for position in spectrum_positions) + 0.010,
            min(position.y0 for position in spectrum_positions),
            0.012,
            max(position.y1 for position in spectrum_positions)
            - min(position.y0 for position in spectrum_positions),
        ]
    )
    chi_colorbar = fig.colorbar(
        ScalarMappable(norm=chi_norm, cmap=chi_cmap),
        cax=cax_chi,
    )
    chi_colorbar.set_ticks([0.0, 0.5, 1.0])
    chi_colorbar.ax.tick_params(labelsize=f_cbt, pad=2)
    chi_colorbar.set_label(
        r"$\hat{\chi}_\nu$",
        fontsize=f_cbtitle,
        labelpad=5,
    )

    # The upper-left slot contains only its paper letter; the schematic itself
    # is intentionally left blank for composition in Keynote.
    letters = []
    figure_axes = [ax_placeholder, ax_map, *ax_specs]
    for index, (axis, label) in enumerate(zip(figure_axes, _LETTERS)):
        letters.append(
            group_letter(
                fig,
                [axis],
                label,
                fontsize=PANEL_LABEL_SIZE["Fig6_source"],
                xpad=-0.015 if index == 1 else -0.010,
                y=1.02,
                ypad=0.0,
                reference_axes=[ax_specs[0]] if index == 0 else None,
            )
        )

    axes = {
        "placeholder_a": ax_placeholder,
        "b": ax_map,
        "c": ax_specs[0],
        "d": ax_specs[1],
        "e": ax_specs[2],
        "f": ax_specs[3],
    }
    caxes = {"pairs": cax_pairs, "chi": cax_chi}
    handles = {
        "pair_mesh": pair_mesh,
        "endpoint_strips": endpoint_strips,
        "endpoint_boundaries": endpoint_boundaries,
        "phase_midline": phase_midline,
        "spectra": spectrum_handles,
        "letters": letters,
    }
    return fig, axes, caxes, handles


def validate_layout(fig, axes, caxes, handles) -> None:
    """Reject invalid extents and actual overlaps between neighboring artists.

    The historical Fig. 6 layout deliberately places the left ylabel and the
    right chi colorbar beyond the source canvas; the tight PDF/SVG export owns
    those margins.  Exported-file clipping is therefore checked independently.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = {
        **{key: axis for key, axis in axes.items() if key != "placeholder_a"},
        **{f"{key} colorbar": axis for key, axis in caxes.items()},
    }
    for name, artist in artists.items():
        bbox = artist.get_tightbbox(renderer)
        if bbox is None:
            raise RuntimeError(f"{name} has no rendered extent")
        if not np.isfinite(bbox.bounds).all():
            raise RuntimeError(f"{name} has a non-finite rendered extent")

    for index, letter in enumerate(handles["letters"]):
        bbox = letter.get_window_extent(renderer)
        if not np.isfinite(bbox.bounds).all():
            raise RuntimeError(f"Panel letter {_LETTERS[index]} has an invalid extent")

    label_b_bbox = handles["letters"][1].get_window_extent(renderer)
    for tick_label in axes["b"].get_yticklabels():
        if tick_label.get_visible() and tick_label.get_text():
            if label_b_bbox.overlaps(tick_label.get_window_extent(renderer)):
                raise RuntimeError("Panel letter (b) overlaps a panel-(b) y tick")

    if axes["b"].bbox.overlaps(caxes["pairs"].bbox):
        raise RuntimeError("Panel (b) and its colorbar axes overlap")

    map_text = [
        axes["b"].xaxis.label,
        axes["b"].yaxis.label,
        *axes["b"].get_xticklabels(),
        *axes["b"].get_yticklabels(),
        handles["letters"][1],
    ]
    cbar_text = [
        caxes["pairs"].title,
        *caxes["pairs"].get_yticklabels(),
    ]
    for map_artist in map_text:
        if not map_artist.get_visible() or not map_artist.get_text():
            continue
        map_artist_bbox = map_artist.get_window_extent(renderer)
        if map_artist_bbox.overlaps(caxes["pairs"].bbox):
            raise RuntimeError(
                "A panel-(b) label overlaps the pair-count colorbar"
            )
        for cbar_artist in cbar_text:
            if not cbar_artist.get_visible() or not cbar_artist.get_text():
                continue
            cbar_artist_bbox = cbar_artist.get_window_extent(renderer)
            if cbar_artist_bbox.overlaps(axes["b"].bbox):
                raise RuntimeError(
                    "Pair-count colorbar typography overlaps panel (b)"
                )
            if map_artist_bbox.overlaps(cbar_artist_bbox):
                raise RuntimeError(
                    "A panel-(b) label overlaps its colorbar typography"
                )

    for left_key, right_key in zip(("c", "d", "e"), ("d", "e", "f")):
        left_bbox = axes[left_key].get_tightbbox(renderer)
        right_bbox = axes[right_key].get_tightbbox(renderer)
        if left_bbox.overlaps(right_bbox):
            raise RuntimeError(
                f"Panels ({left_key}) and ({right_key}) have overlapping extents"
            )


def main() -> None:
    fig, axes, caxes, handles = build_figure_06_component()
    validate_layout(fig, axes, caxes, handles)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "Fig6_component.pdf"
    svg = OUTPUT_DIR / "Fig6_component.svg"
    fig.savefig(pdf, bbox_inches="tight", dpi=DPI)
    fig.savefig(svg, bbox_inches="tight", dpi=DPI)
    fig.savefig(PREVIEW, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {svg}")
    print(f"wrote {PREVIEW}")


if __name__ == "__main__":
    main()
