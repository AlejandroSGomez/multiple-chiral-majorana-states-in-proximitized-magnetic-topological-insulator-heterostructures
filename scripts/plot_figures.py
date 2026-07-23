#!/usr/bin/env python3
"""Regenerate the computational figures from the bundled processed data.

Energies are measured in Delta=0.1 eV, lengths in c=2 Angstrom, and momenta
in inverse c. Run this file with manuscript labels such as ``Fig2`` or
``FigA1``. With no arguments, every figure implemented here is generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "figures" / "generated"

DELTA0 = 0.1     # energy scale (eV)
C0 = 2.0         # length scale (A)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsfonts}",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
DPI = 300
LABEL_SIZE = 20
TICK_SIZE = 16
TITLE_SIZE = 20
CBAR_TICK_SIZE = 16
CBAR_TITLE_SIZE = 20

# Panel-letter sizes are independent of axis typography.  They are chosen for
# a common 12 pt final size after scaling to the journal's 153 mm text width.
PANEL_LABEL_SIZE = {
    "Fig2": 24.2,
    "Fig3": 26.0,
    "Fig4": 27.3,
    "Fig5": 30.5,
    "Fig6_source": 37.6,
    "FigA1": 24.2,
    "FigA2": 27.3,
    "FigA3": 14.3,
}


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def group_letter(
    fig,
    axes,
    letter,
    *,
    fontsize,
    xpad=0.0,
    y=1.0,
    ypad=0.008,
    reference_axes=None,
):
    """Anchor a panel letter to the full rendered left edge of a panel group.

    The offsets retain the previous figure-relative vertical placement, while
    the horizontal position follows whichever artist is furthest left: ylabel,
    tick labels, title overhang, or the axes box itself.
    """
    axes = np.ravel(axes).tolist()
    reference_axes = axes if reference_axes is None else np.ravel(reference_axes).tolist()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    positions = {ax: ax.get_position() for ax in axes}
    left = min(pos.x0 for pos in positions.values())
    left_axes = [ax for ax, pos in positions.items() if np.isclose(pos.x0, left)]
    anchor = max(left_axes, key=lambda ax: positions[ax].y1)
    tight_bboxes = [ax.get_tightbbox(renderer) for ax in reference_axes]
    tight_bboxes = [bbox for bbox in tight_bboxes if bbox is not None]
    target_left_px = min((bbox.x0 for bbox in tight_bboxes), default=anchor.bbox.x0)
    target_offset_px = target_left_px - anchor.bbox.x0 + xpad * fig.bbox.width
    y_offset_pt = ypad * fig.get_figheight() * 72.0
    annotation = anchor.annotate(
        letter,
        xy=(0.0, y),
        xycoords="axes fraction",
        xytext=(0.0, y_offset_pt),
        textcoords="offset points",
        fontsize=fontsize,
        ha="left",
        va="bottom",
        annotation_clip=False,
        clip_on=False,
        zorder=20,
    )
    # Correct for the actual TeX glyph bearing so the visible left edge, not
    # merely the text anchor, lands exactly on the panel's rendered boundary.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    current_target_px = anchor.bbox.x0 + target_offset_px
    glyph_bbox = annotation.get_window_extent(renderer)
    x_offset_pt = (current_target_px - glyph_bbox.x0) * 72.0 / fig.dpi
    annotation.set_position((x_offset_pt, y_offset_pt))
    return annotation


_LETTERS = [r"$\rm{(a)}$", r"$\rm{(b)}$", r"$\rm{(c)}$", r"$\rm{(d)}$",
            r"$\rm{(e)}$", r"$\rm{(f)}$", r"$\rm{(g)}$"]


def createletters(fig, axs, *, size, ypad=0.008, xpads=None, ypads=None):
    xpads = {} if xpads is None else xpads
    ypads = {} if ypads is None else ypads
    for n, ax in enumerate(np.ravel(axs)):
        group_letter(
            fig,
            [ax],
            _LETTERS[n],
            fontsize=size,
            xpad=xpads.get(n, 0.0),
            ypad=ypads.get(n, ypad),
        )


def title_row_letters(fig, axs, *, size, xpads=None, ypads=None):
    """Align letters with the title row and the panel-(a) ylabel column."""
    axs = np.ravel(axs).tolist()
    xpads = {} if xpads is None else xpads
    ypads = {} if ypads is None else ypads
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylabel_bbox = axs[0].yaxis.label.get_window_extent(renderer)
    ylabel_center_x_px = 0.5 * (ylabel_bbox.x0 + ylabel_bbox.x1)
    ylabel_center_offset_px = axs[0].bbox.x0 - ylabel_center_x_px

    annotations = []
    for index, ax in enumerate(axs):
        title_bbox = ax.title.get_window_extent(renderer)
        target_center_x_px = (
            ax.bbox.x0 - ylabel_center_offset_px
            + xpads.get(index, 0.0) * fig.bbox.width
        )
        target_center_y_px = (
            0.5 * (title_bbox.y0 + title_bbox.y1)
            + ypads.get(index, 0.0) * fig.bbox.height
        )
        annotation = ax.annotate(
            _LETTERS[index],
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 0.0),
            textcoords="offset points",
            fontsize=size,
            ha="left",
            va="center",
            annotation_clip=False,
            clip_on=False,
            zorder=20,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        glyph_bbox = annotation.get_window_extent(renderer)
        glyph_center_x_px = 0.5 * (glyph_bbox.x0 + glyph_bbox.x1)
        x_offset_pt = (target_center_x_px - glyph_center_x_px) * 72.0 / fig.dpi
        glyph_center_y_px = 0.5 * (glyph_bbox.y0 + glyph_bbox.y1)
        y_offset_pt = (target_center_y_px - glyph_center_y_px) * 72.0 / fig.dpi
        annotation.set_position((x_offset_pt, y_offset_pt))
        annotations.append(annotation)
    return annotations


def _chern_label(ax, x, y, text, fontsize=16, *, color="black", outline=True):
    path_effects = [pe.withStroke(linewidth=1, foreground="white")] if outline else []
    ax.text(x, y, text, ha="center", va="center", color=color,
            fontsize=fontsize, fontweight="bold",
            path_effects=path_effects, zorder=9)


def figure_02() -> None:
    """Plot the phase map, ribbon bands, and edge density of Fig. 2."""
    gap = load_npz(CACHE / "fig02_gap_map.npz")
    rib = load_npz(CACHE / "fig02_ribbon.npz")
    p1, p2, Z = gap["p1_vals"], gap["p2_vals"], gap["Z"]
    Zc = np.clip(Z, 1e-8, None) / DELTA0

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI, constrained_layout=True)
    ax_gap, ax_band, ax_den = axs

    # (a) bulk gap map dE_min(mu, G)
    extent = [p1[0] / DELTA0, p1[-1] / DELTA0, p2[0] / DELTA0, p2[-1] / DELTA0]
    im = ax_gap.imshow(Zc, extent=extent, origin="lower", aspect="auto",
                       cmap="inferno", norm=LogNorm(vmin=Zc.min(), vmax=Zc.max()))
    ax_gap.set_xlabel(r"$\tilde{\mu}/\Delta$", fontsize=LABEL_SIZE)
    ax_gap.set_ylabel(r"$G/\Delta$", fontsize=LABEL_SIZE)
    ax_gap.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_gap.set_box_aspect(1)
    for x, y, t in [(1, 1, "[0]"), (1, 9, "[0]"), (9, 1, "[0]"), (6.2, 8.0, "[1]")]:
        _chern_label(ax_gap, x, y, rf"${t}$")
    cb = fig.colorbar(im, ax=ax_gap, fraction=0.046, pad=0.035)
    cb.set_ticks([1e-7, 1e-4, 1e-1])
    cb.ax.set_title(r"$\delta E_{\min}/\Delta$", fontsize=CBAR_TITLE_SIZE, pad=15)
    cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (b) ribbon bands coloured by <Q>
    kx = rib["kx_vals"] * C0
    E = rib["E_all"] / DELTA0
    Q = rib["Q_expect"]
    k_idx = int(rib["k_idx"])
    cmap_q, norm_q = plt.cm.PiYG, Normalize(-1, 1)
    for j in range(E.shape[1]):
        pts = np.column_stack([kx, E[:, j]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap_q, norm=norm_q, linewidths=1.4)
        lc.set_array(0.5 * (Q[:-1, j] + Q[1:, j]))
        ax_band.add_collection(lc)
    ax_band.plot(kx[k_idx], float(rib["E0"]) / DELTA0, "o", color="tab:red", ms=6)
    ax_band.plot(kx[k_idx], float(rib["E1"]) / DELTA0, "o", color="skyblue", ms=6)
    ax_band.set_xlim(-0.03 * C0, 0.03 * C0)
    ax_band.set_ylim(-1, 1)
    ax_band.set_xlabel(r"$k_x c$", fontsize=LABEL_SIZE)
    ax_band.set_ylabel(r"$E/\Delta$", fontsize=LABEL_SIZE)
    ax_band.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_band.grid(True)
    ax_band.set_facecolor("lightgrey")
    ax_band.set_box_aspect(1)
    smq = ScalarMappable(cmap=cmap_q, norm=norm_q)
    smq.set_array([])
    cbq = fig.colorbar(smq, ax=ax_band, fraction=0.046, pad=0.035)
    cbq.set_ticks([-1, 0, 1])
    cbq.ax.set_title(r"$\langle Q\rangle/e$", fontsize=CBAR_TITLE_SIZE, pad=15)
    cbq.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (c) edge density profile
    lengths = rib["lengths"] / C0
    d0, d1 = rib["dens0"], rib["dens1"]
    pos = np.concatenate([d0[d0 > 0], d1[d1 > 0]])
    floor = max(float(pos.min()) * 0.5, 1e-14) if pos.size else 1e-14
    d0c, d1c = np.clip(d0, floor, None), np.clip(d1, floor, None)
    ax_den.plot(lengths, d0c, "-", lw=1.4, color="tab:red")
    ax_den.fill_between(lengths, floor, d0c, color="tab:red", alpha=0.4)
    ax_den.plot(lengths, d1c, "-", lw=1.4, color="skyblue")
    ax_den.fill_between(lengths, floor, d1c, color="skyblue", alpha=0.4)
    ax_den.text(0.52, 0.80, rf"$\chi={float(rib['chi1']):.4f}$", transform=ax_den.transAxes,
                ha="center", va="center", fontsize=LABEL_SIZE)
    ax_den.set_yscale("log")
    ax_den.set_ylim(floor, 1.5 * max(float(d0c.max()), float(d1c.max())))
    ax_den.set_xlabel(r"$y/c$", fontsize=LABEL_SIZE)
    ax_den.set_ylabel(r"$\rho(y)$", fontsize=LABEL_SIZE)
    ax_den.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_den.set_xticks(np.arange(0, 1401, 350) / C0)
    ax_den.set_xlim(0, float(rib["total_length"]) / C0)
    ax_den.grid(True, which="both")
    ax_den.set_box_aspect(1)

    # Extra optical clearance from the upper-left y tick in panel (a).
    createletters(fig, axs, size=PANEL_LABEL_SIZE["Fig2"], xpads={0: -0.010})
    save(fig, "Fig2.pdf")


def figure_A1() -> None:
    """Plot the three complementary phase maps of Fig. A1."""
    specs = [
        ("figA1_mu_fixed.npz", r"$G/\Delta_0$", r"$\tilde{\mu}/\Delta_0=1.0$",
         [(1.8, 5.0, "[0]", "black"), (6.2, 8.0, "[1]", "black"), (8.0, 1.5, "[2]", "black")]),
        ("figA1_exchange_fixed.npz", r"$\tilde{\mu}/\Delta_0$", r"$G/\Delta_0=4.6$",
         [(1.8, 3.0, "[1]", "black"), (7.2, 7.2, "[0]", "black"),
          (1.15, 0.85, "[0]", "white")]),
        ("figA1_mixing.npz", r"$T/\Delta_0$", r"$G/\Delta_0=4.6,\ \tilde{\mu}/\Delta_0=1.0$",
         [(1.5, 8.2, "[0]", "black"), (5.0, 5.5, "[1]", "black"),
          (0.8, 0.8, "[2]", "black"), (8.5, 1.2, "[0]", "black")]),
    ]
    # Caches store Z[p1, p2], whereas imshow expects Z[y, x] = Z[p2, p1].
    Zs = [np.clip(load_npz(CACHE / s[0])["Z"].T, 1e-8, None) / DELTA0 for s in specs]
    norm = LogNorm(vmin=min(z.min() for z in Zs), vmax=max(z.max() for z in Zs))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI, sharey=True, constrained_layout=True)
    im = None
    for ax, spec, Z in zip(axs, specs, Zs):
        d = load_npz(CACHE / spec[0])
        ext = [d["p1_vals"][0] / DELTA0, d["p1_vals"][-1] / DELTA0,
               d["p2_vals"][0] / DELTA0, d["p2_vals"][-1] / DELTA0]
        im = ax.imshow(Z, extent=ext, origin="lower", aspect="auto", cmap="inferno", norm=norm)
        ax.set_xlabel(spec[1], fontsize=LABEL_SIZE)
        ax.set_title(spec[2], fontsize=TITLE_SIZE)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_box_aspect(1)
        for x, y, t, color in spec[3]:
            _chern_label(ax, x, y, rf"${t}$", fontsize=18, color=color, outline=False)
    axs[0].set_ylabel(r"$\Delta/\Delta_0$", fontsize=LABEL_SIZE)
    cb = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.035)
    cb.set_ticks([1e-7, 1e-4, 1e-1])
    cb.set_label(r"$\delta E_{\min}/\Delta_0$", fontsize=CBAR_TITLE_SIZE, labelpad=10)
    cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    # Freeze the constrained layout after the colorbar has been incorporated.
    # Letters share the title height and repeat panel (a)'s ylabel offset.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    title_row_letters(
        fig,
        axs,
        size=PANEL_LABEL_SIZE["FigA1"],
        xpads={1: 0.030, 2: 0.030},
        ypads={1: 0.020, 2: 0.020},
    )
    save(fig, "FigA1.pdf")


def figure_03() -> None:
    """Plot the slab phase map, spectrum, and selected states of Fig. 3."""
    phase = load_npz(CACHE / "fig03_phase_map.npz")
    slab = load_npz(CACHE / "fig03_slab.npz")

    G = phase["G_vals"] / DELTA0
    kz_c = phase["kz_vals"] * C0
    L_c = phase["L_vals"]                       # L_vals are layer counts == L/c
    gap = phase["gap_array"] / DELTA0
    chern = np.abs(phase["C_map"])
    c_lat = float(phase["c_lat"])

    kx_c = slab["kx_values"] * C0
    energy = slab["energy_kx"] / DELTA0
    Q = slab["Q_expect"]
    Ly_sites = int(slab["Ly_sites"])
    Lz_sites = int(slab["Lz_sites"])
    Ly_c = float(slab["L_y"]) / C0
    Lz_c = float(slab["L_z"]) / C0

    k_idx = energy.shape[0] // 2 + 1
    sel = [
        {
            "energy": float(energy_value) / DELTA0,
            "chi": float(chi),
            "density": density,
        }
        for energy_value, chi, density in zip(
            slab["selected_energy"],
            slab["selected_chi"],
            slab["selected_density"],
        )
    ]

    fig = plt.figure(figsize=(14.8, 4.5), dpi=DPI)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.55], wspace=0.5)
    gs_a = outer[0].subgridspec(2, 1, height_ratios=[0.8, 3], hspace=0.35)
    ax_gap = fig.add_subplot(gs_a[0])
    ax_count = fig.add_subplot(gs_a[1], sharex=ax_gap)

    # (a) bulk gap map dE_min(kz, G)
    kz_max = min(0.55 * C0, kz_c[-1])
    kz_stop = int(np.searchsorted(kz_c, kz_max, side="right"))
    Z = np.clip(gap[:kz_stop, :], 1e-3, 10.0)
    im_gap = ax_gap.imshow(Z, origin="lower",
                           extent=[G[0], G[-1], kz_c[0], kz_c[kz_stop - 1]],
                           aspect="auto", cmap="inferno",
                           norm=LogNorm(vmin=1e-3, vmax=10.0))
    ax_gap.set_ylabel(r"$k_z c$", fontsize=LABEL_SIZE)
    ax_gap.set_yticks([0.0, 0.5, 1.0])
    ax_gap.tick_params(axis="x", labelbottom=False)
    ax_gap.tick_params(axis="both", labelsize=TICK_SIZE)
    fig.canvas.draw()
    gap_pos = ax_gap.get_position()
    count_pos = ax_count.get_position()
    cax_gap = fig.add_axes([gap_pos.x1 + 0.004, gap_pos.y0, 0.010, gap_pos.height])
    cb_gap = fig.colorbar(im_gap, cax=cax_gap)
    cb_gap.ax.set_title(r"$\delta E_{\min}/\Delta$", fontsize=CBAR_TITLE_SIZE, pad=14)
    cb_gap.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # Absolute Chern number in each connected gapped region.  Unlike Fig. A1,
    # these labels deliberately have no white stroke/outline.
    chern_labels = [
        (1.3, 0.42, "[0]"),
        (4.6, 0.18, "[1]"),
        (8.3, 0.17, "[0]"),
    ]
    for x, y, label in chern_labels:
        _chern_label(ax_gap, x, y, rf"${label}$", outline=False)

    # (b) slab Chern count map
    vals = np.sort(np.unique(chern))
    cmap_d = plt.get_cmap("tab20b", len(vals))
    bounds = np.concatenate(([vals[0] - 0.5], vals + 0.5))
    norm_d = BoundaryNorm(bounds, cmap_d.N, clip=True)
    im_count = ax_count.imshow(chern, origin="lower",
                               extent=[G[0], G[-1], L_c[0], L_c[-1]],
                               aspect="auto", interpolation="none",
                               cmap=cmap_d, norm=norm_d)
    ax_count.text(4.6, 25.0, r"$\times$", va="center", ha="center", c="r",
                  fontsize=16, fontweight="bold")
    ax_count.set_xlabel(r"$G/\Delta$", fontsize=LABEL_SIZE)
    ax_count.set_ylabel(r"$L_z/c$", fontsize=LABEL_SIZE)
    ax_count.tick_params(axis="both", labelsize=TICK_SIZE)
    cax_count = fig.add_axes([count_pos.x1 + 0.004, count_pos.y0, 0.010, count_pos.height])
    cb_count = fig.colorbar(im_count, cax=cax_count, boundaries=bounds, ticks=vals,
                            spacing="proportional")
    cb_count.ax.set_title(r"$|C_{\rm slab}|$", fontsize=CBAR_TITLE_SIZE - 4, pad=14)
    cb_count.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (c) slab spectrum E(kx), coloured by <Q>
    ax_band = fig.add_subplot(outer[1])
    cmap_q = plt.cm.PiYG
    norm_q = Normalize(vmin=-1, vmax=1)
    for j in range(energy.shape[1]):
        pts = np.column_stack([kx_c, energy[:, j]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap_q, norm=norm_q, linewidths=1.4)
        lc.set_array(0.5 * (Q[:-1, j] + Q[1:, j]))
        ax_band.add_collection(lc)
    state_colors = ["tab:red", "tab:blue", "gold", "limegreen"]
    for n, st in enumerate(sel):
        ax_band.scatter(kx_c[k_idx], st["energy"], s=34, color=state_colors[n % 4],
                        edgecolor="k", linewidth=0.4, zorder=4)
    ax_band.set_xlim(kx_c[0], kx_c[-1])
    ax_band.set_ylim(-1.0, 1.0)
    ax_band.set_xlabel(r"$k_x c$", fontsize=LABEL_SIZE)
    ax_band.set_ylabel(r"$E/\Delta$", fontsize=LABEL_SIZE, labelpad=-2)
    ax_band.set_facecolor("lightgrey")
    ax_band.grid(alpha=0.35)
    ax_band.tick_params(axis="both", labelsize=TICK_SIZE)
    sm_q = ScalarMappable(cmap=cmap_q, norm=norm_q)
    sm_q.set_array([])
    # Dedicated cax: the bar has exactly the spectrum height.  The fixed
    # physical gap also keeps the wide <Q>/e title clear of panel (c).
    divider_q = make_axes_locatable(ax_band)
    cax_q = divider_q.append_axes("right", size="5.0%", pad=0.24)
    cb_q = fig.colorbar(sm_q, cax=cax_q)
    cb_q.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb_q.ax.set_title(r"$\langle Q\rangle/e$", fontsize=CBAR_TITLE_SIZE, pad=14)
    cb_q.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (d,e) selected densities
    gs_c = outer[2].subgridspec(len(sel), 1, hspace=0.28)
    axs_den = [fig.add_subplot(gs_c[i]) for i in range(len(sel))]
    for ax in axs_den[1:]:
        ax.sharex(axs_den[0]); ax.sharey(axs_den[0])
    vmax = max(max(float(np.max(s["density"])) for s in sel), 1e-2)
    norm_den = Normalize(vmin=0.0, vmax=vmax)
    chi_labels = [r"$\chi^L_2$", r"$\chi^R_3$"]
    for n, (ax, st) in enumerate(zip(axs_den, sel)):
        im_den = ax.imshow(st["density"], origin="lower",
                           extent=[0, Ly_c, 0, Lz_c], aspect="auto",
                           vmin=0.0, vmax=vmax, cmap="YlOrBr")
        ax.text(0.52, 0.62, rf"{chi_labels[n][:-1]} = {st['chi']:.3f}$",
                transform=ax.transAxes, fontsize=LABEL_SIZE, ha="center", va="center")
        ax.set_ylabel(r"$z/c$", fontsize=LABEL_SIZE)
        ax.set_yticks([0, Lz_c / 2, Lz_c])
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        if n < len(axs_den) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$y/c$", fontsize=LABEL_SIZE)
    cb_den = fig.colorbar(ScalarMappable(norm=norm_den, cmap="YlOrBr"), ax=axs_den,
                          fraction=0.045, pad=0.03)
    cb_den.ax.set_title(r"$\rho(y,z)$", fontsize=CBAR_TITLE_SIZE, pad=14)
    cb_den.set_ticks(np.linspace(0.0, vmax, 4))
    cb_den.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb_den.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    fig3_xpads = {3: -0.010, 4: -0.010}
    fig3_ypads = {1: -0.004}
    for index, (ax, label) in enumerate(zip([ax_gap, ax_count, ax_band, *axs_den], _LETTERS)):
        group_letter(
            fig,
            [ax],
            label,
            fontsize=PANEL_LABEL_SIZE["Fig3"],
            xpad=fig3_xpads.get(index, 0.0),
            ypad=fig3_ypads.get(index, 0.008),
        )
    save(fig, "Fig3.pdf")


_UNIT_C8 = np.kron(np.array([[0, 1], [1, 0]], complex), np.eye(4))


def _global_chi(vec):
    v = vec.reshape(-1, 8)
    return float(np.abs(np.einsum("sa,ab,sb->", v, _UNIT_C8, np.conj(v), optimize=True))**2)


def _density_yz(vec, Ly_sites, Lz_sites):
    d = np.sum(np.abs(vec.reshape(Ly_sites * Lz_sites, 8))**2, axis=1)
    return d.reshape([Ly_sites, Lz_sites]).T


_KP0_COLOR, _KP1_COLOR = "#2aa84a", "#d99c00"
_STATE_PAIR_MARKERS = ("o", "D")


def _panels_cdef(fig, right, cases, chi_kind):
    """Draw the (c,d) fine-kx spectra + (e,f) density panels for the linear-term
    figures. The spectra always come from the fine-kx cache. By default, the
    marked states can be supplied through a state_pair cache containing the two
    lowest positive-energy bands at the requested momentum. This avoids plotting
    redundant particle-hole partners of the same BdG excitation."""
    selected_states = []
    for cs in cases:
        if "state_pair" in cs:
            d0 = cs["state_pair"]
            selected_states.append([
                {
                    "kx": float(d0["kx"]),
                    "energy": float(d0[f"state{n}_energy"]),
                    "density": d0[f"state{n}_density"],
                    "chi": float(d0[f"state{n}_chi"]),
                    "color": color,
                    "marker": marker,
                }
                for n, (color, marker) in enumerate(
                    ((_KP0_COLOR, _STATE_PAIR_MARKERS[0]),
                     (_KP1_COLOR, _STATE_PAIR_MARKERS[1]))
                )
            ])
        elif "kx0_states" in cs:
            d0 = cs["kx0_states"]
            selected_states.append([
                {
                    "kx": float(d0["kx0"]),
                    "energy": float(d0[f"s{n}_energy"]),
                    "density": d0[f"s{n}_density"],
                    "chi": float(d0[f"s{n}_chi"]),
                    "color": color,
                    "marker": marker,
                }
                for n, (color, marker) in enumerate(((_KP1_COLOR, "v"), (_KP0_COLOR, "^")))
            ])
        else:
            d = cs["d"]
            selected_states.append([
                {
                    "kx": float(d[f"{tag}_kx"]),
                    "energy": float(d[f"{tag}_energy"]),
                    "density": d[f"{tag}_density"],
                    "chi": float(d[f"{tag}_chi_regional"]),
                    "color": color,
                    "marker": "o",
                }
                for tag, color in (("kp0", _KP0_COLOR), ("kp1", _KP1_COLOR))
            ])

    vmax = max(float(np.max(state["density"]))
               for case_states in selected_states for state in case_states)
    vmax = max(vmax, 1e-2)
    norm_den = Normalize(0.0, vmax)

    # (c,d) spectra
    band_axes, sm_q = [], None
    for col, cs in enumerate(cases):
        d = cs["d"]
        kx = d["kx_values"] * C0
        e = d["energy_kx"] / DELTA0
        Q = d["Q_expect"]
        share = band_axes[0] if band_axes else None
        ax = fig.add_subplot(right[0, col], sharex=share, sharey=share)
        cmap_q, norm_q = plt.cm.PiYG, Normalize(-1, 1)
        for b in range(e.shape[1]):
            pts = np.column_stack([kx, e[:, b]]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap_q, norm=norm_q, linewidths=1.25)
            lc.set_array(0.5 * (Q[:-1, b] + Q[1:, b]))
            ax.add_collection(lc)
        for state in selected_states[col]:
            hollow = chi_kind == "regional" and state["marker"] == "D"
            ax.scatter(state["kx"] * C0, state["energy"] / DELTA0,
                       s=72, marker=state["marker"],
                       facecolor="none" if hollow else state["color"],
                       edgecolor=state["color"] if hollow else "k",
                       linewidth=1.4 if hollow else 0.6, zorder=6 if hollow else 5)
        ax.set_xlim(kx[0], kx[-1])
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel(r"$k_x c$", fontsize=LABEL_SIZE)
        ax.set_title(rf"{cs['label']}: $L_z/c={cs['L_z_c']:.0f}$", fontsize=CBAR_TITLE_SIZE - 2, pad=2)
        ax.set_facecolor("0.92")
        ax.grid(alpha=0.32)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        if share is not None:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"$E/\Delta$", fontsize=LABEL_SIZE)
        sm_q = ScalarMappable(cmap=cmap_q, norm=norm_q)
        band_axes.append(ax)
    cb_q = fig.colorbar(sm_q, ax=band_axes, fraction=0.035, pad=0.03)
    cb_q.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb_q.ax.set_title(r"$\langle Q\rangle/e$", fontsize=CBAR_TITLE_SIZE, pad=14)
    cb_q.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (e,f) densities corresponding exactly to the marked spectral states.
    den_axes = []
    for col, cs in enumerate(cases):
        d = cs["d"]
        Ly_c, Lz_c = float(d["L_y"]) / C0, float(d["L_z"]) / C0
        den_gs = right[1, col].subgridspec(2, 1, hspace=0.16)
        for i, state in enumerate(selected_states[col]):
            ax = fig.add_subplot(den_gs[i])
            ax.imshow(state["density"], origin="lower", extent=[0, Ly_c, 0, Lz_c],
                      aspect="auto", norm=norm_den, cmap="YlOrBr")
            hollow = chi_kind == "regional" and state["marker"] == "D"
            ax.scatter(0.27, 0.70, s=72, marker=state["marker"],
                       facecolor="none" if hollow else state["color"],
                       edgecolor=state["color"] if hollow else "k",
                       linewidth=1.4 if hollow else 0.6, transform=ax.transAxes,
                       clip_on=False, zorder=5)
            ax.text(0.60, 0.70, rf"$\hat{{\chi}}_\nu={state['chi']:.3f}$", transform=ax.transAxes,
                    fontsize=CBAR_TITLE_SIZE, ha="center", va="center")
            ax.set_ylabel(r"$z/c$", fontsize=LABEL_SIZE)
            ax.set_yticks([0, Lz_c / 2, Lz_c])
            ax.tick_params(axis="both", labelsize=TICK_SIZE)
            if i == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$y/c$", fontsize=LABEL_SIZE)
            den_axes.append(ax)
    cb_den = fig.colorbar(ScalarMappable(norm=norm_den, cmap="YlOrBr"), ax=den_axes, fraction=0.025, pad=0.025)
    cb_den.ax.set_title(r"$\rho(y,z)$", fontsize=CBAR_TITLE_SIZE, pad=14)
    cb_den.set_ticks(np.linspace(0.0, vmax, 4))
    cb_den.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb_den.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    return band_axes, den_axes


def figure_04() -> None:
    """Plot the linear-field pair map and representative states of Fig. 4."""
    BcD = DELTA0 * C0      # B is measured in eV.A; dimensionless B/(Delta c)
    pm = load_npz(CACHE / "fig04_pair_map.npz")
    B = pm["B_vals"] / BcD
    Lz = pm["Lz_vals"] / C0
    pair_map = pm["pair_map"].astype(int)
    cases = [
        {"label": "A", "color": "tab:blue", "B": 0.9 / BcD, "L_z": 50.0 / C0, "L_z_c": 50.0 / C0,
         "d": load_npz(CACHE / "fig04_spectrum_A.npz"),
         "state_pair": load_npz(CACHE / "fig04_states_A.npz")},
        {"label": "B", "color": "tab:red", "B": 1.2 / BcD, "L_z": 40.0 / C0, "L_z_c": 40.0 / C0,
         "d": load_npz(CACHE / "fig04_spectrum_B.npz"),
         "state_pair": load_npz(CACHE / "fig04_states_B.npz")},
    ]

    fig = plt.figure(figsize=(15.8, 8.7), dpi=DPI)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.28, 2.0], wspace=0.38)
    right = outer[0, 1].subgridspec(2, 2, wspace=0.30, hspace=0.38)
    leftg = outer[0, 0].subgridspec(2, 1, height_ratios=[3.0, 1.12], hspace=0.26)
    ax_map = fig.add_subplot(leftg[0])
    ax_cuts = fig.add_subplot(leftg[1], sharex=ax_map)

    # (a) pair map
    vals = np.arange(int(pair_map.min()), int(pair_map.max()) + 1)
    colors = ["#30345f", "#5b62b5", "#8fa96a", "#d5c36a", "#e79d53", "#ad4f56"]
    cmap = ListedColormap(colors[:len(vals)])
    bounds = np.concatenate(([vals[0] - 0.5], vals + 0.5))
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    im = ax_map.imshow(pair_map, origin="lower", extent=[B[0], B[-1], Lz[0], Lz[-1]],
                       aspect="auto", interpolation="none", cmap=cmap, norm=norm)
    for case in cases:
        ax_map.axhline(case["L_z"], color=case["color"], lw=0.9, alpha=0.55)
        ax_map.scatter(case["B"], case["L_z"], marker="X", s=90, color=case["color"],
                       edgecolors="white", linewidth=0.9, zorder=4)
        ax_map.text(case["B"] + 0.3, case["L_z"] + 1.1, case["label"], color=case["color"],
                    fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="center",
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    ax_map.set_ylabel(r"$L_z/c$", fontsize=LABEL_SIZE)
    ax_map.set_xticks([0, 2, 4, 6, 8, 10])
    ax_map.tick_params(axis="both", labelsize=TICK_SIZE, labelbottom=False)
    fig.canvas.draw()
    mp_pos = ax_map.get_position()
    cax = fig.add_axes([mp_pos.x1 + 0.016, mp_pos.y0 + 0.10 * mp_pos.height, 0.014, 0.80 * mp_pos.height])
    cbp = fig.colorbar(im, cax=cax, boundaries=bounds, ticks=vals, spacing="proportional")
    cbp.ax.set_title(r"$N_{\rm pairs}$", fontsize=CBAR_TITLE_SIZE, pad=12)
    cbp.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (b) cuts
    for case in cases:
        row = int(np.argmin(np.abs(Lz - case["L_z"])))
        col = int(np.argmin(np.abs(B - case["B"])))
        ax_cuts.plot(B, pair_map[row, :], drawstyle="steps-mid", lw=1.7, color=case["color"],
                     label=rf"{case['label']}: $L_z/c={case['L_z']:.0f}$")
        ax_cuts.axvline(case["B"], color=case["color"], lw=0.9, alpha=0.4)
        ax_cuts.scatter(B[col], pair_map[row, col], s=52, color=case["color"],
                        edgecolor="white", linewidth=0.7, zorder=3)
    ax_cuts.set_xlim(B[0], B[-1])
    ax_cuts.set_ylim(-0.25, int(pair_map.max()) + 0.35)
    ax_cuts.set_xlabel(r"$B/(\Delta c)$", fontsize=LABEL_SIZE)
    ax_cuts.set_ylabel(r"$N_{\rm pairs}$", fontsize=LABEL_SIZE)
    ax_cuts.set_yticks(np.arange(0, int(pair_map.max()) + 1, 1))
    ax_cuts.set_xticks([0, 2, 4, 6, 8, 10])
    ax_cuts.grid(alpha=0.3)
    ax_cuts.legend(frameon=False, fontsize=CBAR_TITLE_SIZE, handlelength=1.7)
    ax_cuts.tick_params(axis="both", labelsize=TICK_SIZE)

    # (c-f) two distinct positive-energy bands at kx*c=0+.
    band_axes, den_axes = _panels_cdef(fig, right, cases, "global")

    fig4_xpads = {0: -0.010, 1: -0.010, 3: -0.010, 4: -0.010, 5: -0.010}
    for index, (axes, label) in enumerate(zip(
        ([ax_map], [ax_cuts], [band_axes[0]], [band_axes[1]], den_axes[:2], den_axes[2:]),
        _LETTERS,
    )):
        group_letter(
            fig,
            axes,
            label,
            fontsize=PANEL_LABEL_SIZE["Fig4"],
            xpad=fig4_xpads.get(index, 0.0),
            # Panel (b) defines the left column; move (a) to match it.
            reference_axes=[ax_cuts] if index == 0 else None,
        )
    save(fig, "Fig4.pdf")


def _bin_edges(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5])
    d = np.diff(vals)
    e = np.empty(vals.size + 1)
    e[1:-1] = vals[:-1] + 0.5 * d
    e[0] = vals[0] - 0.5 * d[0]
    e[-1] = vals[-1] + 0.5 * d[-1]
    return e


def _phi_fmt(x, _pos=None):
    for k, lab in [(0, "0"), (1, r"\pi"), (2, r"2\pi"), (3, r"3\pi"), (4, r"4\pi")]:
        if np.isclose(x, k * np.pi, atol=0.08):
            return rf"${lab}$"
    return ""


def figure_05() -> None:
    """Plot the disorder robustness analysis of Fig. 5."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.ticker import MaxNLocator
    BLUE, RED = "#1f77b4", "#d62728"
    comp = load_npz(CACHE / "fig05_disorder_summary.npz")
    case_a = load_npz(CACHE / "fig05_state_A.npz")
    case_b = load_npz(CACHE / "fig05_state_B.npz")
    vol = load_npz(CACHE / "fig05_density_3d.npz")
    wind = load_npz(CACHE / "fig05_winding_counts.npz")

    fig = plt.figure(figsize=(17.2, 8.8), dpi=DPI)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.16, 0.92, 1.36], wspace=0.38)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[3.0, 1.12], hspace=0.18)
    ax_chi = fig.add_subplot(left[0])
    ax_gap = fig.add_subplot(left[1], sharex=ax_chi)
    middle = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.24)
    ax_den_A = fig.add_subplot(middle[0])
    ax_den_B = fig.add_subplot(middle[1], sharex=ax_den_A, sharey=ax_den_A)
    right = outer[0, 2].subgridspec(2, 1, height_ratios=[1.42, 0.86], hspace=0.34)
    ax_vox = fig.add_subplot(right[0], projection="3d")
    ax_wind = fig.add_subplot(right[1])

    # (a) edge Majorana polarization vs disorder
    wa, wb = comp["wa_values"] / DELTA0, comp["wb_values"] / DELTA0
    ax_chi.plot(wa, comp["wa_tilde_mean"], "o-", color=BLUE, lw=1.6, ms=4, label=r"$W_a$")
    ax_chi.fill_between(wa, comp["wa_tilde_mean"] - comp["wa_tilde_std"],
                        comp["wa_tilde_mean"] + comp["wa_tilde_std"], color=BLUE, alpha=0.2)
    ax_chi.plot(wb, comp["wb_tilde_mean"], "s-", color=RED, lw=1.6, ms=4, label=r"$W_b$")
    ax_chi.fill_between(wb, comp["wb_tilde_mean"] - comp["wb_tilde_std"],
                        comp["wb_tilde_mean"] + comp["wb_tilde_std"], color=RED, alpha=0.2)
    iA = int(np.argmin(np.abs(comp["wa_values"] - 2.5)))
    iB = int(np.argmin(np.abs(comp["wb_values"] - 1.7)))
    ax_chi.plot(2.5 / DELTA0, comp["wa_tilde_mean"][iA], "x", color=BLUE, ms=11, mew=2.6, zorder=5)
    ax_chi.text(2.5 / DELTA0, comp["wa_tilde_mean"][iA] + 0.45, "A", color=BLUE,
                fontsize=LABEL_SIZE, ha="center", fontweight="bold")
    ax_chi.plot(1.7 / DELTA0, comp["wb_tilde_mean"][iB], "x", color=RED, ms=11, mew=2.6, zorder=5)
    ax_chi.text(1.7 / DELTA0, comp["wb_tilde_mean"][iB] + 0.45, "B", color=RED,
                fontsize=LABEL_SIZE, ha="center", fontweight="bold")
    ax_chi.set_ylabel(r"$\tilde{\chi}$", fontsize=LABEL_SIZE)
    ax_chi.set_ylim(0, 8.6)
    ax_chi.legend(frameon=False, fontsize=TICK_SIZE, loc="lower left")
    ax_chi.grid(alpha=0.3)
    ax_chi.tick_params(axis="both", labelsize=TICK_SIZE, labelbottom=False)

    # (b) bulk gap vs disorder
    ax_gap.plot(wa, comp["wa_gap_mean"] / DELTA0, "o-", color=BLUE, lw=1.6, ms=4)
    ax_gap.fill_between(wa, (comp["wa_gap_mean"] - comp["wa_gap_std"]) / DELTA0,
                        (comp["wa_gap_mean"] + comp["wa_gap_std"]) / DELTA0, color=BLUE, alpha=0.2)
    ax_gap.plot(wb, comp["wb_gap_mean"] / DELTA0, "s-", color=RED, lw=1.6, ms=4)
    ax_gap.fill_between(wb, (comp["wb_gap_mean"] - comp["wb_gap_std"]) / DELTA0,
                        (comp["wb_gap_mean"] + comp["wb_gap_std"]) / DELTA0, color=RED, alpha=0.2)
    ax_gap.axvline(2.5 / DELTA0, color=BLUE, ls=":", lw=1.0)
    ax_gap.axvline(1.7 / DELTA0, color=RED, ls=":", lw=1.0)
    ax_gap.set_xlabel(r"$W/\Delta$", fontsize=LABEL_SIZE)
    ax_gap.set_ylabel(r"$\langle\delta E\rangle/\Delta$", fontsize=LABEL_SIZE)
    ax_gap.grid(alpha=0.3)
    ax_gap.tick_params(axis="both", labelsize=TICK_SIZE)

    # (c,d) selected-state densities with MP arrows
    vmax_den = max(float(np.max(case_a["density"])), float(np.max(case_b["density"])), 1e-12)
    norm_den = Normalize(0.0, vmax_den)

    def density_panel(ax, case, xlabel):
        den = case["density"]
        mp = case["mp_real"] + 1j * case["mp_imag"]
        a, c = float(case["a"]), float(case["c"])
        ly_c = int(case["Ly"]) * a / C0
        lz_c = int(case["Lz"]) * c / C0
        im = ax.imshow(den, origin="lower", extent=[0, ly_c, 0, lz_c], aspect="auto",
                       cmap="YlOrBr", norm=norm_den)
        yy = np.linspace(0, ly_c, den.shape[1])
        zz = np.linspace(0, lz_c, den.shape[0])
        YY, ZZ = np.meshgrid(yy, zz)
        sel = den > 0.25 * den.max()
        sub = np.zeros_like(sel); sub[::2, ::2] = True
        sel &= sub
        nrm = np.max(np.abs(mp)) + 1e-300
        ax.quiver(YY[sel], ZZ[sel], mp.real[sel] / nrm, mp.imag[sel] / nrm,
                  angles="uv", scale=14, width=0.006, color="navy")
        sym = str(case["noise_symbol"])
        ax.set_title(rf"${sym}/\Delta={float(case['value']) / DELTA0:g}$",
                     fontsize=CBAR_TITLE_SIZE - 1, pad=3)
        ax.text(0.52, 0.80, rf"$\hat{{\chi}}_\nu={float(case['chi_value']):.3f}$",
                transform=ax.transAxes, ha="center", va="center", fontsize=CBAR_TITLE_SIZE - 2)
        ax.set_ylabel(r"$z/c$", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        if xlabel:
            ax.set_xlabel(r"$y/c$", fontsize=LABEL_SIZE)
        else:
            ax.tick_params(labelbottom=False)
        return im

    density_panel(ax_den_A, case_a, xlabel=False)
    density_panel(ax_den_B, case_b, xlabel=True)
    for ax in (ax_den_A, ax_den_B):
        cb = fig.colorbar(ScalarMappable(norm=norm_den, cmap="YlOrBr"), ax=ax, fraction=0.054, pad=0.05)
        cb.ax.set_title(r"$\rho(y,z)$", fontsize=CBAR_TITLE_SIZE - 2, pad=10)
        cb.set_ticks(np.linspace(0.0, vmax_den, 4))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    # (e) 3D voxel density
    volume = vol["volume"]
    a3, c3 = float(vol["a"]), float(vol["c"])
    vmax3 = float(np.quantile(volume, 0.995)) or max(float(volume.max()), 1.0)
    norm3 = Normalize(0.0, vmax3)
    thr = max(float(np.quantile(volume, 0.95)), 0.03 * float(volume.max()))
    mask = volume >= thr
    colors = plt.get_cmap("YlOrBr")(norm3(volume))
    colors[..., 3] = np.clip(norm3(volume), 0.22, 0.95)
    xi, yi, zi = np.indices(np.array(volume.shape) + 1)
    ax_vox.voxels(xi * a3 / C0, yi * a3 / C0, zi * c3 / C0, mask, facecolors=colors, edgecolor="none")
    Lx_c, Ly_c, Lz_c = float(vol["Lx_ang"]) / C0, float(vol["Ly_ang"]) / C0, float(vol["Lz_ang"]) / C0
    ax_vox.set_xlabel(r"$L_x/c$", labelpad=5, fontsize=CBAR_TITLE_SIZE - 3)
    ax_vox.set_ylabel(r"$L_y/c$", labelpad=5, fontsize=CBAR_TITLE_SIZE - 3)
    ax_vox.set_zlabel(r"$L_z/c$", labelpad=2, fontsize=CBAR_TITLE_SIZE - 3)
    ax_vox.set_xlim3d([0, Lx_c]); ax_vox.set_ylim3d([0, Ly_c]); ax_vox.set_zlim3d([0, Lz_c])
    ax_vox.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_vox.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_vox.set_zticks([0.0, Lz_c])
    ax_vox.tick_params(axis="x", labelsize=CBAR_TICK_SIZE - 2, pad=0)
    ax_vox.tick_params(axis="y", labelsize=CBAR_TICK_SIZE - 2, pad=0)
    ax_vox.tick_params(axis="z", labelsize=CBAR_TICK_SIZE - 3, pad=-2)
    try:
        ax_vox.set_box_aspect((Lx_c, Ly_c, max(Lz_c * 2.2, 0.42 * Lx_c)))
    except Exception:
        pass
    ax_vox.view_init(elev=23, azim=-54)
    ax_vox.set_title(rf"3D density, $W_a/\Delta={float(vol['wa']) / DELTA0:g}$",
                     fontsize=CBAR_TITLE_SIZE - 1, pad=3)
    cb_vox = fig.colorbar(ScalarMappable(norm=norm3, cmap="YlOrBr"), ax=ax_vox,
                          fraction=0.026, pad=0.105, shrink=0.82)
    cb_vox.set_label(r"$\rho_{3D}$", fontsize=CBAR_TITLE_SIZE - 3, labelpad=4)
    cb_vox.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb_vox.ax.tick_params(labelsize=CBAR_TICK_SIZE - 1)

    # (f) 3D chiral-pair count vs Wa
    wav = wind["wa_values"]
    xpos = np.arange(wav.size)
    ax_wind.bar(xpos, 0.5 * wind["n_mean"], width=0.72, color="#4c72b0",
                edgecolor="k", linewidth=0.6, alpha=0.88, zorder=2)
    ax_wind.errorbar(xpos, 0.5 * wind["n_mean"], yerr=0.5 * wind["n_std"], fmt="none",
                     ecolor="k", elinewidth=0.8, capsize=3, zorder=3)
    ax_wind.set_xticks(xpos)
    ax_wind.set_xticklabels([f"{x / DELTA0:g}" for x in wav], rotation=0)
    ax_wind.set_xlabel(r"$W_a/\Delta$", fontsize=LABEL_SIZE)
    ax_wind.set_ylabel(r"$\tilde{N}_{\rm pairs}$", fontsize=LABEL_SIZE)
    ax_wind.set_title(r"3D chiral pairs", fontsize=CBAR_TITLE_SIZE - 1, pad=3)
    ax_wind.set_ylim(0, 4.3)
    ax_wind.set_yticks([0, 1, 2, 3, 4])
    ax_wind.grid(axis="y", alpha=0.30)
    ax_wind.tick_params(axis="both", labelsize=TICK_SIZE)

    fig5_xpads = {2: -0.010, 3: -0.010}
    for index, (ax, label) in enumerate(
        zip([ax_chi, ax_gap, ax_den_A, ax_den_B, ax_vox, ax_wind], _LETTERS)
    ):
        group_letter(
            fig,
            [ax],
            label,
            fontsize=PANEL_LABEL_SIZE["Fig5"],
            xpad=fig5_xpads.get(index, 0.0),
            # Panel (b) defines the left column; move (a) to match it.
            reference_axes=[ax_gap] if index == 0 else None,
        )
    save(fig, "Fig5.pdf")


APX_LABEL, APX_TICK, APX_TITLE, APX_CB = 16, 13, 15, 14


def _letter_axes(fig, axes, x=-0.02, y=1.02):
    fig.canvas.draw()
    for n, ax in enumerate(np.ravel(axes)):
        pos = ax.get_position()
        fig.text(pos.x0 + x, pos.y1 + 0.012, _LETTERS[n], fontsize=APX_LABEL + 1,
                 ha="left", va="bottom")


def _edges(vals):
    return _bin_edges(vals)


def figure_A3() -> None:
    """Plot Josephson gaps, width scaling, and pair counts for Fig. A3."""
    gap_files = [
        CACHE / "figA3_gap_Lz10.npz",
        CACHE / "figA3_gap_Lz15.npz",
        CACHE / "figA3_gap_Lz20.npz",
    ]
    gap_rows = [load_npz(path) for path in gap_files]
    ly_scan = load_npz(CACHE / "figA3_width_scan.npz")
    pair_map = load_npz(CACHE / "figA3_pair_map.npz")
    expected_lz = np.arange(10.0, 72.0, 2.0)
    actual_lz = np.asarray(pair_map["thickness"])
    if not np.array_equal(actual_lz, expected_lz):
        raise ValueError(
            "Fig. A3 requires the complete validated Lz=10,12,...,70 data set; "
            f"found {actual_lz.tolist()}"
        )

    lzc = actual_lz / C0
    phi_map = pair_map["phi"] / np.pi
    npairs_raw = pair_map["pair_grid"]
    if not np.allclose(npairs_raw, np.rint(npairs_raw), rtol=0.0, atol=1e-12):
        raise ValueError("Fig. A3 pair count contains a non-integer value")
    if not np.array_equal(npairs_raw[:, 0], npairs_raw[:, -1]):
        raise ValueError("Fig. A3 phi=0 and 2pi pair-count columns do not match")
    npairs = np.rint(npairs_raw).astype(int)
    values = np.arange(0, int(npairs.max()) + 1)
    bounds = np.arange(values[0] - 0.5, values[-1] + 1.5, 1.0)
    pair_colors = ["#30345f", "#5b62b5", "#8fa96a", "#d5c36a", "#e79d53", "#ad4f56"]
    if len(values) > len(pair_colors):
        raise ValueError("Pair-count palette must be extended for N_pairs > 5")
    cmap = ListedColormap(pair_colors[:len(values)], name="pair_count")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    line_colors = {20: "#0072B2", 30: "#D55E00", 40: "#009E73"}

    local_style = {
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
    }
    with plt.rc_context(local_style):
        fig = plt.figure(figsize=(7.10, 2.55), dpi=DPI)
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.02, 0.98, 1.42],
            left=0.080,
            right=0.970,
            bottom=0.205,
            top=0.855,
            wspace=0.52,
        )
        right = grid[0, 2].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.055],
            wspace=0.10,
        )
        axes = {
            "a": fig.add_subplot(grid[0, 0]),
            "b": fig.add_subplot(grid[0, 1]),
            "c": fig.add_subplot(right[0, 0]),
        }
        cax = fig.add_subplot(right[0, 1])

        for row in gap_rows:
            lz = int(round(float(row["Lz"])))
            mask = row["phi"] <= 2 * np.pi + 1e-9
            axes["a"].semilogy(
                row["phi"][mask] / np.pi,
                row["gap_min"][mask] / DELTA0,
                color=line_colors[lz],
                label=rf"$L_z/c={lz / C0:g}$",
            )
        axes["a"].axvline(1.0, color="0.55", ls=":", lw=0.8)
        axes["a"].set_xlim(0.0, 2.0)
        axes["a"].set_xticks([0, 1, 2])
        axes["a"].set_xlabel(r"$\phi/\pi$")
        axes["a"].set_ylabel(r"$E_{\min}(k_x=0)/\Delta$")
        axes["a"].legend(frameon=False, loc="lower center", handlelength=1.5)
        axes["a"].grid(alpha=0.22, which="both", lw=0.45)

        axes["b"].semilogy(
            ly_scan["Ly"] / C0,
            ly_scan["gap0"] / DELTA0,
            "o-",
            ms=3.1,
            color="#0072B2",
            label=r"$\phi=0$",
        )
        axes["b"].semilogy(
            ly_scan["Ly"] / C0,
            ly_scan["gap_pi"] / DELTA0,
            "s--",
            ms=3.1,
            color="#E69F00",
            label=r"$\phi=\pi$",
        )
        axes["b"].set_xlabel(r"$L_y/c$")
        axes["b"].set_ylabel(r"$E_{\min}/\Delta$")
        axes["b"].set_title(r"$L_z/c=10$", pad=3)
        axes["b"].legend(frameon=False, loc="lower left", handlelength=1.5)
        axes["b"].grid(alpha=0.22, which="both", lw=0.45)

        phi_edges = _edges(phi_map)
        phi_edges[[0, -1]] = [phi_map[0], phi_map[-1]]
        image = axes["c"].pcolormesh(
            phi_edges,
            _edges(lzc),
            npairs,
            cmap=cmap,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        # Repeat the endpoint samples outside the physical phase interval.
        phi_strip = 0.04 * (phi_map[-1] - phi_map[0])
        y_edges = _edges(lzc)
        axes["c"].pcolormesh(
            [phi_map[0] - phi_strip, phi_map[0]],
            y_edges,
            npairs[:, [0]],
            cmap=cmap,
            norm=norm,
            shading="flat",
            rasterized=True,
            zorder=2,
        )
        axes["c"].pcolormesh(
            [phi_map[-1], phi_map[-1] + phi_strip],
            y_edges,
            npairs[:, [-1]],
            cmap=cmap,
            norm=norm,
            shading="flat",
            rasterized=True,
            zorder=2,
        )
        for boundary in (phi_map[0], phi_map[-1]):
            axes["c"].axvline(boundary, color="white", lw=0.55, alpha=0.85, zorder=3)
        axes["c"].axvline(1.0, color="white", ls=":", lw=0.9)
        axes["c"].set_xlim(
            phi_map[0] - phi_strip,
            phi_map[-1] + phi_strip,
        )
        axes["c"].set_ylim(_edges(lzc)[0], _edges(lzc)[-1])
        axes["c"].set_xticks([phi_map[0], 1, phi_map[-1]])
        axes["c"].spines["bottom"].set_bounds(phi_map[0], phi_map[-1])
        axes["c"].set_yticks([5, 10, 15, 20, 25, 30, 35])
        axes["c"].set_xlabel(r"$\phi/\pi$")
        axes["c"].set_ylabel(r"$L_z/c$")

        colorbar = fig.colorbar(
            image,
            cax=cax,
            boundaries=bounds,
            ticks=values,
            spacing="uniform",
        )
        colorbar.ax.set_title(r"$N_{\rm pairs}$", fontsize=8.5, pad=5)
        colorbar.ax.tick_params(labelsize=7.5, length=2.5)

        for key, label in zip(("a", "b", "c"), _LETTERS):
            group_letter(
                fig,
                [axes[key]],
                label,
                fontsize=PANEL_LABEL_SIZE["FigA3"],
                xpad=0.0 if key == "a" else -0.010,
                ypad=0.008,
            )

        out = OUTPUT_DIR / "FigA3.pdf"
        fig.savefig(out, dpi=DPI)
        fig.savefig(out.with_suffix(".svg"), dpi=DPI)
        plt.close(fig)
        print(f"wrote {out}")


def figure_A2() -> None:
    """Plot the exact-kx=0 pair map and representative states of Fig. A2."""
    BcD = DELTA0 * C0
    md = load_npz(CACHE / "figA2_pair_map.npz")
    B = md["B_vals"] / BcD
    Lz = md["Lz_vals"] / C0
    pair_map = md["pair_map"].astype(int)
    cases = [{"label": "A", "color": "tab:blue", "B": 0.9 / BcD, "L_z": 50.0 / C0, "L_z_c": 50.0 / C0,
              "d": load_npz(CACHE / "figA2_spectrum_A.npz"),
              "state_pair": load_npz(CACHE / "figA2_states_A.npz")},
             {"label": "B", "color": "tab:red", "B": 1.2 / BcD, "L_z": 40.0 / C0, "L_z_c": 40.0 / C0,
              "d": load_npz(CACHE / "figA2_spectrum_B.npz"),
              "state_pair": load_npz(CACHE / "figA2_states_B.npz")}]

    fig = plt.figure(figsize=(15.8, 8.7), dpi=DPI)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.28, 2.0], wspace=0.38)
    right = outer[0, 1].subgridspec(2, 2, wspace=0.30, hspace=0.38)
    leftg = outer[0, 0].subgridspec(2, 1, height_ratios=[3.0, 1.12], hspace=0.26)
    ax_map = fig.add_subplot(leftg[0])
    ax_cuts = fig.add_subplot(leftg[1], sharex=ax_map)

    vals = np.arange(int(pair_map.min()), int(pair_map.max()) + 1)
    colors = ["#30345f", "#5b62b5", "#8fa96a", "#d5c36a", "#e79d53", "#ad4f56"]
    cmap = ListedColormap(colors[:len(vals)])
    bounds = np.concatenate(([vals[0] - 0.5], vals + 0.5))
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    im = ax_map.imshow(pair_map, origin="lower", extent=[B[0], B[-1], Lz[0], Lz[-1]],
                       aspect="auto", interpolation="none", cmap=cmap, norm=norm)
    # Repeat the B=0 sample outside the physical field interval.
    b_strip = 0.04 * (B[-1] - B[0])
    ax_map.imshow(
        pair_map[:, [0]],
        origin="lower",
        extent=[B[0] - b_strip, B[0], Lz[0], Lz[-1]],
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        zorder=2,
    )
    ax_map.set_xlim(B[0] - b_strip, B[-1])
    ax_map.set_ylim(Lz[0], Lz[-1])
    ax_map.axvline(B[0], color="white", lw=0.8, alpha=0.85, zorder=3)
    ax_map.spines["bottom"].set_bounds(B[0], B[-1])
    for cs in cases:
        ax_map.axhline(cs["L_z"], color=cs["color"], lw=0.9, alpha=0.55)
        ax_map.scatter(cs["B"], cs["L_z"], marker="X", s=90, color=cs["color"],
                       edgecolors="white", linewidth=0.9, zorder=4)
        ax_map.text(cs["B"] + 0.3, cs["L_z"] + 1.1, cs["label"], color=cs["color"],
                    fontsize=LABEL_SIZE, fontweight="bold", ha="left", va="center",
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    ax_map.set_ylabel(r"$L_z/c$", fontsize=LABEL_SIZE)
    ax_map.set_xticks([0, 2, 4, 6, 8, 10])
    ax_map.tick_params(axis="both", labelsize=TICK_SIZE, labelbottom=False)
    fig.canvas.draw()
    mp_pos = ax_map.get_position()
    cax = fig.add_axes([mp_pos.x1 + 0.016, mp_pos.y0 + 0.10 * mp_pos.height, 0.014, 0.80 * mp_pos.height])
    cbp = fig.colorbar(im, cax=cax, boundaries=bounds, ticks=vals, spacing="proportional")
    cbp.ax.set_title(r"$N_{\rm pairs}$", fontsize=CBAR_TITLE_SIZE, pad=12)
    cbp.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    for cs in cases:
        row = int(np.argmin(np.abs(Lz - cs["L_z"])))
        col = int(np.argmin(np.abs(B - cs["B"])))
        ax_cuts.plot(B, pair_map[row, :], drawstyle="steps-mid", lw=1.7, color=cs["color"],
                     label=rf"{cs['label']}: $L_z/c={cs['L_z']:.0f}$")
        ax_cuts.axvline(cs["B"], color=cs["color"], lw=0.9, alpha=0.4)
        ax_cuts.scatter(B[col], pair_map[row, col], s=52, color=cs["color"], edgecolor="white", linewidth=0.7, zorder=3)
    ax_cuts.set_xlim(B[0] - b_strip, B[-1])
    ax_cuts.set_ylim(-0.25, int(pair_map.max()) + 0.35)
    ax_cuts.set_xlabel(r"$B/(\Delta c)$", fontsize=LABEL_SIZE)
    ax_cuts.set_ylabel(r"$N_{\rm pairs}$", fontsize=LABEL_SIZE)
    ax_cuts.set_yticks(np.arange(0, int(pair_map.max()) + 1, 1))
    ax_cuts.set_xticks([0, 2, 4, 6, 8, 10])
    ax_cuts.spines["bottom"].set_bounds(B[0], B[-1])
    ax_cuts.grid(alpha=0.3)
    ax_cuts.legend(frameon=False, fontsize=CBAR_TITLE_SIZE, handlelength=1.7)
    ax_cuts.tick_params(axis="both", labelsize=TICK_SIZE)

    # (c-f) two distinct positive-energy bands exactly at kx*c=0.
    band_axes, den_axes = _panels_cdef(fig, right, cases, "regional")

    for index, (axes_, label) in enumerate(zip(
        ([ax_map], [ax_cuts], [band_axes[0]], [band_axes[1]], den_axes[:2], den_axes[2:]),
        _LETTERS,
    )):
        group_letter(
            fig,
            axes_,
            label,
            fontsize=PANEL_LABEL_SIZE["FigA2"],
            xpad=-0.010,
            # Panel (b) defines the left column; move (a) to match it.
            reference_axes=[ax_cuts] if index == 0 else None,
        )
    save(fig, "FigA2.pdf")


FIGS = {
    "Fig2": figure_02,
    "Fig3": figure_03,
    "Fig4": figure_04,
    "Fig5": figure_05,
    "FigA1": figure_A1,
    "FigA2": figure_A2,
    "FigA3": figure_A3,
}


def main():
    which = sys.argv[1:] or list(FIGS)
    for name in which:
        if name not in FIGS:
            raise SystemExit(f"Unknown figure {name!r}; available: {list(FIGS)}")
        FIGS[name]()


if __name__ == "__main__":
    main()
