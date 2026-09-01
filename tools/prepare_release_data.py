#!/usr/bin/env python3
"""Build the compact local release package from the working project."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np


PUBLISHED_FIGURES = {
    "OverleafGit/Fig1.pdf": "Fig1.pdf",
    "OverleafGit/Fig2.pdf": "Fig2.pdf",
    "OverleafGit/Fig3.pdf": "Fig3.pdf",
    "OverleafGit/Fig4.pdf": "Fig4.pdf",
    "OverleafGit/Fig5.pdf": "Fig5.pdf",
    "OverleafGit/Fig6.pdf": "Fig6.pdf",
    "OverleafGit/FigTrilayerMP_sketch.pdf": "Fig7.pdf",
    "OverleafGit/FigA1.pdf": "FigA1.pdf",
    "OverleafGit/FigA2.pdf": "FigA2.pdf",
    "OverleafGit/FigA3.pdf": "FigA3.pdf",
}

PROCESSED_DATA = {
    "figspaper/cache/sc_bands_finite_y_B0_Lz20c_Ly150c.npz": "fig01_bands.npz",
    "figspaper/cache/fig1_gap_C0_G_Delta0p1.npz": "fig02_gap_map.npz",
    "figspaper/cache/fig1_ribbon_data.npz": "fig02_ribbon.npz",
    "figspaper/cache/fig2_phase_maps_G401_kz1201_L600.npz": "fig03_phase_map.npz",
    "figspaper/cache/fig3_pair_map_B151_L35_chi0p5_Q0p1.npz": "fig04_pair_map.npz",
    "Codes/results_fig3kx0/fig3casefine_A.npz": "fig04_spectrum_A.npz",
    "Codes/results_fig3kx0/fig3casefine_B.npz": "fig04_spectrum_B.npz",
    "Codes/results_fig3kx0/fig3statepair_A_kxplus.npz": "fig04_states_A.npz",
    "Codes/results_fig3kx0/fig3statepair_B_kxplus.npz": "fig04_states_B.npz",
    "Codes/Cache/compare_noise_wawb_localizados_cache.npz": "fig05_disorder_summary.npz",
    "figspaper/cache/fig4_2d_disorder_wa_W2p5_seed0_v2.npz": "fig05_state_A.npz",
    "figspaper/cache/fig4_2d_disorder_wb_W1p7_seed27_v2.npz": "fig05_state_B.npz",
    "figspaper/cache/fig4_3d_density_Wa0p5_v1.npz": "fig05_density_3d.npz",
    "figspaper/cache/fig4_3d_winding_counts_v1.npz": "fig05_winding_counts.npz",
    "figspaper/cache/fig5_spectra_Lz20_30_40_50_notebook_order_v1.npz": "fig06_spectra.npz",
    "figspaper/cache/fig5_spectrum_Lz20_4pi_notebook_order_v1.npz": "fig06_spectrum_4pi.npz",
    "Codes/results_parity/trilayer_S20_I6_S20_M0mid0p28.npz": "fig07_spectrum.npz",
    "Codes/results_parity/trilayer_mp_S20I6S20.npz": "fig07_spatial_mp.npz",
    "figspaper/cache/fig1appendix_C0fixed_G_Delta.npz": "figA1_mu_fixed.npz",
    "figspaper/cache/fig1appendix_Gfixed_C0_Delta.npz": "figA1_exchange_fixed.npz",
    "figspaper/cache/fig1appendix_TDelta.npz": "figA1_mixing.npz",
    "Codes/results_fig3kx0/fig3_pairmap_kx0_consolidated.npz": "figA2_pair_map.npz",
    "Codes/results_fig3kx0/fig3casefine_A.npz": "figA2_spectrum_A.npz",
    "Codes/results_fig3kx0/fig3casefine_B.npz": "figA2_spectrum_B.npz",
    "Codes/results_fig3kx0/fig3statepair_A_kx0.npz": "figA2_states_A.npz",
    "Codes/results_fig3kx0/fig3statepair_B_kx0.npz": "figA2_states_B.npz",
    "Codes/results_parity/parity_Ly300_Lz20_ID540172_0.npz": "figA3_gap_Lz10.npz",
    "Codes/results_parity/parity_Ly300_Lz30_ID540172_1.npz": "figA3_gap_Lz15.npz",
    "Codes/results_parity/parity_Ly300_Lz40_ID540172_2.npz": "figA3_gap_Lz20.npz",
    "Codes/results_parity/gap_lyscan_Lz20.npz": "figA3_width_scan.npz",
}


def copy_required(source_root: Path, destination: Path, files: dict[str, str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in files.items():
        source = source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / output_name)


def compact_figure_03(source_root: Path, destination: Path) -> None:
    source = source_root / "figspaper/cache/fig2_slab_spectrum_Nk31_bands20.npz"
    with np.load(source, allow_pickle=False) as raw:
        energy = np.asarray(raw["energy_kx"])
        k_index = energy.shape[0] // 2 + 1
        bands = np.argsort(np.abs(energy[k_index]))[[0, 3]]
        vectors = np.asarray(raw["eigenvectors_kx"])[k_index][:, bands]
        ly_sites = int(raw["Ly_sites"])
        lz_sites = int(raw["Lz_sites"])
        particle_hole = np.kron(np.array([[0, 1], [1, 0]], complex), np.eye(4))

        densities = []
        majorana = []
        for vector in vectors.T:
            sites = vector.reshape(-1, 8)
            majorana.append(
                float(
                    np.abs(
                        np.einsum(
                            "sa,ab,sb->",
                            sites,
                            particle_hole,
                            np.conj(sites),
                            optimize=True,
                        )
                    )
                    ** 2
                )
            )
            density = np.sum(np.abs(sites) ** 2, axis=1)
            densities.append(density.reshape(ly_sites, lz_sites).T)

        np.savez_compressed(
            destination / "fig03_slab.npz",
            kx_values=raw["kx_values"],
            energy_kx=energy,
            Q_expect=raw["Q_expect"],
            Ly_sites=raw["Ly_sites"],
            Lz_sites=raw["Lz_sites"],
            L_y=raw["L_y"],
            L_z=raw["L_z"],
            a=raw["a"],
            c=raw["c"],
            Delta=raw["Delta"],
            selected_bands=bands,
            selected_energy=energy[k_index, bands],
            selected_chi=np.asarray(majorana),
            selected_density=np.asarray(densities),
        )


def combine_figure_06_map(source_root: Path, destination: Path) -> None:
    base_path = source_root / "figspaper/cache/fig5_pair_map_chi0p5_notebook_order_v1.npz"
    with np.load(base_path, allow_pickle=False) as base:
        phi = np.asarray(base["phi_vals"], dtype=float)
        thickness = np.asarray(base["lz_vals"], dtype=float)
        pair_grid = np.asarray(base["pair_grid"], dtype=int)

    extension_dir = source_root / "Codes/results_fig6b_lz35"
    paths = sorted(
        extension_dir.glob("fig6b_Ly300_Lz*_ID20260723legacy.npz"),
        key=lambda path: int(path.name.split("Lz")[1].split("_")[0]),
    )
    if len(paths) != 20:
        raise ValueError(f"Expected 20 Fig. 6 extension rows, found {len(paths)}")

    extra_thickness = []
    extra_rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as row:
            if not np.allclose(row["phi"], phi, rtol=0.0, atol=5e-6):
                raise ValueError(f"Incompatible phase grid in {path}")
            expected = np.asarray(row["nselected"], dtype=int) // 2
            observed = np.rint(row["npairs"]).astype(int)
            if not np.array_equal(expected, observed):
                raise ValueError(f"Inconsistent pair count in {path}")
            extra_thickness.append(float(row["Lz"]))
            extra_rows.append(observed)

    thickness = np.concatenate([thickness, extra_thickness])
    pair_grid = np.vstack([pair_grid, extra_rows])
    if not np.array_equal(pair_grid[:, 0], pair_grid[:, -1]):
        raise ValueError("Fig. 6 endpoint columns are not 2pi-periodic")
    np.savez_compressed(
        destination / "fig06_pair_map.npz",
        phi=phi,
        thickness=thickness,
        pair_grid=pair_grid,
    )


def combine_figure_A3_map(source_root: Path, destination: Path) -> None:
    paths = sorted(
        (source_root / "Codes/results_parity").glob(
            "chimap_Ly300_Lz*_ID20260723Q02.npz"
        ),
        key=lambda path: int(path.name.split("Lz")[1].split("_")[0]),
    )
    if len(paths) != 31:
        raise ValueError(f"Expected 31 Fig. A3 map rows, found {len(paths)}")

    thickness = []
    rows = []
    phi = None
    for path in paths:
        with np.load(path, allow_pickle=False) as row:
            current_phi = np.asarray(row["phi"], dtype=float)
            if phi is None:
                phi = current_phi
            elif not np.allclose(current_phi, phi):
                raise ValueError(f"Incompatible phase grid in {path}")
            thickness.append(float(row["Lz"]))
            rows.append(np.asarray(row["npairs"], dtype=int))

    np.savez_compressed(
        destination / "figA3_pair_map.npz",
        phi=phi,
        thickness=np.asarray(thickness),
        pair_grid=np.asarray(rows),
    )


def copy_manual_sources(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_root / "figspaper/Keynotes/Fig0kEYNOTE.key",
        destination / "Figure_01.key",
    )
    shutil.copy2(
        source_root / "Results 3d copy.key",
        destination / "Figure_06.key",
    )
    shutil.copy2(
        source_root / "diagram-20260721.pdf",
        destination / "Figure_01_schematic.pdf",
    )
    with zipfile.ZipFile(source_root / "Results 3d copy.key") as archive:
        member = "Data/diagram-20260629-4-17299.pdf"
        with archive.open(member) as source, (destination / "Figure_06_schematic.pdf").open(
            "wb"
        ) as output:
            shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    release_root = Path(__file__).resolve().parents[1]
    data_dir = release_root / "data/processed"

    copy_required(workspace_root, release_root / "figures/published", PUBLISHED_FIGURES)
    copy_required(workspace_root, data_dir, PROCESSED_DATA)
    shutil.copy2(data_dir / "figA2_spectrum_A.npz", data_dir / "fig04_spectrum_A.npz")
    shutil.copy2(data_dir / "figA2_spectrum_B.npz", data_dir / "fig04_spectrum_B.npz")
    compact_figure_03(workspace_root, data_dir)
    combine_figure_06_map(workspace_root, data_dir)
    combine_figure_A3_map(workspace_root, data_dir)
    copy_manual_sources(workspace_root, release_root / "manual_sources")
    print(f"Prepared release data in {release_root}")


if __name__ == "__main__":
    main()
