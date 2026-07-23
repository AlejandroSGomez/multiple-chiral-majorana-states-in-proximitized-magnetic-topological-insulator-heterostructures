"""Finite-size (Ly) scaling of the low-energy gap of the vertical JJ at kx=0.

Goal: decide whether the near-zero parity-changing crossings seen in the Pfaffian
diagnostic are protected (robust to system width) or just finite-size splitting
of the near-zero Majorana modes.

For each Ly it does a cheap min-gap scan over phi in [0, 2pi] (sparse eigsh only,
no Pfaffian) and records:
  - gap at phi=0 (the zero-bias near-zero-mode splitting),
  - gap at phi=pi,
  - the minimum gap over (0, 2pi) and the phi where it occurs.
If the crossings are finite-size, the phi=0 splitting and the crossing gaps decay
(roughly exponentially) and/or the crossing locations drift as Ly grows.

Usage:
    python josephson_width_scan.py --Lz 20 --Ly-list 150,200,250,300,400,500,600 \
        --n-phi 81 --out gap_lyscan_Lz20.npz
"""

from __future__ import annotations

import argparse
import pathlib
import numpy as np
import scipy.sparse.linalg as sla

from josephson_model import build_josephson_slab, DEFAULT_PARAMS, delta_func


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ly scaling of the JJ low-energy gap at kx=0")
    p.add_argument("--Lz", type=float, default=20.0)
    p.add_argument("--Ly-list", type=str, default="150,200,250,300,400,500,600")
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--phi-top", type=float, default=0.0)
    p.add_argument("--n-phi", type=int, default=81, help="phi grid over [0,2pi]")
    p.add_argument("--gap-bands", type=int, default=6)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("gap_lyscan.npz"))
    p.add_argument("--progress", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ly_list = [float(x) for x in args.Ly_list.split(",")]
    phis = np.linspace(0.0, 2.0 * np.pi, args.n_phi)

    gap_grid = np.zeros((len(ly_list), args.n_phi), dtype=float)
    gap0 = np.zeros(len(ly_list))
    gap_pi = np.zeros(len(ly_list))
    gap_min = np.zeros(len(ly_list))
    phi_min = np.zeros(len(ly_list))

    for li, ly in enumerate(ly_list):
        params = dict(DEFAULT_PARAMS)
        params["Delta_func"] = delta_func
        params["L_y"] = ly
        params["L_z"] = float(args.Lz)
        params["kx"] = float(args.kx)
        params["phi_top"] = float(args.phi_top)
        syst = build_josephson_slab(params)
        for i, phi in enumerate(phis):
            params["phi_rel"] = float(phi)
            h = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
            ens = sla.eigsh(h, k=args.gap_bands, sigma=0.0, return_eigenvectors=False)
            gap_grid[li, i] = float(np.min(np.abs(ens)))
        gap0[li] = gap_grid[li, 0]
        ipi = int(np.argmin(np.abs(phis - np.pi)))
        gap_pi[li] = gap_grid[li, ipi]
        # Exclude the endpoints when locating the interior crossing dip.
        interior = gap_grid[li, 1:-1]
        jmin = int(np.argmin(interior)) + 1
        gap_min[li] = gap_grid[li, jmin]
        phi_min[li] = phis[jmin]
        msg = (f"Ly={ly:.0f} dim={h.shape[0]}  gap(0)={gap0[li]:.3e}  "
               f"gap(pi)={gap_pi[li]:.3e}  min_gap={gap_min[li]:.3e} at phi/pi={phi_min[li]/np.pi:.3f}")
        print(msg, flush=True)
        if args.progress is not None:
            with open(args.progress, "w") as fh:
                fh.write(msg + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, Lz=float(args.Lz), kx=float(args.kx),
        Ly=np.array(ly_list), phi=phis, gap_grid=gap_grid,
        gap0=gap0, gap_pi=gap_pi, gap_min=gap_min, phi_min=phi_min,
    )
    print("\nSummary (Ly, gap0, gap_pi, min_gap, phi_min/pi):", flush=True)
    for li, ly in enumerate(ly_list):
        print(f"  {ly:.0f}  {gap0[li]:.3e}  {gap_pi[li]:.3e}  {gap_min[li]:.3e}  {phi_min[li]/np.pi:.3f}",
              flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
