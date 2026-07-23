#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consolidate the per-``L_z`` rows from ``linear_b_scan.py`` into one
``N_pairs(B, L_z)`` map with the field layout used by the plotting scripts.

Usage:
    python collate_linear_b_scan.py --rows results_fig3kx0 \
        --out fig3_pairmap_kx0_consolidated.npz
"""
import argparse
from pathlib import Path

import numpy as np

B_VALUES = np.linspace(0.0, 2.2, 151)
LZ_VALUES = np.arange(2.0, 71.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="directory with fig3row_*.npz files")
    ap.add_argument("--out", required=True, help="output consolidated .npz")
    args = ap.parse_args()

    rows_dir = Path(args.rows)
    files = sorted(rows_dir.glob("fig3row_*.npz"))
    if not files:
        raise SystemExit(f"No fig3row_*.npz found in {rows_dir}")

    pair_map = np.full((LZ_VALUES.size, B_VALUES.size), -1, dtype=int)
    kx_seen = set()
    filled = np.zeros(LZ_VALUES.size, dtype=bool)

    for f in files:
        d = np.load(f)
        ilz = int(d["ilz"])
        row = d["pair_row"].astype(int)
        if row.size != B_VALUES.size:
            raise SystemExit(f"{f}: row size {row.size} != {B_VALUES.size}")
        if not np.allclose(d["B_vals"], B_VALUES):
            raise SystemExit(f"{f}: B grid mismatch")
        pair_map[ilz, :] = row
        filled[ilz] = True
        kx_seen.add(round(float(d["kx"]), 6))

    missing = np.where(~filled)[0]
    if missing.size:
        print(f"WARNING: missing L_z rows (indices) {missing.tolist()} "
              f"-> L_z = {LZ_VALUES[missing].tolist()}")
    if len(kx_seen) != 1:
        print(f"WARNING: multiple kx values across rows: {sorted(kx_seen)}")
    kx = float(sorted(kx_seen)[0])

    np.savez_compressed(
        args.out,
        B_vals=B_VALUES,
        Lz_vals=LZ_VALUES,
        pair_map=pair_map,
        kx=np.array(kx),
        chi_threshold=np.array(0.50),
        charge_threshold=np.array(0.10),
    )
    print(f"Wrote {args.out}  shape={pair_map.shape}  kx={kx:g}  "
          f"rows_filled={int(filled.sum())}/{LZ_VALUES.size}  "
          f"N_pairs in [{int(pair_map[pair_map >= 0].min())}, {int(pair_map.max())}]")


if __name__ == "__main__":
    main()
