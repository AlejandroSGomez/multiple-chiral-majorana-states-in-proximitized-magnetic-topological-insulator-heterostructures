"""Compute spectra and LDOS versus disorder in a finite 3D BHZ+SC sample.

For each disorder strength, the script evaluates several realizations,
computes the low-energy eigenpairs and integrated LDOS, and stores the results
in one ``.npz`` file.
"""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime
from typing import Dict, Iterable, Tuple

import kwant
import numpy as np
import scipy.sparse.linalg as sla
from kwant.digest import uniform


sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
s0 = np.eye(2, dtype=complex)


def sorted_eigs(ev: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = ev
    order = np.argsort(evals)
    return np.array(evals[order]), np.array(evecs[:, order])


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def gamma_matrices() -> Tuple[np.ndarray, ...]:
    g_z00 = kron3(sz, s0, s0)
    g_z0z = kron3(sz, s0, sz)
    g_00x = kron3(s0, s0, sx)
    g_zzy = kron3(sz, sz, sy)
    g_zzz = kron3(sz, sz, sz)
    g_0yy = kron3(s0, sy, sy)
    g_yyz = kron3(sy, sy, sz)
    g_xyz = kron3(sx, sy, sz)
    g_zyy = kron3(sz, sy, sy)
    g_zxx = kron3(sz, sx, sx)
    return g_z00, g_z0z, g_00x, g_zzy, g_zzz, g_0yy, g_yyz, g_zyy, g_xyz, g_zxx


GZ00, GZ0Z, G00X, GZZY, GZZZ, G0YY, GYYZ, GZYY, GXYZ, GZXX = gamma_matrices()


def disorder_potential(site: kwant.builder.Site, Wa_local: float, salt: str) -> float:
    return Wa_local * (uniform(site.tag, salt) - 0.5)


def build_disordered_cluster(params: Dict) -> kwant.system.FiniteSystem:
    a = params["a"]
    c = params["c"]
    L_y = params["L_y"]
    L_x = params["L_x"]
    L_z = params["L_z"]

    lat = kwant.lattice.general([(a, 0, 0), (0, a, 0), (0, 0, c)], basis=[(0, 0, 0)], norbs=8)

    def onsite(
        site,
        a,
        c,
        C0,
        C1,
        C2,
        M0,
        M1,
        M2,
        G,
        Delta,
        T,
        W_func,
        Wa,
        salt,
    ):
        ck2_xy = 2.0
        ck2_z = 1
        epsilon_0 = C0 + (2 * C2 / (a**2)) * ck2_xy + (2 * C1 / (c**2)) * ck2_z
        M_0 = M0 + (2 * M2 / (a**2)) * ck2_xy + (2 * M1 / (c**2)) * ck2_z
        return (
            (epsilon_0 + W_func(site, Wa, salt)) * GZ00
            + M_0 * GZ0Z
            + G * GZZZ
            + Delta * GYYZ
            + T * GZYY
        )

    def hopping_z(site1, site2, c, C1, M1, B, T):
        return (C1 / c**2) * GZ00 + (M1 / c**2) * GZ0Z + (1j * B / (2 * c) * G0YY)

    def hopping_y(site1, site2, a, C2, M2, A):
        return (C2 / a**2) * GZ00 + (M2 / a**2) * GZ0Z + (1j * A) / (2 * a) * GZZY

    def hopping_x(site1, site2, a, C2, M2, A):
        return (C2 / a**2) * GZ00 + (M2 / a**2) * GZ0Z + (1j * A) / (2 * a) * G00X

    syst = kwant.Builder()

    def shape(pos):
        x, y, z = pos
        return (0 <= x < L_x) and (0 <= y < L_y) and (0 <= z < L_z)

    syst[lat.shape(shape, (0, 0, 0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0, 0), lat.sublattices[0], lat.sublattices[0])] = hopping_x
    syst[kwant.builder.HoppingKind((0, 1, 0), lat.sublattices[0], lat.sublattices[0])] = hopping_y
    syst[kwant.builder.HoppingKind((0, 0, 1), lat.sublattices[0], lat.sublattices[0])] = hopping_z
    return syst.finalized()


DEFAULT_PARAMS = {
    "a": 8.0,
    "c": 2.0,
    "C0": 0.1,
    "C1": 0.0,
    "C2": 0.0,
    "M0": -0.28,
    "M1": 10.0,
    "M2": 56.59,
    "A": 4.1,
    "B": 0.0,
    "T": 0.45,
    "G": -0.46,
    "Delta": 0.1,
    "L_y": 200.0,
    "L_z": 50.0,
    "L_x": 200.0,
    "W_func": disorder_potential,
    "Wa": 0.4,
    "salt": "seed-0",
}


def compute_eigensystem(
    syst: kwant.system.FiniteSystem,
    params: Dict,
    n_eigs: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    H = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
    hilbert_dim = H.shape[0]
    k = min(int(n_eigs), max(1, hilbert_dim - 2))
    if k >= hilbert_dim:
        raise ValueError(f"n_eigs={n_eigs} incompatible with Hilbert space {hilbert_dim}")
    evals, evecs = sla.eigsh(H, k=k, sigma=sigma, return_eigenvectors=True)
    return sorted_eigs((evals, evecs))


def ldos_from_states(
    wavefuncs: np.ndarray,
    energies: np.ndarray,
    ldos_op: kwant.operator.Density,
    eps: float,
    n_sites: int,
) -> np.ndarray:
    ldos_flat = np.zeros(n_sites, dtype=float)
    sel = np.where(np.abs(energies) <= eps)[0]
    if sel.size == 0:
        return ldos_flat
    for j in sel:
        ldos_flat += ldos_op(wavefuncs[:, j]).real
    return ldos_flat


def site_indexer(
    syst: kwant.system.FiniteSystem, params: Dict
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    coords = np.array([site.pos for site in syst.sites])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError("Unexpected site positions for 3D system")

    a = params["a"]
    c = params["c"]

    x_coords = np.rint(coords[:, 0] / a).astype(int)
    y_coords = np.rint(coords[:, 1] / a).astype(int)
    z_coords = np.rint(coords[:, 2] / c).astype(int)

    x_vals = np.unique(x_coords)
    y_vals = np.unique(y_coords)
    z_vals = np.unique(z_coords)

    x_idx = np.searchsorted(x_vals, x_coords)
    y_idx = np.searchsorted(y_vals, y_coords)
    z_idx = np.searchsorted(z_vals, z_coords)

    lx_sites = len(x_vals)
    ly_sites = len(y_vals)
    lz_sites = len(z_vals)

    linear_index = x_idx + lx_sites * (y_idx + ly_sites * z_idx)
    total_sites = syst.graph.num_nodes
    if np.prod((lx_sites, ly_sites, lz_sites)) != total_sites:
        raise RuntimeError("Grid size mismatch with number of sites")
    return linear_index, (lx_sites, ly_sites, lz_sites)


def ldos_flat_to_grid(
    ldos_flat: np.ndarray, linear_index: np.ndarray, grid_shape: Tuple[int, int, int]
) -> np.ndarray:
    grid = np.zeros(np.prod(grid_shape), dtype=ldos_flat.dtype)
    grid[linear_index] = ldos_flat
    return grid.reshape(grid_shape)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Wa disorder calculations for the 3D cluster")
    parser.add_argument("--wa-list", type=float, nargs="+", default=None, help="disorder strengths to evaluate")
    parser.add_argument("--Wa", type=float, default=None, help="single Wa value")
    parser.add_argument("--repetitions", type=int, default=20, help="realizations per disorder strength")
    parser.add_argument("--n-eigs", type=int, default=40, help="eigenvalues nearest zero")
    parser.add_argument("--sigma", type=float, default=1e-4, help="spectral shift for eigsh")
    parser.add_argument("--eps", type=float, default=0.05, help="LDOS window |E| <= eps")
    parser.add_argument("--seed-base", type=int, default=1000, help="first disorder seed")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("calcs_cluster.npz"),
        help="output file",
    )
    parser.add_argument("--Lx", type=float, default=None, help="Lx in Angstrom")
    parser.add_argument("--Ly", type=float, default=None, help="Ly in Angstrom")
    parser.add_argument("--Lz", type=float, default=None, help="Lz in Angstrom")
    parser.add_argument("--a", type=float, default=None, help="in-plane lattice constant in Angstrom")
    parser.add_argument("--c", type=float, default=None, help="vertical lattice constant in Angstrom")
    return parser.parse_args(argv)


def override_params(base: Dict, args: argparse.Namespace) -> Dict:
    params = dict(base)
    if args.Lx is not None:
        params["L_x"] = float(args.Lx)
    if args.Ly is not None:
        params["L_y"] = float(args.Ly)
    if args.Lz is not None:
        params["L_z"] = float(args.Lz)
    if args.a is not None:
        params["a"] = float(args.a)
    if args.c is not None:
        params["c"] = float(args.c)
    return params


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.wa_list is None:
        if args.Wa is not None:
            args.wa_list = [float(args.Wa)]
        else:
            args.wa_list = [DEFAULT_PARAMS["Wa"]]

    params = override_params(DEFAULT_PARAMS, args)
    params["Wa"] = float(args.wa_list[0])
    params.setdefault("salt", "seed-0")

    syst = build_disordered_cluster(params)
    op_mat = np.diag(np.ones(8, dtype=float))
    ldos_op = kwant.operator.Density(syst, op_mat)
    linear_index, grid_shape = site_indexer(syst, params)

    n_sites = syst.graph.num_nodes
    seeds = args.seed_base + np.arange(args.repetitions)

    metadata = {
        "wa_list": np.array(args.wa_list, dtype=float),
        "seeds": seeds,
        "eps": args.eps,
        "n_eigs_requested": args.n_eigs,
        "sigma": args.sigma,
        "grid_shape": np.array(grid_shape, dtype=int),
        "Lx_ang": params["L_x"],
        "Ly_ang": params["L_y"],
        "Lz_ang": params["L_z"],
        "a_ang": params["a"],
        "c_ang": params["c"],
    }

    start = datetime.now()
    results: Dict[str, np.ndarray] = {}

    for wa in args.wa_list:
        ens_acc = []
        wfs_acc = []
        ldos_acc = []
        loop_params = dict(params)
        loop_params["Wa"] = float(wa)
        for seed in seeds:
            loop_params["salt"] = f"seed-{int(seed)}"
            energies, wavefuncs = compute_eigensystem(
                syst,
                dict(loop_params),
                args.n_eigs,
                args.sigma,
            )
            ldos_flat = ldos_from_states(wavefuncs, energies, ldos_op, args.eps, n_sites)
            ldos_map = ldos_flat_to_grid(ldos_flat, linear_index, grid_shape)

            ens_acc.append(energies)
            wfs_acc.append(wavefuncs)
            ldos_acc.append(ldos_map)

        wa_label = f"{wa:g}"
        results[f"ens_{wa_label}"] = np.stack(ens_acc, axis=0)
        results[f"wfs_{wa_label}"] = np.stack(wfs_acc, axis=0)
        results[f"ldos_maps_{wa_label}"] = np.stack(ldos_acc, axis=0)

    results.update(metadata)
    np.savez(args.out, **results)

    elapsed = datetime.now() - start
    print(f"Wrote {args.out} with keys: {list(results.keys())}")
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
