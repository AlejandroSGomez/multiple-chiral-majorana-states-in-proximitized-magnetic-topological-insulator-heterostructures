"""Compute spectra and LDOS versus disorder strength ``Wa``.

For each disorder strength, the script evaluates several realizations over a
``kx`` grid and stores the eigenvalues, eigenvectors, and integrated LDOS maps
in one ``.npz`` file.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
from datetime import datetime
from typing import Dict, Iterable, Tuple

import kwant
import numpy as np
import scipy.sparse.linalg as sla
from kwant.digest import uniform
from tqdm import tqdm


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


def disorder_potential(site, Wa, salt):
    return Wa * (uniform(site.tag, salt) - 0.5)


def slab_onsite(
    site, kx, a, c, C0, C1, C2, M0, M1, M2, A, G, Delta, T, B, W_func, Wa, salt
):
    ck2_xy = 2.0 - np.cos(kx * a)
    ck2_z = 1
    epsilon_0 = C0 + (2 * C2 / (a ** 2)) * ck2_xy + (2 * C1 / (c ** 2)) * ck2_z
    M_0 = M0 + (2 * M2 / (a ** 2)) * ck2_xy + (2 * M1 / (c ** 2)) * ck2_z
    return (
        (epsilon_0 + W_func(site, Wa, salt)) * GZ00
        + (A / a) * (np.sin(kx * a) * G00X)
        + M_0 * GZ0Z
        + G * GZZZ
        + Delta * GYYZ
        + T * GZYY
    )


def slab_hopping_z(site1, site2, c, C1, M1, B, T):
    return (C1 / c**2) * GZ00 + (M1 / c**2) * GZ0Z + (1j * B / (2 * c) * G0YY)


def slab_hopping_y(site1, site2, a, C2, M2, A):
    return (C2 / a**2) * GZ00 + (M2 / a**2) * GZ0Z + (1j * A) / (2 * a) * GZZY


def build_disordered_slab(params: Dict) -> kwant.system.FiniteSystem:
    a = params["a"]
    c = params["c"]
    L_y = params["L_y"]
    L_z = params["L_z"]

    lat = kwant.lattice.general([(a, 0), (0, c)], basis=[(0, 0)], norbs=8)

    syst = kwant.Builder()

    def shape(pos):
        y, z = pos
        return (0 <= y < L_y) and (0 <= z < L_z)

    syst[lat.shape(shape, (0, 0))] = slab_onsite
    syst[kwant.builder.HoppingKind((1, 0), lat.sublattices[0], lat.sublattices[0])] = slab_hopping_y
    syst[kwant.builder.HoppingKind((0, 1), lat.sublattices[0], lat.sublattices[0])] = slab_hopping_z
    return syst.finalized()


DEFAULT_PARAMS = {
    "L_z": 50,
    "L_y": 300.0,
    "C0": 0.1,
    "C2": 0.0,
    "C1": 0.0,
    "M0": -0.28,
    "G": -0.46,
    "M1": 10.0,
    "M2": 56.59,
    "A": 4.1,
    "B": 0.0,
    "T": 0.45,
    "Delta": 0.1,
    "a": 8.0,
    "c": 2.0,
    "ky": 0.0,
    "kz": 0.0,
    "kx": 0.0,
    "W_func": disorder_potential,
    "Wa": 0.2,
    "salt": "seed-0",
}


def compute_kx_eigensystem(
    syst: kwant.system.FiniteSystem,
    params: Dict,
    kx_values: np.ndarray,
    bands: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    energies = []
    wavefuncs = []
    for ki in kx_values:
        params["kx"] = float(ki)
        H_k = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        if bands >= H_k.shape[0]:
            raise ValueError(f"bands={bands} incompatible with Hilbert space {H_k.shape[0]}")
        ens, wfs = sla.eigsh(H_k, k=bands, sigma=sigma, return_eigenvectors=True)
        ens, wfs = sorted_eigs((ens, wfs))
        energies.append(ens)
        wavefuncs.append(wfs)
    energies = np.stack(energies, axis=0)
    wavefuncs = np.stack(wavefuncs, axis=0)
    return energies, wavefuncs


def ldos_from_states(
    wavefuncs: np.ndarray,
    energies: np.ndarray,
    ldos_op: kwant.operator.Density,
    eps: float,
    ly_sites: int,
    lz_sites: int,
) -> np.ndarray:
    ldos_flat = np.zeros(ly_sites * lz_sites, dtype=float)
    for kk in range(energies.shape[0]):
        sel = np.where(np.abs(energies[kk]) <= eps)[0]
        if sel.size == 0:
            continue
        for bb in sel:
            psi = wavefuncs[kk, :, bb]
            ldos_flat += ldos_op(psi).real
    return ldos_flat.reshape((ly_sites, lz_sites)).T


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Wa disorder calculations")
    parser.add_argument(
        "--wa-list",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 1.0, 3.0, 5.0],
        help="disorder strengths to evaluate",
    )
    parser.add_argument("--repetitions", type=int, default=30, help="realizations per disorder strength")
    parser.add_argument("--bands", type=int, default=20, help="eigenvalues nearest zero")
    parser.add_argument("--n-k", type=int, default=31, help="number of kx points")
    parser.add_argument("--kmin", type=float, default=-0.05, help="minimum kx")
    parser.add_argument("--kmax", type=float, default=0.05, help="maximum kx")
    parser.add_argument("--eps", type=float, default=0.05, help="LDOS window |E| <= eps")
    parser.add_argument("--sigma", type=float, default=1e-5, help="spectral shift for eigsh")
    parser.add_argument("--seed-base", type=int, default=1000, help="first disorder seed")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("calcs_noise.npz"),
        help="output file",
    )
    parser.add_argument("--Ly", type=float, default=None, help="Ly in Angstrom")
    parser.add_argument("--Lz", type=float, default=None, help="Lz in Angstrom")
    parser.add_argument("--a", type=float, default=None, help="in-plane lattice constant in Angstrom")
    parser.add_argument("--c", type=float, default=None, help="vertical lattice constant in Angstrom")
    parser.add_argument("--store-system", action="store_true", help="store the serialized Kwant system")
    return parser.parse_args(argv)


def override_params(base: Dict, args: argparse.Namespace) -> Dict:
    params = dict(base)
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
    params = override_params(DEFAULT_PARAMS, args)
    params["Wa"] = float(args.wa_list[0])
    params.setdefault("salt", "seed-0")

    syst = build_disordered_slab(params)
    op_mat = np.diag(np.ones(8, dtype=float))
    ldos_op = kwant.operator.Density(syst, op_mat)

    ly_sites = int(round(params["L_y"] / params["a"]))
    lz_sites = int(round(params["L_z"] / params["c"]))
    if ly_sites * lz_sites != syst.graph.num_nodes:
        raise RuntimeError("Grid size mismatch with final system")

    kx_values = np.linspace(args.kmin, args.kmax, args.n_k)
    seeds = args.seed_base + np.arange(args.repetitions)

    results: Dict[str, np.ndarray] = {}
    metadata = {
        "kx_values": kx_values,
        "wa_list": np.array(args.wa_list, dtype=float),
        "seeds": seeds,
        "eps": args.eps,
        "bands": args.bands,
        "n_k": args.n_k,
        "Ly_sites": ly_sites,
        "Lz_sites": lz_sites,
        "a_ang": params["a"],
        "c_ang": params["c"],
    }

    system_blob = None
    if args.store_system:
        system_blob = np.frombuffer(
            pickle.dumps(syst, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8
        ).copy()

    start = datetime.now()
    for wa in args.wa_list:
        ens_acc = []
        wfs_acc = []
        ldos_acc = []
        loop_params = dict(params)
        loop_params["Wa"] = float(wa)
        pbar = tqdm(seeds, unit="real", desc=f"Wa={wa:g}")
        for seed in pbar:
            loop_params["salt"] = f"seed-{int(seed)}"
            energies, wavefuncs = compute_kx_eigensystem(
                syst,
                dict(loop_params),
                kx_values,
                args.bands,
                args.sigma,
            )
            ldos_map = ldos_from_states(wavefuncs, energies, ldos_op, args.eps, ly_sites, lz_sites)
            ens_acc.append(energies)
            wfs_acc.append(wavefuncs)
            ldos_acc.append(ldos_map)

        wa_label = f"{wa:g}"
        results[f"ens_{wa_label}"] = np.array(ens_acc)
        results[f"wfs_{wa_label}"] = np.array(wfs_acc)
        results[f"ldos_maps_{wa_label}"] = np.array(ldos_acc)
        if system_blob is not None:
            results[f"system_blob_{wa_label}"] = system_blob.copy()

    results.update(metadata)
    np.savez(args.out, **results)

    elapsed = datetime.now() - start
    print(f"Wrote {args.out} with keys: {list(results.keys())}")
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
