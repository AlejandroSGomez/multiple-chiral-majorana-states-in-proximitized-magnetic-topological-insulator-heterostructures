"""Josephson-phase sweep for a BHZ+SC slab open in y and z.

For each phase, the script stores the spectrum, eigenvectors, and LDOS
integrated over the selected energy window and momenta near kx=0.
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
from datetime import datetime
from datetime import timedelta
from typing import Dict, Iterable, Tuple

import kwant
import numpy as np
import scipy.sparse.linalg as sla

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
s0 = np.eye(2, dtype=complex)


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

def sorted_eigs(ev: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = ev
    order = np.argsort(evals)
    return np.array(evals[order]), np.array(evecs[:, order])


def delta_func(
    z_index: int,
    L_z: float,
    c: float,
    Delta_u: float,
    Delta_d: float,
    phi_rel: float,
    phi_top: float,
    delta_profile: str = "step",
    xi_delta: float | None = None,
) -> complex:
    """Return a step profile or a normalized exponential surface mixture."""
    z_phys = z_index * c + c / 2
    if delta_profile == "exp":
        xi = c if (xi_delta is None or xi_delta <= 0) else xi_delta
        w_up = np.exp(-(L_z - z_phys) / xi)
        w_down = np.exp(-z_phys / xi)
        norm = w_up + w_down
        if norm <= 0:
            return 0.0
        w_up /= norm
        w_down /= norm
        return (
            w_up * Delta_u * np.exp(1j * phi_top)
            + w_down * Delta_d * np.exp(1j * (phi_top + phi_rel))
        )
    if z_phys >= L_z / 2:
        return Delta_u * np.exp(1j * phi_top)
    return Delta_d * np.exp(1j * (phi_top + phi_rel))


def onsite_kx(
    site,
    kx,
    a,
    c,
    C0,
    C1,
    C2,
    M0,
    M1,
    M2,
    A,
    G,
    Delta_func,
    Delta_u,
    Delta_d,
    phi_rel,
    phi_top,
    delta_profile,
    xi_delta,
    T,
    B,
    L_z,
):
    z_index = int(site.tag[1])
    ck2_xy = 2.0 - np.cos(kx * a)
    ck2_z = 1
    epsilon_0 = C0 + (2 * C2 / (a**2)) * ck2_xy + (2 * C1 / (c**2)) * ck2_z
    M_0 = M0 + (2 * M2 / (a**2)) * ck2_xy + (2 * M1 / (c**2)) * ck2_z
    Delta_val = Delta_func(
        z_index, L_z, c, Delta_u, Delta_d, phi_rel, phi_top, delta_profile, xi_delta
    )
    return (
        epsilon_0 * GZ00
        + (A / a) * (np.sin(kx * a) * G00X)
        + M_0 * GZ0Z
        + G * GZZZ
        + np.real(Delta_val) * GYYZ
        + np.imag(Delta_val) * GXYZ
        + T * GZYY
    )


def hopping_z(site1, site2, c, C1, M1, B, T):
    return (C1 / c**2) * GZ00 + (M1 / c**2) * GZ0Z + (1j * B / (2 * c) * G0YY)


def hopping_y(site1, site2, a, C2, M2, A):
    return (C2 / a**2) * GZ00 + (M2 / a**2) * GZ0Z + (1j * A) / (2 * a) * GZZY


def build_josephson_slab(params: Dict) -> kwant.system.FiniteSystem:
    a, c = params["a"], params["c"]
    L_y, L_z = params["L_y"], params["L_z"]

    lat = kwant.lattice.general([(a, 0), (0, c)], basis=[(0, 0)], norbs=8)
    syst = kwant.Builder()

    def shape(pos):
        y, z = pos
        return (0 <= y < L_y) and (0 <= z < L_z)

    syst[lat.shape(shape, (0, 0))] = onsite_kx
    syst[kwant.builder.HoppingKind((1, 0), lat.sublattices[0], lat.sublattices[0])] = hopping_y
    syst[kwant.builder.HoppingKind((0, 1), lat.sublattices[0], lat.sublattices[0])] = hopping_z
    return syst.finalized()


DEFAULT_PARAMS = {
    "L_z": 50.0,
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
    "Delta_u": 0.1,
    "Delta_d": 0.1,
    "phi_rel": 0.0,
    "phi_top": 0.0,
    "delta_profile": "step",
    "xi_delta": 5.0,
    "a": 4.0,
    "c": 2.0,
    "kx": 0.0,
    "Delta_func": delta_func,
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
    energies = np.array(energies)
    wavefuncs = np.array(wavefuncs)
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


def site_indexer(syst, params):
    coords = np.array([site.pos for site in syst.sites])
    a, c = params["a"], params["c"]
    y_coords = np.rint(coords[:, 0] / a).astype(int)
    z_coords = np.rint(coords[:, 1] / c).astype(int)

    y_vals, z_vals = np.unique(y_coords), np.unique(z_coords)
    ly_sites, lz_sites = len(y_vals), len(z_vals)

    linear_index = np.searchsorted(y_vals, y_coords) + ly_sites * np.searchsorted(z_vals, z_coords)
    return linear_index, (ly_sites, lz_sites)


def ldos_flat_to_grid(ldos_flat, linear_index, grid_shape):
    grid = np.zeros(np.prod(grid_shape), dtype=ldos_flat.dtype)
    grid[linear_index] = ldos_flat
    return grid.reshape(grid_shape)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Josephson-phase sweep at conserved kx")
    parser.add_argument("--phi-min", type=float, default=0.0,
                        help="minimum relative phase in radians")
    parser.add_argument("--phi-max", type=float, default=np.pi,
                        help="maximum relative phase in radians")
    parser.add_argument("--n-phi", type=int, default=3, help="number of phase values")
    parser.add_argument("--phi", type=float, default=None,
                        help="single phase; overrides the sweep")
    parser.add_argument("--phi-top", type=float, default=0.0, help="fixed top phase")
    parser.add_argument("--bands", type=int, default=20, help="eigenvalues nearest zero")
    parser.add_argument("--n-k", type=int, default=31, help="number of kx points")
    parser.add_argument("--kmin", type=float, default=-0.05, help="minimum kx")
    parser.add_argument("--kmax", type=float, default=0.05, help="maximum kx")
    parser.add_argument("--eps", type=float, default=0.05, help="LDOS window |E| <= eps")
    parser.add_argument("--sigma", type=float, default=0, help="eigsh shift")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("calcs_phase.npz"),
                        help="output stem; one file is written per phase")
    parser.add_argument("--progress", type=pathlib.Path, default=None,
                        help="optional progress file")
    parser.add_argument("--Ly", type=float, default=None, help="Ly in Angstrom")
    parser.add_argument("--Lz", type=float, default=None, help="Lz in Angstrom")
    parser.add_argument("--a", type=float, default=None, help="in-plane lattice constant")
    parser.add_argument("--c", type=float, default=None, help="vertical lattice constant")
    parser.add_argument("--Delta-u", dest="Delta_u", type=float, default=None, help="top gap magnitude")
    parser.add_argument("--Delta-d", dest="Delta_d", type=float, default=None, help="bottom gap magnitude")
    parser.add_argument("--delta-profile", choices=["step", "exp"], default="step",
                        help="step profile or normalized exponential mixture")
    parser.add_argument("--xi-delta", dest="xi_delta", type=float, default=None,
                        help="decay length for the exponential profile")
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
    if args.Delta_u is not None:
        params["Delta_u"] = float(args.Delta_u)
    if args.Delta_d is not None:
        params["Delta_d"] = float(args.Delta_d)
    params["delta_profile"] = args.delta_profile
    if args.xi_delta is not None:
        params["xi_delta"] = float(args.xi_delta)
    params["phi_top"] = float(args.phi_top)
    return params


def format_timedelta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    return str(timedelta(seconds=int(round(seconds))))


def write_progress(path, *, status, current_index, total, phi_val, start_time, current_file=None):
    if path is None:
        return
    now = datetime.now()
    completed = max(0, int(current_index))
    total = max(1, int(total))
    elapsed_s = (now - start_time).total_seconds()
    avg_s = elapsed_s / completed if completed > 0 else np.nan
    remaining = max(0, total - completed)
    eta_s = avg_s * remaining if completed > 0 else np.nan
    percent = 100.0 * completed / total
    lines = [
        f"status: {status}",
        f"updated: {now.isoformat(timespec='seconds')}",
        f"completed: {completed}/{total}",
        f"percent: {percent:.2f}",
        f"current_phi: {phi_val:.12g}" if phi_val is not None else "current_phi: none",
        f"elapsed: {format_timedelta(elapsed_s)}",
        f"avg_per_phi: {format_timedelta(avg_s)}",
        f"eta: {format_timedelta(eta_s)}",
    ]
    if current_file is not None:
        lines.append(f"last_file: {current_file}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args(argv)
    if args.phi is not None:
        phi_values = [float(args.phi)]
    else:
        phi_values = list(np.linspace(args.phi_min, args.phi_max, args.n_phi))
    params = override_params(DEFAULT_PARAMS, args)

    kx_values = np.linspace(args.kmin, args.kmax, args.n_k)

    print(f"Building system with Ly={params['L_y']}, Lz={params['L_z']}...")
    params["phi_rel"] = 0.0
    syst = build_josephson_slab(params)

    op_mat = np.diag(np.ones(8, dtype=float))
    ldos_op = kwant.operator.Density(syst, op_mat)
    linear_index, grid_shape = site_indexer(syst, params)
    n_sites = syst.graph.num_nodes

    base_metadata = {
        "eps": args.eps,
        "bands": args.bands,
        "n_k": args.n_k,
        "kmin": args.kmin,
        "kmax": args.kmax,
        "kx_values": kx_values,
        "sigma": args.sigma,
        "grid_shape": np.array(grid_shape, dtype=int),
        "Ly_ang": params["L_y"],
        "Lz_ang": params["L_z"],
        "a_ang": params["a"],
        "c_ang": params["c"],
        "Delta_u": params["Delta_u"],
        "Delta_d": params["Delta_d"],
        "phi_top": params["phi_top"],
        "delta_profile": params["delta_profile"],
        "xi_delta": params["xi_delta"],
    }

    start_total = datetime.now()

    out_path = args.out
    out_stem = out_path.stem
    out_suffix = out_path.suffix
    out_parent = out_path.parent

    print(f"Starting phase sweep: {phi_values}")
    write_progress(
        args.progress,
        status="starting",
        current_index=0,
        total=len(phi_values),
        phi_val=phi_values[0] if phi_values else None,
        start_time=start_total,
    )

    for iphi, phi_val in enumerate(phi_values, start=1):
        print(f"\nProcessing phi={phi_val} rad")
        write_progress(
            args.progress,
            status="running",
            current_index=iphi - 1,
            total=len(phi_values),
            phi_val=phi_val,
            start_time=start_total,
        )
        loop_start = datetime.now()

        loop_params = dict(params)
        loop_params["phi_rel"] = float(phi_val)

        energies, wavefuncs = compute_kx_eigensystem(
            syst, dict(loop_params), kx_values, args.bands, args.sigma
        )
        ldos_map = ldos_from_states(wavefuncs, energies, ldos_op, args.eps, *grid_shape)

        phi_results = {
            "phi_rel": float(phi_val),
            "kx_values": kx_values,
            "ens": energies,
            "wfs": wavefuncs,
            "ldos_map": ldos_map,
        }
        phi_results.update(base_metadata)

        phi_label = f"{phi_val:g}"
        filename = out_parent / f"{out_stem}_Phi{phi_label}{out_suffix}"

        print(f"Writing {filename}")
        np.savez(filename, **phi_results)

        elapsed = datetime.now() - loop_start
        print(f"Finished phi={phi_val} in {elapsed}")
        write_progress(
            args.progress,
            status="running",
            current_index=iphi,
            total=len(phi_values),
            phi_val=phi_val,
            start_time=start_total,
            current_file=filename,
        )

        del energies, wavefuncs, ldos_map, phi_results
        gc.collect()

    total_elapsed = datetime.now() - start_total
    print(f"\nSweep completed in {total_elapsed}")
    write_progress(
        args.progress,
        status="done",
        current_index=len(phi_values),
        total=len(phi_values),
        phi_val=phi_values[-1] if phi_values else None,
        start_time=start_total,
    )


if __name__ == "__main__":
    main()
