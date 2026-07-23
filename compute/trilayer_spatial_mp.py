"""Per-site complex Majorana polarization MP(y,z)=2 sum_a u_a v_a (= psi^T C psi)
and density rho(y,z) for the S-I-S trilayer low-energy states, at a few phi.

Saves MP (complex) and rho grids for the lowest nstates, plus the regional sums
|sum_L MP|, |sum_R MP| and chi_L/chi_R, to verify whether the MP cancels within
each edge region.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import scipy.sparse.linalg as sla

from josephson_model import DEFAULT_PARAMS
from trilayer_spectrum import build_trilayer
from josephson_pair_scan import c_per_site


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Spatial Majorana polarization in the S-I-S trilayer"
    )
    p.add_argument("--Ly", type=float, default=300.0)
    p.add_argument("--Lz-bot", type=float, default=20.0)
    p.add_argument("--Lz-mid", type=float, default=6.0)
    p.add_argument("--Lz-top", type=float, default=20.0)
    p.add_argument("--M0-bulk", type=float, default=-0.28)
    p.add_argument("--M0-mid", type=float, default=0.28)
    p.add_argument("--Delta", type=float, default=0.1)
    p.add_argument("--phis", type=str, default="0,1.5707963267948966,3.141592653589793")
    p.add_argument("--phi-top", type=float, default=0.0)
    p.add_argument("--nstates", type=int, default=4)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("trilayer_mp.npz"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = dict(DEFAULT_PARAMS)
    for k in ("M0", "L_z", "Delta_u", "Delta_d", "delta_profile", "xi_delta", "Delta_func"):
        params.pop(k, None)
    params["L_y"] = float(args.Ly)
    params["Lz_bot"] = float(args.Lz_bot)
    params["Lz_mid"] = float(args.Lz_mid)
    params["Lz_top"] = float(args.Lz_top)
    params["M0_bulk"] = float(args.M0_bulk)
    params["M0_mid"] = float(args.M0_mid)
    params["Delta"] = float(args.Delta)
    params["G_mid"] = float(params["G"])
    params["T_mid"] = float(params["T"])
    params["kx"] = 0.0
    params["phi_top"] = float(args.phi_top)

    syst = build_trilayer(params)
    norb = 8
    sites = list(syst.sites)
    y_index = np.array([site.tag[0] for site in sites], dtype=int)
    z_index = np.array([site.tag[1] for site in sites], dtype=int)
    ny = int(y_index.max()) + 1
    nz = int(z_index.max()) + 1
    a, c = params["a"], params["c"]
    particle_hole = c_per_site(norb)
    mask_left = y_index < (ny / 2.0)
    mask_right = ~mask_left

    phis = np.array([float(x) for x in args.phis.split(",")])
    n_phases = len(phis)
    n_states = args.nstates
    density_grid = np.zeros((n_phases, n_states, ny, nz))
    mp_grid = np.zeros((n_phases, n_states, ny, nz), dtype=complex)
    energy_matrix = np.zeros((n_phases, n_states))
    chi_left = np.zeros((n_phases, n_states))
    chi_right = np.zeros((n_phases, n_states))

    for ip, phi in enumerate(phis):
        params["phi_rel"] = float(phi)
        hamiltonian = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        energies, vectors = sla.eigsh(
            hamiltonian,
            k=n_states,
            sigma=0.0,
            return_eigenvectors=True,
        )
        order = np.argsort(np.abs(energies))
        energies = energies[order]
        vectors = vectors[:, order]
        for n in range(n_states):
            psi = vectors[:, n].reshape(-1, norb)
            density = np.sum(np.abs(psi) ** 2, axis=1)
            mp_site = np.einsum(
                "si,ij,sj->s",
                psi,
                particle_hole,
                psi,
                optimize=True,
            )
            state_density = np.zeros((ny, nz))
            state_mp = np.zeros((ny, nz), dtype=complex)
            state_density[y_index, z_index] = density
            state_mp[y_index, z_index] = mp_site
            density_grid[ip, n] = state_density
            mp_grid[ip, n] = state_mp
            energy_matrix[ip, n] = energies[n]
            for mask, store in (
                (mask_left, chi_left),
                (mask_right, chi_right),
            ):
                msum = np.sum(mp_site[mask])
                rsum = np.sum(density[mask])
                store[ip, n] = abs(msum) ** 2 / (rsum ** 2 + 1e-300)
            left_sum = abs(np.sum(mp_site[mask_left]))
            right_sum = abs(np.sum(mp_site[mask_right]))
            left_abs_sum = np.sum(np.abs(mp_site[mask_left]))
            right_abs_sum = np.sum(np.abs(mp_site[mask_right]))
            print(
                f"phi/pi={phi / np.pi:.2f} n={n} E={energies[n] * 1e3:+.3f}meV "
                f"chiL={chi_left[ip, n]:.3f} chiR={chi_right[ip, n]:.3f} | "
                f"|sumMP_L|/sum|MP|_L={left_sum / (left_abs_sum + 1e-300):.3f} "
                f"|sumMP_R|/sum|MP|_R={right_sum / (right_abs_sum + 1e-300):.3f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        phis=phis,
        rho=density_grid,
        MP_re=mp_grid.real,
        MP_im=mp_grid.imag,
        E=energy_matrix,
        chiL=chi_left,
        chiR=chi_right,
        ny=ny,
        nz=nz,
        a=a,
        c=c,
        Ly=params["L_y"],
        Lz_bot=params["Lz_bot"],
        Lz_mid=params["Lz_mid"],
        Lz_top=params["Lz_top"],
        M0_mid=params["M0_mid"],
    )
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
