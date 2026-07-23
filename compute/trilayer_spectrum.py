"""S-I-S vertical Josephson junction with an insulating interlayer, at kx=0.

Three regions stacked along z:
  bottom  [0, Lz_bot)              : Delta e^{i phi}  (variable phase), M0 = M0_bulk
  middle  [Lz_bot, Lz_bot+Lz_mid)  : Delta = 0 (insulator), M0 = M0_mid (non-inverted)
  top     [..., Ltot)              : Delta e^{i phi_top=0}, M0 = M0_bulk

Only the onsite Dirac mass M0 and the pairing Delta are made z-dependent; the
gamma matrices and the y/z hoppings are reused from josephson_model.py.
By default G (exchange) and T are kept in every layer; set --G-mid/--T-mid to
change them in the interlayer.

For each phi it diagonalizes at kx=0 and stores the low-energy spectrum coloured by
the Majorana indicator max(chi_L,chi_R) (left/right halves in y), plus the gap and
the Majorana-pair count. The output keys match josephson_pair_scan.py.
"""

from __future__ import annotations

import argparse
import pathlib

import kwant
import numpy as np
import scipy.sparse.linalg as sla

from josephson_model import (
    DEFAULT_PARAMS,
    G00X,
    GXYZ,
    GYYZ,
    GZ00,
    GZ0Z,
    GZYY,
    GZZZ,
    hopping_y,
    hopping_z,
)
from josephson_pair_scan import c_per_site, site_y_index, regional_chi


def trilayer_onsite(
    site,
    kx,
    a,
    c,
    C0,
    C1,
    C2,
    M0_bulk,
    M0_mid,
    M1,
    M2,
    A,
    G,
    G_mid,
    Delta,
    phi_rel,
    phi_top,
    T,
    T_mid,
    B,
    Lz_bot,
    Lz_mid,
    Lz_top,
):
    z_phys = int(site.tag[1]) * c + c / 2.0
    if z_phys < Lz_bot:
        M0_h, dval, G_h, T_h = M0_bulk, Delta * np.exp(1j * phi_rel), G, T
    elif z_phys < Lz_bot + Lz_mid:
        M0_h, dval, G_h, T_h = M0_mid, 0.0 + 0.0j, G_mid, T_mid
    else:
        M0_h, dval, G_h, T_h = M0_bulk, Delta * np.exp(1j * phi_top), G, T
    ck2_xy = 2.0 - np.cos(kx * a)
    ck2_z = 1
    epsilon_0 = C0 + (2 * C2 / a**2) * ck2_xy + (2 * C1 / c**2) * ck2_z
    M_0 = M0_h + (2 * M2 / a**2) * ck2_xy + (2 * M1 / c**2) * ck2_z
    return (
        epsilon_0 * GZ00
        + (A / a) * (np.sin(kx * a) * G00X)
        + M_0 * GZ0Z
        + G_h * GZZZ
        + np.real(dval) * GYYZ
        + np.imag(dval) * GXYZ
        + T_h * GZYY
    )


def build_trilayer(params):
    a, c = params["a"], params["c"]
    L_y = params["L_y"]
    total_height = params["Lz_bot"] + params["Lz_mid"] + params["Lz_top"]
    lat = kwant.lattice.general([(a, 0), (0, c)], basis=[(0, 0)], norbs=8)
    syst = kwant.Builder()

    def shape(pos):
        y, z = pos
        return (0 <= y < L_y) and (0 <= z < total_height)

    syst[lat.shape(shape, (0, 0))] = trilayer_onsite
    syst[kwant.builder.HoppingKind((1, 0), lat.sublattices[0], lat.sublattices[0])] = hopping_y
    syst[kwant.builder.HoppingKind((0, 1), lat.sublattices[0], lat.sublattices[0])] = hopping_z
    return syst.finalized()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S-I-S trilayer JJ spectrum vs phi at kx=0")
    p.add_argument("--Ly", type=float, default=300.0)
    p.add_argument("--Lz-bot", type=float, default=20.0)
    p.add_argument("--Lz-mid", type=float, default=6.0)
    p.add_argument("--Lz-top", type=float, default=20.0)
    p.add_argument("--M0-bulk", type=float, default=-0.28)
    p.add_argument("--M0-mid", type=float, default=0.28, help="interlayer mass (non-inverted)")
    p.add_argument("--Delta", type=float, default=0.1)
    p.add_argument("--G-mid", type=float, default=None, help="exchange in interlayer (default = bulk G)")
    p.add_argument("--T-mid", type=float, default=None, help="T in interlayer (default = bulk T)")
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--phi-top", type=float, default=0.0)
    p.add_argument("--phi-min", type=float, default=0.0)
    p.add_argument("--phi-max", type=float, default=2.0 * np.pi)
    p.add_argument("--n-phi", type=int, default=121)
    p.add_argument("--bands", type=int, default=24)
    p.add_argument("--chi-thr", type=float, default=0.5)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("trilayer.npz"))
    p.add_argument("--progress", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = dict(DEFAULT_PARAMS)
    params.pop("M0", None)
    params.pop("L_z", None)
    params["L_y"] = float(args.Ly)
    params["Lz_bot"] = float(args.Lz_bot)
    params["Lz_mid"] = float(args.Lz_mid)
    params["Lz_top"] = float(args.Lz_top)
    params["M0_bulk"] = float(args.M0_bulk)
    params["M0_mid"] = float(args.M0_mid)
    params["Delta"] = float(args.Delta)
    params["G_mid"] = float(args.G_mid) if args.G_mid is not None else float(params["G"])
    params["T_mid"] = float(args.T_mid) if args.T_mid is not None else float(params["T"])
    params["kx"] = float(args.kx)
    params["phi_top"] = float(args.phi_top)
    # Kwant rejects parameters that are unused by the onsite and hopping callables.
    for k in ("Delta_u", "Delta_d", "delta_profile", "xi_delta", "Delta_func"):
        params.pop(k, None)

    syst = build_trilayer(params)
    norb = 8
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1
    mask_L = ysite < (ly / 2.0)
    mask_R = ~mask_L
    cmat = c_per_site(norb)

    phis = np.linspace(args.phi_min, args.phi_max, args.n_phi)
    gap_min = np.zeros(args.n_phi)
    maxchi0 = np.zeros(args.n_phi)
    npairs = np.zeros(args.n_phi)
    energy_matrix = np.zeros((args.bands, args.n_phi))
    majorana_matrix = np.zeros((args.bands, args.n_phi))

    for i, phi in enumerate(phis):
        params["phi_rel"] = float(phi)
        hamiltonian = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        energies, vectors = sla.eigsh(
            hamiltonian,
            k=args.bands,
            sigma=0.0,
            return_eigenvectors=True,
        )
        order = np.argsort(np.abs(energies))
        energies = energies[order]
        vectors = vectors[:, order]
        majorana_score = np.array(
            [
                max(
                    regional_chi(vectors[:, j], mask_L, cmat, norb),
                    regional_chi(vectors[:, j], mask_R, cmat, norb),
                )
                for j in range(vectors.shape[1])
            ]
        )
        energy_matrix[:, i] = energies
        majorana_matrix[:, i] = majorana_score
        gap_min[i] = float(np.min(np.abs(energies)))
        maxchi0[i] = float(majorana_score[int(np.argmin(np.abs(energies)))])
        npairs[i] = 0.5 * int(np.sum(majorana_score > args.chi_thr))
        msg = (
            f"phi {i + 1}/{args.n_phi}={phi:.4f} gap={gap_min[i]:.3e} "
            f"maxchi0={maxchi0[i]:.3f} npairs={npairs[i]:.1f}"
        )
        if args.progress is not None:
            with open(args.progress, "w") as fh:
                fh.write(msg + "\n")
        if i % 10 == 0 or i == args.n_phi - 1:
            print("[" + msg + "]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        Lz=float(args.Lz_bot + args.Lz_mid + args.Lz_top),
        Lz_bot=float(args.Lz_bot),
        Lz_mid=float(args.Lz_mid),
        Lz_top=float(args.Lz_top),
        M0_bulk=float(args.M0_bulk),
        M0_mid=float(args.M0_mid),
        Delta=float(args.Delta),
        Ly=float(args.Ly),
        kx=float(args.kx),
        chi_thr=float(args.chi_thr),
        bands=int(args.bands),
        phi=phis,
        gap_min=gap_min,
        maxchi0=maxchi0,
        npairs=npairs,
        emat=energy_matrix,
        maxchi_mat=majorana_matrix,
    )
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
