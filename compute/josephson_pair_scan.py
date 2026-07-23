"""Majorana-character map of the vertical JJ at kx=0, vs (Lz, phi).

Recomputes the information of Fig. 5(b) but at the particle-hole-invariant
momentum kx=0 (the natural cut for the zero-crossing / parity story).

For each Josephson phase phi (one Lz per run; use an array over Lz) it
diagonalizes the kx=0 BdG slab and, from the lowest `bands` states, computes the
regional Majorana polarization chi_{L/R} = |sum_region MP|^2 / (sum_region rho)^2
with MP = psi^T C psi (= 2 sum_a u_a v_a) over the left/right halves in y.

Saved per phi:
  - gap_min  : min_n |E_n|  (the near-zero gap; its zeros are the parity flips)
  - maxchi0  : max(chi_L,chi_R) of the lowest-|E| state  (Majorana character)
  - npairs   : half the number of the `bands` states satisfying both
               max(chi_L,chi_R) > chi_thr and |<Q>| < charge_thr
  - emat,maxchi_mat,charge_mat : E, max(chi_L,chi_R), and <Q>/e per state
                                 (bands x nphi)

chi_{L/R} are computed directly from the eigenvectors and the site y-index, so the
result is independent of kwant's internal site ordering.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import numpy as np
import scipy.sparse.linalg as sla

# The local macOS MUMPS extension can abort while kwant is imported even though
# this calculation uses scipy.sparse.linalg directly.  Keep the workaround
# opt-in so cluster environments with a working MUMPS installation are
# unaffected.
if os.environ.get("KWANT_DISABLE_MUMPS") == "1":
    sys.modules["mumps"] = None

from josephson_model import build_josephson_slab, DEFAULT_PARAMS, delta_func

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
CHARGE_DIAG = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)


def c_per_site(norb):
    return np.kron(SX, np.eye(norb // 2, dtype=complex))


def site_y_index(syst):
    """y lattice index of each site, in the order used by hamiltonian_submatrix."""
    return np.array([s.tag[0] for s in syst.sites], dtype=int)


def region_masks(syst, mode):
    """Return the two regional masks used by a given published calculation."""
    if mode == "site-y":
        ysite = site_y_index(syst)
        ly = int(ysite.max()) + 1
        mask_L = ysite < (ly / 2.0)
        return mask_L, ~mask_L
    if mode == "notebook-order":
        # Preserve the ordering convention used for the published Fig. 6(b).
        ly = len({int(site.tag[0]) for site in syst.sites})
        lz = len({int(site.tag[1]) for site in syst.sites})
        raw_to_plot = np.array(
            [y + ly * z for y in range(ly) for z in range(lz)],
            dtype=int,
        )
        if raw_to_plot.size != len(syst.sites):
            raise ValueError(
                "Notebook-order mask is incompatible with the system site count"
            )
        split = (ly // 2) * lz
        mask_L = raw_to_plot < split
        return mask_L, ~mask_L
    raise ValueError(f"Unknown region-mask mode: {mode}")


def regional_chi(psi, mask, cmat, norb):
    ps = psi.reshape(-1, norb)[mask]
    mp = np.einsum("si,ij,sj->", ps, cmat, ps, optimize=True)  # bilinear (no conj): sum psi^T C psi
    rho = float(np.sum(np.abs(ps) ** 2).real)
    if rho < 1e-300:
        return 0.0
    return float(abs(mp) ** 2 / rho ** 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Majorana-character (Lz,phi) map at kx=0")
    p.add_argument("--Ly", type=float, default=300.0)
    p.add_argument("--Lz", type=float, default=20.0)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--phi-top", type=float, default=0.0)
    p.add_argument("--phi-min", type=float, default=0.0)
    p.add_argument("--phi-max", type=float, default=2.0 * np.pi)
    p.add_argument("--n-phi", type=int, default=121)
    p.add_argument("--bands", type=int, default=20, help="states nearest 0 (as in Fig.5b)")
    p.add_argument("--chi-thr", type=float, default=0.5, help="Majorana selection threshold")
    p.add_argument("--charge-thr", type=float, default=0.1,
                   help="absolute charge threshold |<Q>|/e (same as Fig. A2)")
    p.add_argument("--seed", type=int, default=20260723,
                   help="seed for the deterministic ARPACK starting vector")
    p.add_argument(
        "--region-mask",
        choices=("site-y", "notebook-order"),
        default="site-y",
        help="regional mask convention; notebook-order reproduces Fig. 6(b)",
    )
    p.add_argument(
        "--pair-rounding",
        choices=("half", "floor"),
        default="half",
        help="half keeps Nselected/2; floor reproduces the historical Fig. 6(b) count",
    )
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("chimap.npz"))
    p.add_argument("--progress", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = dict(DEFAULT_PARAMS)
    params["Delta_func"] = delta_func
    params["L_y"] = float(args.Ly)
    params["L_z"] = float(args.Lz)
    params["kx"] = float(args.kx)
    params["phi_top"] = float(args.phi_top)

    syst = build_josephson_slab(params)
    norb = 8
    mask_L, mask_R = region_masks(syst, args.region_mask)
    cmat = c_per_site(norb)

    phis = np.linspace(args.phi_min, args.phi_max, args.n_phi)
    gap_min = np.zeros(args.n_phi)
    maxchi0 = np.zeros(args.n_phi)
    npairs = np.zeros(args.n_phi)
    nselected = np.zeros(args.n_phi, dtype=int)
    emat = np.zeros((args.bands, args.n_phi))
    maxchi_mat = np.zeros((args.bands, args.n_phi))
    charge_mat = np.zeros((args.bands, args.n_phi))
    rng = np.random.default_rng(args.seed)
    v0 = None

    for i, phi in enumerate(phis):
        # Make the two physically identical endpoints numerically identical.
        phi_eval = 0.0 if np.isclose(phi, 2.0 * np.pi, rtol=0.0, atol=1e-12) else float(phi)
        params["phi_rel"] = phi_eval
        H = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        if v0 is None:
            v0 = rng.standard_normal(H.shape[0]) + 1j * rng.standard_normal(H.shape[0])
            v0 /= np.linalg.norm(v0)
        E, V = sla.eigsh(
            H,
            k=args.bands,
            sigma=0.0,
            v0=v0.copy(),
            return_eigenvectors=True,
        )
        order = np.argsort(np.abs(E))
        E = E[order]
        V = V[:, order]
        mxchi = np.array([
            max(regional_chi(V[:, j], mask_L, cmat, norb),
                regional_chi(V[:, j], mask_R, cmat, norb))
            for j in range(V.shape[1])
        ])
        vecs = V.reshape(-1, norb, V.shape[1])
        charge = np.einsum(
            "sin,i->n", np.abs(vecs) ** 2, CHARGE_DIAG, optimize=True
        ).real
        selected = (mxchi > args.chi_thr) & (np.abs(charge) < args.charge_thr)
        emat[:, i] = E
        maxchi_mat[:, i] = mxchi
        charge_mat[:, i] = charge
        gap_min[i] = float(np.min(np.abs(E)))
        maxchi0[i] = float(mxchi[int(np.argmin(np.abs(E)))])
        nselected[i] = int(np.count_nonzero(selected))
        if args.pair_rounding == "floor":
            npairs[i] = nselected[i] // 2
        else:
            npairs[i] = 0.5 * nselected[i]
        msg = (f"phi {i+1}/{args.n_phi}={phi:.4f} gap={gap_min[i]:.3e} "
               f"maxchi0={maxchi0[i]:.3f} npairs={npairs[i]:.1f}")
        if args.progress is not None:
            with open(args.progress, "w") as fh:
                fh.write(msg + "\n")
        if i % 10 == 0 or i == args.n_phi - 1:
            print("[" + msg + "]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, Lz=float(args.Lz), Ly=float(args.Ly), kx=float(args.kx),
        chi_thr=float(args.chi_thr), charge_thr=float(args.charge_thr),
        bands=int(args.bands), seed=int(args.seed), region_mask=str(args.region_mask),
        pair_rounding=str(args.pair_rounding),
        phi=phis, gap_min=gap_min, maxchi0=maxchi0, npairs=npairs,
        nselected=nselected, emat=emat, maxchi_mat=maxchi_mat,
        charge_mat=charge_mat,
    )
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
