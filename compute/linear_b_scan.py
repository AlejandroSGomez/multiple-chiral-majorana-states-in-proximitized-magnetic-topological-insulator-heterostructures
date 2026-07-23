#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scan the linear ``B`` term at a fixed in-plane momentum ``kx``.

This script generates the linear-B data used in Fig. 4 and Fig. A2. The
``--kx`` option selects either the small positive momentum of Fig. 4 or the
particle-hole-invariant momentum used in Fig. A2.

Subcommands
-----------
map-row   Compute one row of the N_pairs(B, L_z) map: a single L_z (selected by
          --ilz into the canonical L_z grid) swept over the full B grid.  Meant to
          be launched as a SLURM array, one L_z per task.
case      Compute the slab spectrum sweep E(kx) for one representative case
          (A or B), select the two lowest-|E| states *at kx=0 exactly*, and store
          the bands (for panels c,d), the selected-state densities/chi/Q
          (for panels e,f).
state-pair
          Compute only one fixed-kx eigensystem and store the two lowest
          non-negative-energy bands with their densities/chi/Q. This is the
          non-redundant state selection used by the current paper figures.
point     Single (L_z, B, kx) -> N_pairs.  Used for validation / smoke tests.

The defaults reproduce, at the requested kx, the manuscript values:
    B   in linspace(0.0, 2.2, 151)
    L_z in arange(2.0, 71.0, 2.0)              (35 values)
    L_y = 300, Delta = 0.1, T = 0.45, G = -0.46
    bands = 14, n_select = 20, chi > 0.5, |Q| < 0.1, N_pairs = count // 2
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla

# 8x8 gamma matrices: Nambu x orbital x spin.
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
s0 = np.eye(2, dtype=complex)


def kron3(A, B, C):
    return np.kron(np.kron(A, B), C)


Gz00 = kron3(sz, s0, s0)
Gz0z = kron3(sz, s0, sz)
G00x = kron3(s0, s0, sx)
Gzzy = kron3(sz, sz, sy)
Gzzz = kron3(sz, sz, sz)
G0yy = kron3(s0, sy, sy)
Gyyz = kron3(sy, sy, sz)
Gzyy = kron3(sz, sy, sy)

UNIT_C = kron3(sx, s0, s0)
CHARGE_DIAG = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)

BASE_PARAMS = dict(
    L_z=50.0, L_y=300.0,
    C0=0.1, C1=0.0, C2=0.0,
    M0=-0.28, M1=10.0, M2=56.59,
    A=4.1, G=-0.46,
    B=0.0, T=0.45, Delta=0.1,
    a=4.0, c=2.0,
    ky=0.0, kz=0.0,
)

B_VALUES = np.linspace(0.0, 2.2, 151)
LZ_VALUES = np.arange(2.0, 71.0, 2.0)            # 2, 4, ..., 70  (35 values)

PAIR_MAP_BANDS = 14
PAIR_MAP_N_SELECT = 20
CHI_THRESHOLD = 0.50
CHARGE_THRESHOLD = 0.10


def build_slab(params):
    import kwant

    a = params["a"]
    c = params["c"]
    L_y = params["L_y"]
    L_z = params["L_z"]

    lat = kwant.lattice.general([(a, 0), (0, c)], basis=[(0, 0)], norbs=8)

    def onsite(site, kx, a, c, C0, C1, C2, M0, M1, M2, A, G, Delta, T, B):
        ck2_xy = 2.0 - np.cos(kx * a)
        ck2_z = 1
        epsilon_0 = C0 + (2 * C2 / a**2) * ck2_xy + (2 * C1 / c**2) * ck2_z
        M_0 = M0 + (2 * M2 / a**2) * ck2_xy + (2 * M1 / c**2) * ck2_z
        return (
            epsilon_0 * Gz00
            + (A / a) * (np.sin(kx * a) * G00x)
            + M_0 * Gz0z
            + G * Gzzz
            + Delta * Gyyz
            + T * Gzyy
        )

    def hopping_z(site1, site2, c, C1, M1, B, T):
        return (C1 / c**2) * Gz00 + (M1 / c**2) * Gz0z + 1j * B / (2 * c) * G0yy

    def hopping_y(site1, site2, a, C2, M2, A):
        return (C2 / a**2) * Gz00 + (M2 / a**2) * Gz0z + 1j * A / (2 * a) * Gzzy

    syst = kwant.Builder()

    def shape(pos):
        y, z = pos
        return (0 <= y < L_y) and (0 <= z < L_z)

    syst[lat.shape(shape, (0, 0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat.sublattices[0], lat.sublattices[0])] = hopping_y
    syst[kwant.builder.HoppingKind((0, 1), lat.sublattices[0], lat.sublattices[0])] = hopping_z
    return syst.finalized()


def _sorted_eigs(ens, wfs):
    order = np.argsort(ens)
    return ens[order], wfs[:, order]


def site_y_index(syst):
    """y lattice index of each site, in hamiltonian_submatrix row order.

    Robust to kwant's internal site ordering (same approach as the Josephson
    chi-map code). tag[0] is the y index for the lattice ([(a,0),(0,c)] basis).
    """
    return np.array([s.tag[0] for s in syst.sites], dtype=int)


def regional_chi_per_state(vecs, mask):
    """chi_Omega = |sum_{s in Omega} MP_s|^2 / (sum_{s in Omega} rho_s)^2 for each
    state, with the *bilinear* Majorana polarization MP_s = psi_s^T C psi_s
    (= 2 sum_a u_a v_a), exactly as in eq.(regional-majorana-polarization) and the
    Josephson chi-map. ``vecs`` has shape (Nsites, 8, Nstates); ``mask`` selects the
    region's sites along the first axis.
    """
    ps = vecs[mask]                                            # (nreg, 8, Nstates)
    mp = np.einsum("sin,ij,sjn->n", ps, UNIT_C, ps, optimize=True)   # bilinear, no conj
    rho = np.sum(np.abs(ps)**2, axis=(0, 1))                  # (Nstates,)
    out = np.zeros(vecs.shape[-1])
    good = rho > 1e-300
    out[good] = np.abs(mp[good])**2 / rho[good]**2
    return out


def majorana_pair_count(energies, wfs, ysite, ly,
                        chi_threshold=CHI_THRESHOLD,
                        charge_threshold=CHARGE_THRESHOLD,
                        n_select=PAIR_MAP_N_SELECT):
    """Fig.3 pair-count criterion, with the *regional* Majorana indicator
    chi = max(chi_L, chi_R) (left/right halves in y) instead of the global one.
    Charge criterion |<Q>| < charge_threshold is unchanged.
    """
    idx = np.argsort(np.abs(energies))[:int(n_select)]
    selected = wfs[:, idx]
    vecs = selected.reshape(-1, 8, idx.size)                  # (Nsites, 8, k)

    mask_L = ysite < (ly / 2.0)
    chi_L = regional_chi_per_state(vecs, mask_L)
    chi_R = regional_chi_per_state(vecs, ~mask_L)
    chi = np.maximum(chi_L, chi_R)

    charge = np.abs(np.einsum("sin,i->n", np.abs(vecs)**2, CHARGE_DIAG, optimize=True))

    majorana_like = (chi > chi_threshold) & (charge < charge_threshold)
    return int(np.count_nonzero(majorana_like) // 2)


def count_for_B(B_value, syst, base_params, ysite, ly, kx, bands=PAIR_MAP_BANDS):
    params = dict(base_params)
    params["B"] = float(B_value)
    params["kx"] = float(kx)
    H_k = syst.hamiltonian_submatrix(params=params, sparse=True)
    ens, wfs = sla.eigsh(H_k.tocsc(), k=int(bands), sigma=0, which="LM",
                         return_eigenvectors=True)
    ens, wfs = _sorted_eigs(ens, wfs)
    return majorana_pair_count(ens, wfs, ysite, ly)


def run_map_row(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.lz is not None:
        L_z = float(args.lz)
        ilz = int(np.argmin(np.abs(LZ_VALUES - L_z)))
    else:
        ilz = int(args.ilz)
        L_z = float(LZ_VALUES[ilz])

    params = dict(BASE_PARAMS)
    params["L_z"] = L_z
    params["B"] = 0.0
    syst = build_slab(params)
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1

    progress = Path(args.progress) if args.progress else None
    row = np.empty(B_VALUES.size, dtype=int)
    for j, B in enumerate(B_VALUES):
        row[j] = count_for_B(B, syst, params, ysite, ly, args.kx)
        if progress is not None and (j % 10 == 0 or j == B_VALUES.size - 1):
            progress.write_text(
                f"Lz={L_z:g} kx={args.kx:g}  {j + 1}/{B_VALUES.size}  "
                f"last N_pairs={row[j]}\n"
            )

    ktag = f"{args.kx:.4g}".replace("-", "m").replace(".", "p")
    out_file = outdir / f"fig3row_kx{ktag}_iLz{ilz:02d}_Lz{L_z:g}.npz"
    np.savez(out_file, B_vals=B_VALUES, Lz=L_z, ilz=ilz, kx=float(args.kx),
             pair_row=row, chi_threshold=CHI_THRESHOLD,
             charge_threshold=CHARGE_THRESHOLD)
    print(f"[map-row] wrote {out_file}  (max N_pairs in row = {int(row.max())})")


def run_case(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    params = dict(BASE_PARAMS)
    params["B"] = float(args.B)
    params["L_z"] = float(args.lz)
    syst = build_slab(params)
    Ly_sites = int(params["L_y"] / params["a"])
    Lz_sites = int(params["L_z"] / params["c"])
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1
    mask_L = ysite < (ly / 2.0)

    n_k = int(args.n_k)
    bands = int(args.bands)
    kx_values = np.linspace(-0.05, 0.05, n_k)

    energy_kx = np.empty((n_k, bands))
    evec_kx = np.empty((n_k, 8 * Ly_sites * Lz_sites, bands), dtype=complex)
    for i, kx in enumerate(kx_values):
        p = dict(params)
        p["kx"] = float(kx)
        H_k = syst.hamiltonian_submatrix(params=p, sparse=True)
        ens, wfs = sla.eigsh(H_k.tocsc(), k=bands, sigma=0, return_eigenvectors=True)
        ens, wfs = _sorted_eigs(ens, wfs)
        energy_kx[i] = ens
        evec_kx[i] = wfs

    q_diag = np.diag(Gz00).real
    evs = evec_kx.reshape(n_k, Ly_sites * Lz_sites, 8, bands)
    Q_expect = np.einsum("ksan,a->kn", np.abs(evs)**2, q_diag, optimize=True).real

    k0 = int(np.argmin(np.abs(kx_values)))
    order0 = np.argsort(np.abs(energy_kx[k0]))

    payload = dict(
        kx_values=kx_values, energy_kx=energy_kx, Q_expect=Q_expect,
        k0=k0, kx0=float(kx_values[k0]),
        L_y=float(params["L_y"]), L_z=float(params["L_z"]),
        a=float(params["a"]), c=float(params["c"]),
        B=float(params["B"]), Delta=float(params["Delta"]),
        label=str(args.label),
        Ly_sites=Ly_sites, Lz_sites=Lz_sites,
    )

    def chi_region_single(vec, mask):
        ps = vec.reshape(-1, 8)[mask]
        mp = np.einsum("si,ij,sj->", ps, UNIT_C, ps, optimize=True)   # bilinear, no conj
        rho = float(np.sum(np.abs(ps)**2))
        return float(abs(mp)**2 / rho**2) if rho > 1e-300 else 0.0

    for n, lab in enumerate(("a", "b")):
        band = int(order0[n])
        vec = evec_kx[k0, :, band]
        chi_L = chi_region_single(vec, mask_L)
        chi_R = chi_region_single(vec, ~mask_L)
        side = "R" if chi_R >= chi_L else "L"
        chi = max(chi_L, chi_R)
        density = np.sum(np.abs(vec.reshape(Ly_sites * Lz_sites, 8))**2, axis=1)
        density = density.reshape([Ly_sites, Lz_sites]).T          # (Lz_sites, Ly_sites)
        payload[f"s{n}_label"] = lab
        payload[f"s{n}_band"] = band
        payload[f"s{n}_energy"] = float(energy_kx[k0, band])
        payload[f"s{n}_chi"] = chi
        payload[f"s{n}_chi_L"] = chi_L
        payload[f"s{n}_chi_R"] = chi_R
        payload[f"s{n}_Q"] = float(Q_expect[k0, band])
        payload[f"s{n}_side"] = side
        payload[f"s{n}_density"] = density

    out_file = outdir / f"fig3case_{args.label}_kx0.npz"
    np.savez(out_file, **payload)
    print(f"[case {args.label}] wrote {out_file}  "
          f"kx0={kx_values[k0]:.5f}  "
          f"chi_a={payload['s0_chi']:.3f} ({payload['s0_side']}), "
          f"chi_b={payload['s1_chi']:.3f} ({payload['s1_side']})")


def run_casefine(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    params = dict(BASE_PARAMS)
    params["B"] = float(args.B)
    params["L_z"] = float(args.lz)
    syst = build_slab(params)
    Ly_sites = int(params["L_y"] / params["a"])
    Lz_sites = int(params["L_z"] / params["c"])
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1
    mask_L = ysite < (ly / 2.0)
    q_diag = np.diag(Gz00).real

    n_k = int(args.n_k)
    bands = int(args.bands)
    kx_values = np.linspace(-0.05, 0.05, n_k)
    k0 = int(np.argmin(np.abs(kx_values)))

    energy_kx = np.empty((n_k, bands))
    Q_expect = np.empty((n_k, bands))
    wfs_k0 = None
    for i, kx in enumerate(kx_values):
        p = dict(params)
        p["kx"] = float(kx)
        H = syst.hamiltonian_submatrix(params=p, sparse=True)
        ens, wfs = sla.eigsh(H.tocsc(), k=bands, sigma=0, return_eigenvectors=True)
        ens, wfs = _sorted_eigs(ens, wfs)
        energy_kx[i] = ens
        evs = wfs.reshape(Ly_sites * Lz_sites, 8, bands)
        Q_expect[i] = np.einsum("san,a->n", np.abs(evs)**2, q_diag, optimize=True).real
        if i == k0:
            wfs_k0 = wfs

    # E=0 state at kx != 0: minimum |E| excluding a central |kx c| window, so it is
    # always a distinct point from the kx=0 state (a genuine off-zero crossing).
    kx_excl = float(args.kx_excl)                   # in units of kx*c
    absE = np.abs(energy_kx).astype(float)
    absE[np.abs(kx_values * params["c"]) < kx_excl, :] = np.inf
    kE0, bandE0 = np.unravel_index(int(np.argmin(absE)), absE.shape)
    kE0, bandE0 = int(kE0), int(bandE0)
    pE = dict(params)
    pE["kx"] = float(kx_values[kE0])
    HE = syst.hamiltonian_submatrix(params=pE, sparse=True)
    ensE, wfsE = sla.eigsh(HE.tocsc(), k=bands, sigma=0, return_eigenvectors=True)
    ensE, wfsE = _sorted_eigs(ensE, wfsE)

    # kx=0 state: lowest |E| band at the kx=0 grid point.
    band_k0 = int(np.argmin(np.abs(energy_kx[k0])))

    def chi_region(vec, mask):
        ps = vec.reshape(-1, 8)[mask]
        mp = np.einsum("si,ij,sj->", ps, UNIT_C, ps, optimize=True)   # bilinear, no conj
        rho = float(np.sum(np.abs(ps)**2))
        return float(abs(mp)**2 / rho**2) if rho > 1e-300 else 0.0

    def make_state(prefix, vec, kx, energy):
        cL = chi_region(vec, mask_L)
        cR = chi_region(vec, ~mask_L)
        v = vec.reshape(-1, 8)
        cg = float(np.abs(np.einsum("sa,ab,sb->", v, UNIT_C, np.conj(v), optimize=True))**2)
        dens = np.sum(np.abs(vec.reshape(Ly_sites * Lz_sites, 8))**2, axis=1)
        dens = dens.reshape([Ly_sites, Lz_sites]).T
        return {
            f"{prefix}_density": dens,
            f"{prefix}_chi_global": cg,
            f"{prefix}_chi_L": cL,
            f"{prefix}_chi_R": cR,
            f"{prefix}_chi_regional": max(cL, cR),
            f"{prefix}_side": "R" if cR >= cL else "L",
            f"{prefix}_energy": float(energy),
            f"{prefix}_kx": float(kx),
        }

    payload = dict(
        kx_values=kx_values, energy_kx=energy_kx, Q_expect=Q_expect,
        k0=k0, kE0=kE0, label=str(args.label),
        L_y=float(params["L_y"]), L_z=float(params["L_z"]),
        a=float(params["a"]), c=float(params["c"]),
        B=float(params["B"]), Delta=float(params["Delta"]),
        Ly_sites=Ly_sites, Lz_sites=Lz_sites, n_k=n_k, bands=bands,
    )
    payload.update(make_state("kx0", wfs_k0[:, band_k0], kx_values[k0], energy_kx[k0, band_k0]))
    payload.update(make_state("E0", wfsE[:, bandE0], kx_values[kE0], ensE[bandE0]))

    # kx=0+ states used in the main Fig. 4 density panels. This is the same
    # fixed momentum at which the pair-count map is defined.
    kx_plus = float(args.kx_plus)
    pP = dict(params)
    pP["kx"] = kx_plus
    HP = syst.hamiltonian_submatrix(params=pP, sparse=True)
    ensP, wfsP = sla.eigsh(HP.tocsc(), k=bands, sigma=0, return_eigenvectors=True)
    ensP, wfsP = _sorted_eigs(ensP, wfsP)
    orderP = np.argsort(np.abs(ensP))
    payload["kx_plus"] = kx_plus
    for n, band in enumerate(orderP[:2]):
        payload.update(make_state(f"kp{n}", wfsP[:, int(band)], kx_plus, ensP[int(band)]))

    out_file = outdir / f"fig3casefine_{args.label}.npz"
    np.savez(out_file, **payload)
    print(
        f"[casefine {args.label}] wrote {out_file}  n_k={n_k}  "
        f"kx0-state: E/D={payload['kx0_energy'] / params['Delta']:+.3f} "
        f"chiG={payload['kx0_chi_global']:.3f} "
        f"chiReg={payload['kx0_chi_regional']:.3f}  "
        f"E0-state: kxc={payload['E0_kx'] * params['c']:+.4f} "
        f"E/D={payload['E0_energy'] / params['Delta']:+.4f} "
        f"chiG={payload['E0_chi_global']:.3f} "
        f"chiReg={payload['E0_chi_regional']:.3f}  "
        f"kxplus: kxc={kx_plus * params['c']:+.4f} "
        f"chi0={payload['kp0_chi_regional']:.3f} "
        f"chi1={payload['kp1_chi_regional']:.3f}"
    )


def run_state_pair(args):
    """Store the two lowest positive-energy bands and their real-space data.

    This lightweight calculation supports the paper figures without repeating
    the fine-kx sweep. Selecting two positive-energy bands avoids displaying a
    redundant particle-hole pair of the same BdG excitation.
    """
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    params = dict(BASE_PARAMS)
    params["B"] = float(args.B)
    params["L_z"] = float(args.lz)
    params["kx"] = float(args.kx)
    syst = build_slab(params)
    Ly_sites = int(params["L_y"] / params["a"])
    Lz_sites = int(params["L_z"] / params["c"])
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1
    mask_L = ysite < (ly / 2.0)

    H = syst.hamiltonian_submatrix(params=params, sparse=True)
    ens, wfs = sla.eigsh(
        H.tocsc(),
        k=int(args.bands),
        sigma=0,
        which="LM",
        return_eigenvectors=True,
    )
    ens, wfs = _sorted_eigs(ens, wfs)
    positive_bands = np.flatnonzero(ens >= 0.0)
    if positive_bands.size < 2:
        raise RuntimeError("Fewer than two non-negative-energy bands were computed")

    q_diag = np.diag(Gz00).real

    def chi_region_single(vec, mask):
        ps = vec.reshape(-1, 8)[mask]
        mp = np.einsum("si,ij,sj->", ps, UNIT_C, ps, optimize=True)
        rho = float(np.sum(np.abs(ps)**2))
        return float(abs(mp)**2 / rho**2) if rho > 1e-300 else 0.0

    payload = dict(
        label=str(args.label),
        selection="two_lowest_nonnegative_energy_bands",
        kx=float(args.kx),
        L_y=float(params["L_y"]),
        L_z=float(params["L_z"]),
        a=float(params["a"]),
        c=float(params["c"]),
        B=float(params["B"]),
        Delta=float(params["Delta"]),
        bands=int(args.bands),
        Ly_sites=Ly_sites,
        Lz_sites=Lz_sites,
    )

    for n, band in enumerate(positive_bands[:2]):
        band = int(band)
        vec = wfs[:, band]
        chi_L = chi_region_single(vec, mask_L)
        chi_R = chi_region_single(vec, ~mask_L)
        density = np.sum(
            np.abs(vec.reshape(Ly_sites * Lz_sites, 8))**2,
            axis=1,
        ).reshape([Ly_sites, Lz_sites]).T
        payload[f"state{n}_band"] = band
        payload[f"state{n}_energy"] = float(ens[band])
        payload[f"state{n}_chi_L"] = chi_L
        payload[f"state{n}_chi_R"] = chi_R
        payload[f"state{n}_chi"] = max(chi_L, chi_R)
        payload[f"state{n}_Q"] = float(
            np.einsum(
                "sa,a->",
                np.abs(vec.reshape(Ly_sites * Lz_sites, 8))**2,
                q_diag,
                optimize=True,
            ).real
        )
        payload[f"state{n}_density"] = density

    out_file = outdir / f"fig3statepair_{args.label}_{args.tag}.npz"
    np.savez(out_file, **payload)
    print(
        f"[state-pair {args.label}/{args.tag}] wrote {out_file}  "
        f"kxc={float(args.kx) * params['c']:+.6f}  "
        f"E/Delta={[round(payload[f'state{n}_energy'] / params['Delta'], 6) for n in (0, 1)]}  "
        f"chi={[round(payload[f'state{n}_chi'], 6) for n in (0, 1)]}"
    )


def run_point(args):
    params = dict(BASE_PARAMS)
    params["L_z"] = float(args.lz)
    params["B"] = 0.0
    syst = build_slab(params)
    ysite = site_y_index(syst)
    ly = int(ysite.max()) + 1
    n = count_for_B(args.B, syst, params, ysite, ly, args.kx)
    print(f"[point] Lz={args.lz:g} B={args.B:g} kx={args.kx:g} -> N_pairs={n}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("map-row", help="one L_z row of the N_pairs(B,Lz) map")
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--ilz", type=int, help="index into LZ_VALUES (SLURM array)")
    g.add_argument("--lz", type=float, help="explicit L_z value (Angstrom)")
    m.add_argument("--kx", type=float, default=0.0)
    m.add_argument("--out", default="results_fig3kx0")
    m.add_argument("--progress", default=None)
    m.set_defaults(func=run_map_row)

    c = sub.add_parser("case", help="spectrum sweep + selected states at kx=0")
    c.add_argument("--label", required=True)
    c.add_argument("--B", type=float, required=True)
    c.add_argument("--lz", type=float, required=True)
    c.add_argument("--n-k", dest="n_k", type=int, default=61)
    c.add_argument("--bands", type=int, default=14)
    c.add_argument("--out", default="results_fig3kx0")
    c.set_defaults(func=run_case)

    cf = sub.add_parser("casefine", help="fine-kx spectrum + density states at kx=0, kx=0+, and at E=0")
    cf.add_argument("--label", required=True)
    cf.add_argument("--B", type=float, required=True)
    cf.add_argument("--lz", type=float, required=True)
    cf.add_argument("--n-k", dest="n_k", type=int, default=301)
    cf.add_argument("--bands", type=int, default=14)
    cf.add_argument("--kx-excl", dest="kx_excl", type=float, default=0.01,
                    help="exclude |kx c| < this when picking the E~0, kx!=0 state")
    cf.add_argument("--kx-plus", dest="kx_plus", type=float, default=0.003333333333333334,
                    help="positive kx used for the main Fig. 4 density states")
    cf.add_argument("--out", default="results_fig3kx0")
    cf.set_defaults(func=run_casefine)

    sp = sub.add_parser(
        "state-pair",
        help="two lowest positive-energy states and densities at one fixed kx",
    )
    sp.add_argument("--label", required=True)
    sp.add_argument("--tag", required=True)
    sp.add_argument("--B", type=float, required=True)
    sp.add_argument("--lz", type=float, required=True)
    sp.add_argument("--kx", type=float, required=True)
    sp.add_argument("--bands", type=int, default=14)
    sp.add_argument("--out", default="results_fig3kx0")
    sp.set_defaults(func=run_state_pair)

    q = sub.add_parser("point", help="single (Lz,B,kx) -> N_pairs")
    q.add_argument("--lz", type=float, required=True)
    q.add_argument("--B", type=float, required=True)
    q.add_argument("--kx", type=float, default=0.0)
    q.set_defaults(func=run_point)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
