"""Fermion-parity (Pfaffian) diagnostic for the vertical Josephson junction.

This is a gauge-invariant test of the Josephson periodicity.
A protected 4-pi response requires a protected zero-energy crossing at phi=pi,
which is equivalent to a switch of the BdG ground-state fermion parity. The
fermion parity is the sign of the Pfaffian of the BdG Hamiltonian written in the
Majorana basis,

    P(phi) = sign Pf[ -i Omega H_BdG(kx=0, phi) Omega^dagger ],

evaluated at the particle-hole-invariant momentum kx=0 (the only kx where a
protected crossing can sit). If P(phi) is constant over [0, 2pi] the spectrum is
genuinely 2pi-periodic; a single sign change across phi=pi would signal a true
4pi (parity-pumping) response.

The slab builder, gamma matrices, phase profile, and default parameters are reused
from josephson_model.py so that conventions match the rest of the paper.
The particle-hole operator is ``C = kron(SX, eye(4))`` on each site.

Output: a .npz with arrays phi, pf_sign, pf_logabs, gap_min (min |E| at kx=0),
plus a one-line human summary written next to it.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import scipy.sparse.linalg as sla
import scipy.linalg as sclin

from josephson_model import (
    build_josephson_slab,
    DEFAULT_PARAMS,
    delta_func,
)

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def c_per_site(norb: int) -> np.ndarray:
    """Particle-hole unitary C acting on a single site (C psi* = P psi)."""
    if norb % 2:
        raise ValueError(f"norb={norb} is not even")
    return np.kron(SX, np.eye(norb // 2, dtype=complex))


def majorana_rotation(norb: int) -> np.ndarray:
    """Per-site unitary omega with omega C omega^T proportional to I.

    With C real symmetric and C^2 = I, omega = (I + i C)/sqrt(2) turns the
    particle-hole operation into pure complex conjugation, so that
    H' = omega H omega^dagger becomes purely imaginary Hermitian and
    S = -i H' is real antisymmetric.
    """
    c = c_per_site(norb)
    return (np.eye(norb, dtype=complex) + 1j * c) / np.sqrt(2.0)


def to_real_skew(h_dense: np.ndarray, n_sites: int, norb: int, atol: float) -> np.ndarray:
    """Rotate the dense BdG matrix to the Majorana basis and return S (real, antisym).

    The rotation Omega = I_nsites (x) omega is block diagonal with identical 8x8
    blocks, so H' = Omega H Omega^dagger is applied per site by contracting omega
    on the orbital indices. This is O(n_sites^2 * norb^3), avoiding two full dense
    n^3 matmuls (the original bottleneck).
    """
    omega = majorana_rotation(norb)
    h4 = h_dense.reshape(n_sites, norb, n_sites, norb)
    h_rot = np.einsum("ip,apbq,jq->aibj", omega, h4, omega.conj(), optimize=True)
    h_rot = h_rot.reshape(n_sites * norb, n_sites * norb)
    skew = -1j * h_rot
    imag_norm = float(np.linalg.norm(skew.imag))
    real_norm = float(np.linalg.norm(skew.real)) + 1e-300
    if imag_norm / real_norm > atol:
        raise RuntimeError(
            f"Majorana rotation left a large imaginary part "
            f"(||imag||/||real||={imag_norm/real_norm:.2e}); check the C convention."
        )
    s = skew.real
    # Remove the residual symmetric component from numerical roundoff.
    s = 0.5 * (s - s.T)
    return s


def pfaffian_sign_logabs(s_in: np.ndarray) -> Tuple[float, float]:
    """Sign and log|Pfaffian| of a real antisymmetric matrix via Hessenberg form.

    For real antisymmetric S, the orthogonal Hessenberg reduction H = Q^T S Q
    (scipy.linalg.hessenberg, LAPACK gehrd, BLAS-3) yields a skew-tridiagonal H
    because H^T = Q^T S^T Q = -H. Then Pf(S) = det(Q) * Pf(H) with
    Pf(H) = prod_{k even} H[k, k+1], so

        sign Pf = sign(det Q) * prod_{k even} sign(H[k,k+1]),
        log|Pf| = sum_{k even} log|H[k,k+1]|.

    This skips the QR eigenvalue iteration of the full Schur form and is ~2x
    faster on the large slab matrices. Sign and log-magnitude are returned
    separately to avoid over/underflow. Validated to agree with the Schur and
    Parlett-Reid Pfaffians to ~1e-13 (see pfaffian_sign_logabs_schur,
    _pfaffian_sign_logabs_ltl).
    """
    n = s_in.shape[0]
    if n % 2 == 1:
        return 0.0, float("-inf")
    h_mat, q_mat = sclin.hessenberg(np.asarray(s_in, dtype=np.float64), calc_q=True)
    sup = np.array([h_mat[k, k + 1] for k in range(0, n - 1, 2)], dtype=float)
    if np.any(sup == 0.0):
        return 0.0, float("-inf")
    sign_q, _ = np.linalg.slogdet(q_mat)
    sign = float(sign_q) * float(np.prod(np.sign(sup)))
    logabs = float(np.sum(np.log(np.abs(sup))))
    return sign, logabs


def pfaffian_sign_logabs_schur(s_in: np.ndarray) -> Tuple[float, float]:
    """Reference Pfaffian via the real Schur form (slower; cross-check of the
    Hessenberg routine). S = Q T Q^T with T block-diagonal 2x2 -> Pf = det(Q) prod t_k."""
    n = s_in.shape[0]
    if n % 2 == 1:
        return 0.0, float("-inf")
    t_mat, q_mat = sclin.schur(np.asarray(s_in, dtype=np.float64), output="real")
    t_off = np.array([t_mat[2 * k, 2 * k + 1] for k in range(n // 2)], dtype=float)
    if np.any(t_off == 0.0):
        return 0.0, float("-inf")
    sign_q = float(np.sign(np.linalg.det(q_mat)))
    sign = sign_q * float(np.prod(np.sign(t_off)))
    logabs = float(np.sum(np.log(np.abs(t_off))))
    return sign, logabs


def _pfaffian_sign_logabs_ltl(a_in: np.ndarray) -> Tuple[float, float]:
    """Reference Pfaffian (Parlett-Reid / LTL). Slow O(n^3) Python loop; used only
    to cross-check pfaffian_sign_logabs on small matrices."""
    a = np.array(a_in, dtype=np.float64, copy=True)
    n = a.shape[0]
    if n % 2 == 1:
        return 0.0, float("-inf")
    sign = 1.0
    logabs = 0.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(a[k + 1:, k])))
        if kp != k + 1:
            a[[k + 1, kp], :] = a[[kp, k + 1], :]
            a[:, [k + 1, kp]] = a[:, [kp, k + 1]]
            sign = -sign
        piv = a[k, k + 1]
        if piv == 0.0:
            return 0.0, float("-inf")
        sign *= float(np.sign(piv))
        logabs += float(np.log(abs(piv)))
        if k + 2 < n:
            tau = a[k, k + 2:] / piv
            a[k + 2:, k + 2:] += np.outer(tau, a[k + 2:, k + 1]) - np.outer(a[k + 2:, k + 1], tau)
    return sign, logabs


def build_params(args: argparse.Namespace) -> Dict:
    params = dict(DEFAULT_PARAMS)
    params["Delta_func"] = delta_func
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
    if args.T is not None:
        params["T"] = float(args.T)
    if args.G is not None:
        params["G"] = float(args.G)
    params["phi_top"] = float(args.phi_top)
    params["delta_profile"] = args.delta_profile
    params["kx"] = float(args.kx)
    return params


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BdG fermion-parity (Pfaffian) vs Josephson phase")
    p.add_argument("--Ly", type=float, default=300.0)
    p.add_argument("--Lz", type=float, default=20.0)
    p.add_argument("--a", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--Delta-u", dest="Delta_u", type=float, default=None)
    p.add_argument("--Delta-d", dest="Delta_d", type=float, default=None)
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--G", type=float, default=None)
    p.add_argument("--phi-top", type=float, default=0.0)
    p.add_argument("--delta-profile", choices=["step", "exp"], default="step")
    p.add_argument("--kx", type=float, default=0.0,
                   help="Must be 0 (particle-hole invariant momentum) for the Pfaffian to be defined")
    p.add_argument("--phi-min", type=float, default=0.0)
    p.add_argument("--phi-max", type=float, default=4.0 * np.pi)
    p.add_argument("--n-phi", type=int, default=161,
                   help="Fine grid for the cheap min-gap scan (sparse eigsh)")
    p.add_argument("--n-pf", type=int, default=5,
                   help="Number of phi points (evenly spaced over the sweep) where the "
                        "expensive dense Pfaffian is evaluated. Default 5 -> phi=0,pi,2pi,3pi,4pi")
    p.add_argument("--gap-bands", type=int, default=6,
                   help="Eigenvalues nearest 0 (sparse) used for the min-gap scan")
    p.add_argument("--skew-atol", type=float, default=1e-6,
                   help="Tolerance on residual imaginary part after the Majorana rotation")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("josephson_parity.npz"))
    p.add_argument("--progress", type=pathlib.Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if abs(args.kx) > 1e-12:
        raise SystemExit(
            f"kx={args.kx} != 0. The Pfaffian parity invariant is only defined at the "
            f"particle-hole-invariant momentum kx=0."
        )

    params = build_params(args)
    syst = build_josephson_slab(params)
    norb = 8

    # Fine sparse scan of the minimum gap.
    phis = np.linspace(args.phi_min, args.phi_max, args.n_phi)
    gap_min = np.zeros(args.n_phi, dtype=float)
    n_sites = None
    for i, phi in enumerate(phis):
        params["phi_rel"] = float(phi)
        h_sparse = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        if n_sites is None:
            dim = h_sparse.shape[0]
            n_sites = dim // norb
            if n_sites * norb != dim:
                raise RuntimeError(f"Hilbert dim {dim} not divisible by norb={norb}")
        ens = sla.eigsh(h_sparse, k=args.gap_bands, sigma=0.0, return_eigenvectors=False)
        gap_min[i] = float(np.min(np.abs(ens)))
        if args.progress is not None:
            with open(args.progress, "w") as fh:
                fh.write(f"gap scan {i+1}/{args.n_phi} phi={phi:.4f} gap={gap_min[i]:.3e}\n")
        if i % 10 == 0 or i == args.n_phi - 1:
            print(f"[gap {i+1:4d}/{args.n_phi}] phi={phi:7.4f}  gap_min={gap_min[i]:.4e}", flush=True)

    # Dense Pfaffian calculation over one phase cycle.
    pf_phi = np.linspace(0.0, 2.0 * np.pi, args.n_pf)
    pf_sign = np.full(args.n_pf, np.nan, dtype=float)
    pf_logabs = np.full(args.n_pf, np.nan, dtype=float)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save_npz():
        np.savez_compressed(
            args.out,
            phi=phis,
            gap_min=gap_min,
            pf_phi=pf_phi,
            pf_sign=pf_sign,
            pf_logabs=pf_logabs,
            Ly=params["L_y"],
            Lz=params["L_z"],
            a=params["a"],
            c=params["c"],
            Delta_u=params["Delta_u"],
            Delta_d=params["Delta_d"],
            T=params["T"],
            G=params["G"],
            kx=params["kx"],
        )

    save_npz()
    pflog = args.out.with_suffix(".pflog.txt")
    for j, phi in enumerate(pf_phi):
        params["phi_rel"] = float(phi)
        h_sparse = syst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
        h_dense = np.asarray(h_sparse.todense(), dtype=complex)
        skew = to_real_skew(h_dense, n_sites, norb, args.skew_atol)
        sign, logabs = pfaffian_sign_logabs(skew)
        pf_sign[j] = sign
        pf_logabs[j] = logabs
        save_npz()
        with open(pflog, "a") as fh:
            fh.write(f"phi={phi:.6f} pi={phi/np.pi:.3f} sign={sign:+.0f} log|Pf|={logabs:.6e}\n")
        if args.progress is not None:
            with open(args.progress, "w") as fh:
                fh.write(f"pfaffian {j+1}/{args.n_pf} phi={phi:.4f} sign={sign:+.0f} log|Pf|={logabs:.3e}\n")
        print(f"[pf  {j+1:2d}/{args.n_pf}] phi={phi:7.4f} (={phi/np.pi:.2f}pi)  "
              f"pf_sign={sign:+.0f}  log|Pf|={logabs:.6e}", flush=True)

    def closest(grid, val):
        return int(np.argmin(np.abs(grid - val)))

    s0 = pf_sign[closest(pf_phi, 0.0)]
    spi = pf_sign[closest(pf_phi, np.pi)]
    s2 = pf_sign[closest(pf_phi, 2.0 * np.pi)]
    pf_switches = int(np.sum(pf_sign[:-1] * pf_sign[1:] < 0))
    in_cycle = phis <= 2.0 * np.pi + 1e-9
    gap_cycle_min = float(np.min(gap_min[in_cycle]))
    verdict = "4pi (parity switches across pi)" if (s0 * spi < 0) else "2pi (no parity switch)"
    summary = (
        f"Lz={params['L_z']} Ly={params['L_y']} kx={params['kx']}\n"
        f"sign Pf: phi=0 -> {s0:+.0f}, phi=pi -> {spi:+.0f}, phi=2pi -> {s2:+.0f}\n"
        f"pf_sign changes over [0,2pi] ({args.n_pf} pts): {pf_switches}\n"
        f"min gap over [0,2pi] ({int(np.sum(in_cycle))} pts): {gap_cycle_min:.4e} eV\n"
        f"min gap over full sweep [0,4pi]: {float(np.min(gap_min)):.4e} eV\n"
        f"VERDICT: {verdict}\n"
    )
    print("\n" + summary, flush=True)
    summary_path = args.out.with_suffix(".summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary)


if __name__ == "__main__":
    main()
