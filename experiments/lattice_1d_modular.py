#!/usr/bin/env python3
"""
1+1D Modular Hamiltonian — Cross-check Script
==============================================

Computes ΔK on a 1+1D chain using the same Williamson methodology as
lattice_3d_modular_external.py, for cross-checking the published v0.5
lattice paper's reported "1/55" ratio at moderate (L, m).

Two protocols are tested:

  A) TWO-MASS PROTOCOL (matches v0.5 published methodology):
     pos1 at index (N-L)//2, pos2 at index (N+L)//2
     Both masses present; bipartition at chain midpoint.
     BW prediction: 2π·m·L (the convention used in v0.5).

  B) SINGLE-MASS-IN-A PROTOCOL (cleaner BW interpretation):
     Single mass at distance d1 from the cut, inside subsystem A.
     No second mass.
     BW prediction: 2π·m·d1.

The question this script answers: does 1+1D also show the peaked-then-decaying
d1-dependence that 3+1D revealed, or is 1+1D genuinely linear (matching the
BW asymptote up to the 1/55 prefactor)?

Methodology mirrors the 3+1D code:
- Free massless scalar on N-site chain with Dirichlet BCs (K[i,i] = 2 + m²,
  K[i, i±1] = -1).
- Ground-state covariances M_φ = (1/2)K^(-1/2), M_π = (1/2)K^(1/2).
- Restrict to subsystem A = first N/2 sites.
- Williamson decomposition with vacuum-mode projection (ν_k → 1/2 modes
  contribute zero).
- Modular kernels Q, P built on active subspace.
- ΔK = ⟨K_A_quad⟩_pert − ⟨K_A_quad⟩_vac via original-basis trace formula.
- Validations V1 (relative-entropy positivity) and V2 (modal-vs-original
  cross-check) enforced.

Tolerances and conventions are identical to the 3+1D script.
"""

import numpy as np
from scipy.linalg import eigh
import json
import time
import sys

VACUUM_MODE_TOL = 1e-9
V2_TOLERANCE = 1e-3
V1_TOLERANCE = -1e-3


def build_K_matrix_1d(N, mass_positions, m_value):
    """
    Free massless scalar on N-site chain with Dirichlet BCs.
    K[i,i] = 2 + m^2 (if i in mass_positions else 0)
    K[i, i±1] = -1 (with chain truncation at boundaries).
    """
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        K[i, i] = 2.0
        if i in mass_positions:
            K[i, i] += m_value ** 2
        if i + 1 < N:
            K[i, i + 1] = -1.0
            K[i + 1, i] = -1.0
    return K


def covariances_from_K(K):
    eigvals, eigvecs = eigh(K)
    if eigvals.min() < 1e-12:
        raise ValueError(f"K not positive definite: min eigval = {eigvals.min():.3e}")
    sqrt_K = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    inv_sqrt_K = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    M_phi = 0.5 * inv_sqrt_K
    M_pi = 0.5 * sqrt_K
    return M_phi, M_pi


def restrict_to_subsystem(M, A_indices):
    return M[np.ix_(A_indices, A_indices)]


def williamson_modular_kernel(M_phi_A, M_pi_A, vacuum_mode_tol=VACUUM_MODE_TOL):
    eigvals_phi, eigvecs_phi = eigh(M_phi_A)
    if eigvals_phi.min() < 1e-14:
        raise ValueError(f"M_phi singular: min eigval = {eigvals_phi.min():.3e}")
    sqrt_M_phi = eigvecs_phi @ np.diag(np.sqrt(eigvals_phi)) @ eigvecs_phi.T
    inv_sqrt_M_phi = eigvecs_phi @ np.diag(1.0 / np.sqrt(eigvals_phi)) @ eigvecs_phi.T

    P_sym = sqrt_M_phi @ M_pi_A @ sqrt_M_phi
    P_sym = 0.5 * (P_sym + P_sym.T)
    nu_squared, V_full = eigh(P_sym)
    if nu_squared.min() < 0.25 - 1e-10:
        raise ValueError(f"Williamson eigenvalues below 1/2: {nu_squared.min():.3e}")
    nu_full = np.sqrt(np.clip(nu_squared, 0.25, None))

    active_mask = (nu_full - 0.5) > vacuum_mode_tol
    n_active = int(active_mask.sum())
    n_total = len(nu_full)
    if n_active == 0:
        raise ValueError("No active modes after vacuum projection.")
    nu = nu_full[active_mask]
    V = V_full[:, active_mask]

    eps = np.log((nu + 0.5) / (nu - 0.5))

    Q = inv_sqrt_M_phi @ V @ np.diag(eps * nu) @ V.T @ inv_sqrt_M_phi
    P = sqrt_M_phi @ V @ np.diag(eps / nu) @ V.T @ sqrt_M_phi

    return {'Q': Q, 'P': P, 'nu': nu, 'eps': eps,
            'n_active': n_active, 'n_total': n_total, 'V': V}


def expectation_K_quad(Q, P, M_phi_state, M_pi_state):
    return 0.5 * np.trace(Q @ M_phi_state) + 0.5 * np.trace(P @ M_pi_state)


def vacuum_entropy_from_nu(nu):
    return float(np.sum(
        (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)
    ))


def half_chain_indices(N):
    """First N//2 sites of the chain = subsystem A."""
    return np.arange(N // 2)


def run_two_mass(N, L, m, verbose=False):
    """Two-mass protocol, matching v0.5 published methodology.
    pos1 = (N-L)//2 in A; pos2 = (N+L)//2 in B.
    BW prediction: 2πmL (v0.5 convention).
    """
    pos1 = (N - L) // 2
    pos2 = (N + L) // 2

    K_vac = build_K_matrix_1d(N, set(), 0.0)
    M_phi_vac, M_pi_vac = covariances_from_K(K_vac)
    K_pert = build_K_matrix_1d(N, {pos1, pos2}, m)
    M_phi_pert, M_pi_pert = covariances_from_K(K_pert)

    A_idx = half_chain_indices(N)
    M_phi_vac_A = restrict_to_subsystem(M_phi_vac, A_idx)
    M_pi_vac_A = restrict_to_subsystem(M_pi_vac, A_idx)
    M_phi_pert_A = restrict_to_subsystem(M_phi_pert, A_idx)
    M_pi_pert_A = restrict_to_subsystem(M_pi_pert, A_idx)

    will_vac = williamson_modular_kernel(M_phi_vac_A, M_pi_vac_A)
    Q, P, nu, eps = will_vac['Q'], will_vac['P'], will_vac['nu'], will_vac['eps']

    K_vac_modal = float(np.sum(eps * nu))
    K_vac_orig = expectation_K_quad(Q, P, M_phi_vac_A, M_pi_vac_A)
    v2_rel = abs(K_vac_modal - K_vac_orig) / max(abs(K_vac_orig), 1e-12)
    if v2_rel > V2_TOLERANCE:
        raise ValueError(f"V2 FAILED: rel_diff={v2_rel:.3e}")

    K_pert_orig = expectation_K_quad(Q, P, M_phi_pert_A, M_pi_pert_A)
    delta_K = K_pert_orig - K_vac_orig

    will_pert = williamson_modular_kernel(M_phi_pert_A, M_pi_pert_A)
    S_vac = vacuum_entropy_from_nu(nu)
    S_pert = vacuum_entropy_from_nu(will_pert['nu'])
    S_rel = delta_K + S_vac - S_pert
    if S_rel < V1_TOLERANCE:
        raise ValueError(f"V1 FAILED: S_rel={S_rel:.3e}")

    d1 = N // 2 - pos1  # perpendicular distance of pos1 (in A) to cut
    BW_v05 = 2 * np.pi * m * L          # v0.5 convention
    BW_d1 = 2 * np.pi * m * d1          # corrected single-particle BW
    return {
        'protocol': 'two_mass', 'N': N, 'L': L, 'm': m,
        'pos1': pos1, 'pos2': pos2, 'd1': d1,
        'delta_K': float(delta_K),
        'BW_v05': float(BW_v05), 'ratio_v05': float(delta_K / BW_v05),
        'BW_d1': float(BW_d1), 'ratio_d1': float(delta_K / BW_d1),
        'S_rel': float(S_rel),
        'S_vac': float(S_vac), 'S_pert': float(S_pert),
        'V1_passed': bool(S_rel >= V1_TOLERANCE),
        'V2_passed': bool(v2_rel <= V2_TOLERANCE),
        'V2_rel_diff': float(v2_rel),
    }


def run_single_mass_in_A(N, d1, m, verbose=False):
    """Single mass in A at perpendicular distance d1 from cut.
    BW prediction: 2π·m·d1.
    """
    half = N // 2
    pos1 = half - d1  # in A

    K_vac = build_K_matrix_1d(N, set(), 0.0)
    M_phi_vac, M_pi_vac = covariances_from_K(K_vac)
    K_pert = build_K_matrix_1d(N, {pos1}, m)
    M_phi_pert, M_pi_pert = covariances_from_K(K_pert)

    A_idx = half_chain_indices(N)
    M_phi_vac_A = restrict_to_subsystem(M_phi_vac, A_idx)
    M_pi_vac_A = restrict_to_subsystem(M_pi_vac, A_idx)
    M_phi_pert_A = restrict_to_subsystem(M_phi_pert, A_idx)
    M_pi_pert_A = restrict_to_subsystem(M_pi_pert, A_idx)

    will_vac = williamson_modular_kernel(M_phi_vac_A, M_pi_vac_A)
    Q, P, nu, eps = will_vac['Q'], will_vac['P'], will_vac['nu'], will_vac['eps']

    K_vac_modal = float(np.sum(eps * nu))
    K_vac_orig = expectation_K_quad(Q, P, M_phi_vac_A, M_pi_vac_A)
    v2_rel = abs(K_vac_modal - K_vac_orig) / max(abs(K_vac_orig), 1e-12)
    if v2_rel > V2_TOLERANCE:
        raise ValueError(f"V2 FAILED: rel_diff={v2_rel:.3e}")

    K_pert_orig = expectation_K_quad(Q, P, M_phi_pert_A, M_pi_pert_A)
    delta_K = K_pert_orig - K_vac_orig

    will_pert = williamson_modular_kernel(M_phi_pert_A, M_pi_pert_A)
    S_vac = vacuum_entropy_from_nu(nu)
    S_pert = vacuum_entropy_from_nu(will_pert['nu'])
    S_rel = delta_K + S_vac - S_pert
    if S_rel < V1_TOLERANCE:
        raise ValueError(f"V1 FAILED: S_rel={S_rel:.3e}")

    BW_d1 = 2 * np.pi * m * d1
    return {
        'protocol': 'single_mass', 'N': N, 'd1': d1, 'm': m, 'pos1': pos1,
        'delta_K': float(delta_K),
        'BW_d1': float(BW_d1), 'ratio_d1': float(delta_K / BW_d1),
        'S_rel': float(S_rel),
        'S_vac': float(S_vac), 'S_pert': float(S_pert),
        'V1_passed': bool(S_rel >= V1_TOLERANCE),
        'V2_passed': bool(v2_rel <= V2_TOLERANCE),
        'V2_rel_diff': float(v2_rel),
    }
