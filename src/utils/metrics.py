# quantum_gateway/utils/metrics.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Information & Quantum Metrics

"""
Utility functions for quantum and information-theoretic metrics.

Includes:
  - State fidelity
  - Von Neumann entropy
  - KL divergence
  - Trace distance
"""

from __future__ import annotations

import numpy as np
from typing import Union

from qiskit.quantum_info import Statevector, state_fidelity


def fidelity(
    state1: Union[np.ndarray, Statevector], state2: Union[np.ndarray, Statevector]
) -> float:
    """
    Quantum state fidelity: F(ρ, σ) = |⟨ψ|φ⟩|²

    Parameters
    ----------
    state1, state2 : array or Statevector

    Returns
    -------
    float in [0, 1]
    """
    if not isinstance(state1, Statevector):
        state1 = Statevector(np.asarray(state1, dtype=complex))
    if not isinstance(state2, Statevector):
        state2 = Statevector(np.asarray(state2, dtype=complex))
    return float(state_fidelity(state1, state2))


def von_neumann_entropy(probs: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon entropy of probability distribution.

        H = -Σ p_i log_b(p_i)

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution (will be normalized).
    base : float
        Logarithm base. Default 2 (bits).

    Returns
    -------
    float
    """
    probs = np.asarray(probs, dtype=float).flatten()
    probs = probs / probs.sum()
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL divergence D_KL(p || q).

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (will be normalized).

    Returns
    -------
    float
    """
    p = np.asarray(p, dtype=float).flatten()
    q = np.asarray(q, dtype=float).flatten()

    # Align lengths
    n = max(len(p), len(q))
    p = np.pad(p, (0, n - len(p)))
    q = np.pad(q, (0, n - len(q)))

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def trace_distance(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """
    Trace distance between two pure states.

        T(|ψ⟩, |φ⟩) = √(1 - |⟨ψ|φ⟩|²)

    Parameters
    ----------
    sv1, sv2 : np.ndarray
        Statevectors.

    Returns
    -------
    float in [0, 1]
    """
    sv1 = np.asarray(sv1, dtype=complex)
    sv2 = np.asarray(sv2, dtype=complex)
    inner = np.abs(np.dot(sv1.conj(), sv2)) ** 2
    return float(np.sqrt(max(0.0, 1.0 - inner)))


def distribution_overlap(p: np.ndarray, q: np.ndarray) -> float:
    """
    Bhattacharyya coefficient: measures overlap between distributions.

        BC = Σ √(p_i · q_i)

    Returns
    -------
    float in [0, 1]. 1 = identical, 0 = disjoint.
    """
    p = np.asarray(p, dtype=float).flatten()
    q = np.asarray(q, dtype=float).flatten()
    n = max(len(p), len(q))
    p = np.pad(p, (0, n - len(p)))
    q = np.pad(q, (0, n - len(q)))
    return float(np.sum(np.sqrt(p * q)))
