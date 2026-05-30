# quantum_gateway/evolution/parameterized.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Parameterized Evolution Layer

"""
ParameterizedEvolution: Applies U(θ) to encoded quantum state.

Default ansatz (hardware-efficient, 2-qubit):

    H(0) — RY(θ₀) — CX(0,1) — RZ(θ₁/2) — RX(θ₂/3)

This implements a shallow entangling circuit suitable for both
simulation and near-term quantum hardware (NISQ regime).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


class ParameterizedEvolution:
    """
    Parameterized unitary evolution layer U(θ).

    Applies a variational quantum circuit to an encoded state circuit,
    building the composed circuit: U(θ)|ψ(I)⟩.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    layers : int
        Number of repetitions of the ansatz block (default: 1).

    Author
    ------
    Tai D. Nguyen © 2026
    """

    # Parameters per layer per qubit pair
    PARAMS_PER_LAYER = 3

    def __init__(self, n_qubits: int, layers: int = 1):
        self.n_qubits = n_qubits
        self.layers   = layers
        self.n_params = layers * self.PARAMS_PER_LAYER * max(1, n_qubits // 2)

    def apply(
        self,
        circuit: QuantumCircuit,
        theta: np.ndarray,
    ) -> QuantumCircuit:
        """
        Apply parameterized evolution U(θ) to an encoded circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            Encoding circuit |ψ(I)⟩.
        theta : np.ndarray
            Variational parameters. If fewer than n_params are given,
            remaining parameters default to π/4.

        Returns
        -------
        QuantumCircuit
            Full circuit: encoding + evolution.
        """
        theta = np.atleast_1d(theta).flatten().astype(float)

        # Pad or truncate theta to expected length
        if len(theta) < self.n_params:
            pad   = np.full(self.n_params - len(theta), np.pi / 4)
            theta = np.concatenate([theta, pad])
        theta = theta[:self.n_params]

        evolved = circuit.copy()
        evolved.barrier(label="U(θ) start")

        param_idx = 0
        for _ in range(self.layers):
            evolved = self._apply_layer(evolved, theta, param_idx)
            param_idx += self.PARAMS_PER_LAYER * max(1, self.n_qubits // 2)

        evolved.barrier(label="U(θ) end")
        return evolved

    def _apply_layer(
        self,
        qc: QuantumCircuit,
        theta: np.ndarray,
        offset: int,
    ) -> QuantumCircuit:
        """Apply one layer of the variational ansatz."""
        n = self.n_qubits
        idx = offset

        if n == 1:
            # Single qubit: basic rotation set
            qc.h(0)
            qc.rz(theta[idx], 0)
            return qc

        # Multi-qubit ansatz
        for pair_start in range(0, n - 1, 2):
            q0, q1 = pair_start, pair_start + 1
            t0     = theta[idx]     if idx     < len(theta) else np.pi / 4
            t1     = theta[idx + 1] if idx + 1 < len(theta) else np.pi / 4
            t2     = theta[idx + 2] if idx + 2 < len(theta) else np.pi / 4

            qc.h(q0)
            qc.ry(t0, q1)
            qc.cx(q0, q1)
            qc.rz(t1 / 2, q0)
            qc.rx(t2 / 3, q1)

            idx += self.PARAMS_PER_LAYER

        # If odd qubit count, apply single rotation to last qubit
        if n % 2 == 1:
            last = n - 1
            qc.ry(theta[idx % len(theta)], last)

        return qc

    def build_ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        """
        Build standalone ansatz circuit (without encoding prefix).

        Useful for inspecting or exporting the evolution circuit.
        """
        base = QuantumCircuit(self.n_qubits, name="U(θ)")
        return self.apply(base, theta)

    def parameter_count(self) -> int:
        """Return number of trainable parameters."""
        return self.n_params

    def __repr__(self) -> str:
        return (
            f"ParameterizedEvolution(n_qubits={self.n_qubits}, "
            f"layers={self.layers}, n_params={self.n_params})"
        )
