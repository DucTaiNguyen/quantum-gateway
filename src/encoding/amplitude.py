# quantum_gateway/encoding/amplitude.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Amplitude Encoding Layer

"""
AmplitudeEncoder: Encodes classical data as quantum state amplitudes.

Given a normalized vector x ∈ R^{2^n}, prepares the state:

    |ψ(x)⟩ = Σ_i x_i |i⟩

This maximally dense encoding requires 2^n amplitudes for n qubits,
allowing exponential data compression in state space.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize


class AmplitudeEncoder:
    """
    Amplitude encoding: data → quantum amplitudes.

        |ψ(x)⟩ = Σ_i x_i |i⟩,  where Σ_i x_i² = 1

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Encodes 2^n_qubits amplitudes.

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state_dim = 2**n_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data as quantum amplitudes.

        Parameters
        ----------
        data : np.ndarray
            1D array of length <= 2^n_qubits. Will be padded with zeros
            and L2-normalized automatically.

        Returns
        -------
        QuantumCircuit
            Circuit that initializes |ψ(data)⟩.

        Notes
        -----
        Uses Qiskit's Initialize instruction, which introduces O(2^n)
        depth. Suitable for simulation; hardware-efficient alternatives
        are planned for v0.3.
        """
        data = np.asarray(data, dtype=complex).flatten()

        if len(data) > self.state_dim:
            raise ValueError(
                f"Data length {len(data)} exceeds 2^{self.n_qubits} = {self.state_dim}. "
                "Reduce data dimensionality or increase n_qubits."
            )

        # Pad to state_dim
        if len(data) < self.state_dim:
            data = np.pad(data, (0, self.state_dim - len(data)))

        # L2 normalize → valid quantum state
        norm = np.linalg.norm(data)
        if norm < 1e-12:
            raise ValueError(
                "Cannot encode zero vector: no quantum state corresponds to it."
            )
        data = data / norm

        qc = QuantumCircuit(self.n_qubits, name="AmplitudeEncoding")
        qc.append(Initialize(data), range(self.n_qubits))
        return qc

    def required_qubits(self, data_dim: int) -> int:
        """Return minimum qubits needed to encode data_dim amplitudes."""
        return int(np.ceil(np.log2(data_dim)))

    def __repr__(self) -> str:
        return f"AmplitudeEncoder(n_qubits={self.n_qubits}, state_dim={self.state_dim})"
