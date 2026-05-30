# quantum_gateway/encoding/angle.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Angle Encoding Layer

"""
AngleEncoder: Maps classical data to qubit rotation angles.

Encoding strategy:
    For n-dimensional input x ∈ R^n:
        qubit i → RY(π * x_i) gate

This embeds data into the Bloch sphere latitudes.
Supports data vectors of length up to n_qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


class AngleEncoder:
    """
    Angle encoding: classical data → qubit rotations.

        ε_angle(x_i) = RY(π · x_i) on qubit i

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= maximum feature dimension).

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data as qubit rotation angles.

        Parameters
        ----------
        data : np.ndarray
            1D array of floats, length <= n_qubits.
            Values should be normalized to [0, 1] for best results,
            but arbitrary floats are accepted (angle = π * x).

        Returns
        -------
        QuantumCircuit
            Encoded circuit with RY gates applied.

        Raises
        ------
        ValueError
            If data length exceeds n_qubits.
        """
        data = np.asarray(data, dtype=float).flatten()

        if len(data) > self.n_qubits:
            raise ValueError(
                f"Data length {len(data)} exceeds n_qubits={self.n_qubits}. "
                "Truncate or reduce dimensionality."
            )

        qc = QuantumCircuit(self.n_qubits, name="AngleEncoding")

        for i, x in enumerate(data):
            angle = float(np.pi * x)
            qc.ry(angle, i)

        return qc

    def encode_batch(self, batch: np.ndarray) -> list[QuantumCircuit]:
        """
        Encode a batch of data samples.

        Parameters
        ----------
        batch : np.ndarray
            2D array of shape (n_samples, n_features).

        Returns
        -------
        list of QuantumCircuit
        """
        batch = np.atleast_2d(batch)
        return [self.encode(row) for row in batch]

    def __repr__(self) -> str:
        return f"AngleEncoder(n_qubits={self.n_qubits})"
