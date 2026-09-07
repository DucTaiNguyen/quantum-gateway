# quantum_gateway/measurement/statevector.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Statevector Measurement Layer

"""
StatevectorMeasurer: Exact measurement via Qiskit statevector simulation.

Implements:
    M(|ψ⟩) = { p(x) = |⟨x|ψ⟩|² }

Returns the exact Born rule probability distribution without
sampling noise, enabling deterministic gradient computation.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _von_neumann_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of probability distribution (bits)."""
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log2(probs)))


class StatevectorMeasurer:
    """
    Exact statevector measurement layer.

    Computes the Born rule probability distribution:

        p(x) = |⟨x|ψ⟩|²

    for all computational basis states x ∈ {0,1}^n.

    This is a noiseless simulation method suitable for gradient-based
    optimization and analysis. For hardware-realistic simulation,
    use ProbabilisticMeasurer.

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def measure(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,  # ignored; kept for API compatibility
        observable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Measure quantum circuit via statevector simulation.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to measure (no classical registers required).
        shots : int
            Ignored for statevector; included for API consistency.
        observable : str, optional
            Pauli string for expectation value (e.g. 'ZZ', 'XZ').

        Returns
        -------
        dict
            {
              'counts': {bitstring: float_count},
              'probabilities': {bitstring: float},
              'entropy': float,            # Shannon entropy (bits)
              'statevector': complex array,
              'expectation': float | None,
            }
        """
        # Simulate exact statevector
        sv = Statevector.from_instruction(circuit)
        sv_array = sv.data
        n = circuit.num_qubits

        # Born rule probabilities
        probs_array = np.abs(sv_array) ** 2

        # Build bitstring → probability mapping
        bitstrings = [format(i, f"0{n}b") for i in range(2**n)]
        probabilities = {bs: float(p) for bs, p in zip(bitstrings, probs_array)}

        # Pseudo-counts (for API compatibility with ProbabilisticMeasurer)
        counts = {bs: float(p * shots) for bs, p in probabilities.items() if p > 1e-12}

        # Shannon entropy
        entropy = _von_neumann_entropy(probs_array)

        # Expectation value (optional)
        expectation = None
        if observable is not None:
            try:
                from qiskit.quantum_info import SparsePauliOp

                op = SparsePauliOp(observable)
                expectation = float(sv.expectation_value(op).real)
            except Exception as e:
                expectation = None

        return {
            "counts": counts,
            "probabilities": probabilities,
            "entropy": entropy,
            "statevector": sv_array,
            "expectation": expectation,
        }

    def __repr__(self) -> str:
        return "StatevectorMeasurer()"
