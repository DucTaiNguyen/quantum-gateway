# quantum_gateway/measurement/probabilistic.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Probabilistic Measurement Layer

"""
ProbabilisticMeasurer: Shot-based measurement via Qiskit Aer QASM simulator.

Simulates real quantum hardware measurement by sampling the
output distribution over `shots` repeated executions.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class ProbabilisticMeasurer:
    """
    Shot-based probabilistic measurement.

    Adds measurement gates to all qubits and runs the circuit
    on the AerSimulator QASM backend for `shots` repetitions.

    Parameters
    ----------
    shots : int
        Default number of measurement repetitions.

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def __init__(self, shots: int = 1024):
        self.shots   = shots
        self._backend = AerSimulator()

    def measure(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        observable: Optional[str] = None,  # not used in shot mode
    ) -> Dict[str, Any]:
        """
        Sample the quantum circuit output distribution.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to measure.
        shots : int, optional
            Override default shots.

        Returns
        -------
        dict
            {
              'counts': {bitstring: int},
              'probabilities': {bitstring: float},
              'entropy': float,
            }
        """
        shots = shots or self.shots

        # Attach measurement gates
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Run on AerSimulator
        job    = self._backend.run(meas_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        total  = sum(counts.values())
        probs  = {bs: c / total for bs, c in counts.items()}

        # Entropy
        p_arr   = np.array(list(probs.values()))
        p_arr   = p_arr[p_arr > 1e-12]
        entropy = float(-np.sum(p_arr * np.log2(p_arr)))

        return {
            "counts":        counts,
            "probabilities": probs,
            "entropy":       entropy,
        }

    def __repr__(self) -> str:
        return f"ProbabilisticMeasurer(shots={self.shots})"
