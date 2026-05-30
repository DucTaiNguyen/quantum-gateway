# quantum_gateway/noise/kraus.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Kraus Noise Model

"""
KrausNoiseModel: Structural noise via Kraus operator representation.

In Quantum Gateway, noise is not treated as error but as a
*structural exploration mechanism* — diversifying the state space
and enabling stochastic evolution.

Kraus density matrix evolution:
    ρ' = Σ_k E_k ρ E_k†

where {E_k} are Kraus operators satisfying Σ_k E_k† E_k = I.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error


class KrausNoiseModel:
    """
    Structural noise model for Quantum Gateway evolution.

    Supports depolarizing and amplitude damping noise channels.
    Noise is injected via Qiskit Aer's NoiseModel interface.

    Parameters
    ----------
    depolarizing_rate : float
        Depolarizing error probability per gate (0 to 1).
    amplitude_damping_rate : float
        T1 decay probability (0 to 1).
    apply_to : list[str]
        Gate names to apply noise to. Default: ['cx', 'ry', 'rz', 'rx'].

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def __init__(
        self,
        depolarizing_rate: float = 0.01,
        amplitude_damping_rate: float = 0.005,
        apply_to: Optional[list] = None,
    ):
        if not (0 <= depolarizing_rate <= 1):
            raise ValueError("depolarizing_rate must be in [0, 1]")
        if not (0 <= amplitude_damping_rate <= 1):
            raise ValueError("amplitude_damping_rate must be in [0, 1]")

        self.depolarizing_rate      = depolarizing_rate
        self.amplitude_damping_rate = amplitude_damping_rate
        self.apply_to               = apply_to or ["cx", "ry", "rz", "rx", "h"]

        self._noise_model = self._build()

    def _build(self) -> NoiseModel:
        nm = NoiseModel()

        # Depolarizing error (1-qubit gates)
        if self.depolarizing_rate > 0:
            dep_1q = depolarizing_error(self.depolarizing_rate, 1)
            dep_2q = depolarizing_error(self.depolarizing_rate * 10, 2)
            single_gates = [g for g in self.apply_to if g != "cx"]
            if single_gates:
                nm.add_all_qubit_quantum_error(dep_1q, single_gates)
            if "cx" in self.apply_to:
                nm.add_all_qubit_quantum_error(dep_2q, ["cx"])

        # Amplitude damping (T1 decay on single-qubit gates)
        if self.amplitude_damping_rate > 0:
            amp_damp = amplitude_damping_error(self.amplitude_damping_rate)
            amp_gates = [g for g in self.apply_to if g != "cx"]
            if amp_gates:
                nm.add_all_qubit_quantum_error(amp_damp, amp_gates)

        return nm

    def qiskit_noise_model(self) -> NoiseModel:
        """Return the underlying Qiskit NoiseModel for use with AerSimulator."""
        return self._noise_model

    def apply(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Tag circuit with noise metadata for downstream simulators.

        Note: Actual noise application happens at simulation time
        via AerSimulator(noise_model=...). This method returns
        the circuit unchanged but marks it for noisy simulation.

        For direct noisy statevector simulation, use:
            AerSimulator(method='density_matrix', noise_model=self.qiskit_noise_model())
        """
        circuit._data  # ensure circuit is valid
        # Attach noise model as circuit metadata
        circuit.metadata = circuit.metadata or {}
        circuit.metadata["quantum_gateway_noise"] = {
            "depolarizing_rate":      self.depolarizing_rate,
            "amplitude_damping_rate": self.amplitude_damping_rate,
        }
        return circuit

    def kraus_summary(self) -> dict:
        """Return summary of noise parameters."""
        return {
            "type":               "KrausNoiseModel",
            "depolarizing_rate":  self.depolarizing_rate,
            "amplitude_damping":  self.amplitude_damping_rate,
            "applied_to_gates":   self.apply_to,
            "interpretation":     "Noise as structural exploration (QG principle)",
            "author":             "Tai D. Nguyen © 2026",
        }

    def __repr__(self) -> str:
        return (
            f"KrausNoiseModel(depolarizing={self.depolarizing_rate}, "
            f"amplitude_damping={self.amplitude_damping_rate})"
        )
