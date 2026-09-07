# quantum_gateway/core/gateway.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Core Pipeline
# See LICENSE for terms of use.

"""
QuantumGateway: Main orchestration class.

Implements the formal mapping:
    Q: I → |ψ(I)⟩ → I'

As a composed pipeline:
    Q = M ∘ U(θ) ∘ ε
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Dict, Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator

from .pipeline import GatewayPipeline
from ..encoding.amplitude import AmplitudeEncoder
from ..encoding.angle import AngleEncoder
from ..evolution.parameterized import ParameterizedEvolution
from ..measurement.probabilistic import ProbabilisticMeasurer
from ..measurement.statevector import StatevectorMeasurer
from ..noise.kraus import KrausNoiseModel
from ..utils.metrics import fidelity, von_neumann_entropy


class QuantumGateway:
    """
    Quantum Gateway — Hybrid Classical–Quantum Transformation Layer.

    The central abstraction implementing:

        Q = M ∘ U(θ) ∘ ε

    Where:
        ε      : Encoding Layer   — Classical data I → |ψ(I)⟩
        U(θ)   : Evolution Layer  — Parameterized unitary transformation
        M      : Measurement Layer — |ψ⟩ → Classical distribution I'

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum register.
    encoding : str
        Encoding strategy: 'amplitude' | 'angle' | 'basis'.
    noise_model : optional KrausNoiseModel
        If provided, applies Kraus noise during evolution.
    backend : str
        Simulation backend: 'statevector' | 'qasm'.

    Author
    ------
    Tai D. Nguyen © 2026
    """

    VERSION = "0.1.0-alpha"
    AUTHOR = "Tai D. Nguyen"

    def __init__(
        self,
        n_qubits: int = 2,
        encoding: str = "angle",
        noise_model: Optional[KrausNoiseModel] = None,
        backend: str = "statevector",
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")

        self.n_qubits = n_qubits
        self.encoding = encoding
        self.noise_model = noise_model
        self.backend = backend

        # Build encoder
        if encoding == "amplitude":
            self._encoder = AmplitudeEncoder(n_qubits)
        elif encoding == "angle":
            self._encoder = AngleEncoder(n_qubits)
        else:
            raise ValueError(
                f"Unknown encoding: '{encoding}'. Choose 'amplitude' or 'angle'."
            )

        # Build evolution layer
        self._evolution = ParameterizedEvolution(n_qubits)

        # Build measurement layer
        if backend == "statevector":
            self._measurer = StatevectorMeasurer()
        else:
            self._measurer = ProbabilisticMeasurer(shots=1024)

        # Pipeline
        self._pipeline = GatewayPipeline(
            encoder=self._encoder,
            evolution=self._evolution,
            measurer=self._measurer,
            noise_model=noise_model,
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def encode(self, data: Union[list, np.ndarray]) -> QuantumCircuit:
        """
        Encoding layer ε: classical data → quantum circuit encoding |ψ(I)⟩.

        Parameters
        ----------
        data : array-like
            Classical input data of appropriate dimension.

        Returns
        -------
        QuantumCircuit
            Circuit prepared to represent |ψ(I)⟩.
        """
        data = np.asarray(data, dtype=float)
        return self._encoder.encode(data)

    def evolve(
        self,
        circuit: QuantumCircuit,
        theta: Union[float, np.ndarray],
    ) -> QuantumCircuit:
        """
        Evolution layer U(θ): apply parameterized unitary to encoded state.

        Parameters
        ----------
        circuit : QuantumCircuit
            Encoded quantum circuit |ψ(I)⟩.
        theta : float or array-like
            Variational parameters for U(θ).

        Returns
        -------
        QuantumCircuit
            Evolved circuit representing U(θ)|ψ(I)⟩.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return self._evolution.apply(circuit, theta)

    def measure(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        observable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Measurement layer M: collapse |ψ⟩ → classical probability distribution.

        Parameters
        ----------
        circuit : QuantumCircuit
            Quantum circuit to measure.
        shots : int
            Number of measurement shots (for probabilistic backend).
        observable : str, optional
            Pauli observable string for expectation value (e.g. 'ZZ').

        Returns
        -------
        dict
            {'counts': {...}, 'probabilities': {...}, 'entropy': float}
        """
        return self._measurer.measure(circuit, shots=shots, observable=observable)

    def transform(
        self,
        data: Union[list, np.ndarray],
        theta: Union[float, np.ndarray] = np.pi / 4,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Full Q = M ∘ U(θ) ∘ ε pipeline.

        Parameters
        ----------
        data : array-like
            Classical input I.
        theta : float or array-like
            Variational parameters θ.
        shots : int
            Measurement shots.

        Returns
        -------
        dict
            Measurement output I', including counts, probabilities, entropy.
        """
        return self._pipeline.run(data=data, theta=theta, shots=shots)

    def loss(
        self,
        data: Union[list, np.ndarray],
        target: Union[list, np.ndarray],
        theta: Union[float, np.ndarray],
    ) -> float:
        """
        Compute variational loss:

            L(θ) = L( M(U(θ) ε(I)), I_target )

        Uses KL divergence between output distribution and target.

        Parameters
        ----------
        data : array-like
            Input I.
        target : array-like
            Target distribution I_target.
        theta : float or array-like
            Parameters θ.

        Returns
        -------
        float
            Loss value.
        """
        result = self.transform(data, theta=theta, shots=4096)
        probs = np.array(list(result["probabilities"].values()))
        target_p = np.asarray(target, dtype=float)
        target_p = target_p / target_p.sum()

        # KL divergence: D_KL(output || target)
        eps = 1e-12
        kl = np.sum(probs * np.log((probs + eps) / (target_p[: len(probs)] + eps)))
        return float(kl)

    def info(self) -> Dict[str, Any]:
        """Return system metadata."""
        return {
            "name": "Quantum Gateway",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "copyright": "© 2026 Tai D. Nguyen. All rights reserved.",
            "n_qubits": self.n_qubits,
            "encoding": self.encoding,
            "backend": self.backend,
            "noise": self.noise_model is not None,
        }

    def __repr__(self) -> str:
        return (
            f"QuantumGateway(n_qubits={self.n_qubits}, "
            f"encoding='{self.encoding}', backend='{self.backend}')"
        )
