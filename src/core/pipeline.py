# quantum_gateway/core/pipeline.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Pipeline Orchestration

"""
GatewayPipeline: Orchestrates the full Q = M ∘ U(θ) ∘ ε transformation.

Separates concern from QuantumGateway to allow custom pipeline injection,
noise insertion, and layer-by-layer inspection.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, Optional, Union


class GatewayPipeline:
    """
    Orchestrates the Quantum Gateway pipeline:

        I  →  ε(I)  →  U(θ)|ψ⟩  →  M  →  I'

    Parameters
    ----------
    encoder   : encoding layer instance
    evolution : evolution layer instance
    measurer  : measurement layer instance
    noise_model : optional noise model

    Author
    ------
    Tai D. Nguyen © 2026
    """

    def __init__(self, encoder, evolution, measurer, noise_model=None):
        self.encoder = encoder
        self.evolution = evolution
        self.measurer = measurer
        self.noise_model = noise_model
        self._history: list[Dict[str, Any]] = []

    def run(
        self,
        data: Union[list, np.ndarray],
        theta: Union[float, np.ndarray] = np.pi / 4,
        shots: int = 1024,
        record: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Parameters
        ----------
        data   : classical input I
        theta  : evolution parameters θ
        shots  : measurement shots
        record : whether to save run to history

        Returns
        -------
        dict with keys: counts, probabilities, entropy, elapsed_ms, metadata
        """
        t0 = time.perf_counter()
        data = np.asarray(data, dtype=float)

        # Stage 1 — Encoding ε
        circuit = self.encoder.encode(data)

        # Stage 2 — Evolution U(θ)
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        circuit = self.evolution.apply(circuit, theta)

        # Stage 3 — Optional noise injection
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)

        # Stage 4 — Measurement M
        result = self.measurer.measure(circuit, shots=shots)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        output = {
            **result,
            "elapsed_ms": round(elapsed_ms, 3),
            "metadata": {
                "n_qubits": circuit.num_qubits,
                "encoding": self.encoder.__class__.__name__,
                "evolution": self.evolution.__class__.__name__,
                "theta": theta.tolist(),
                "shots": shots,
                "noisy": self.noise_model is not None,
            },
        }

        if record:
            self._history.append(output)

        return output

    def history(self) -> list:
        """Return list of all past pipeline runs."""
        return self._history

    def clear_history(self) -> None:
        """Clear run history."""
        self._history.clear()
