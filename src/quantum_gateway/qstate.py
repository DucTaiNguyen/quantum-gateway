import numpy as np


class QuantumState:
    """
    Minimal NumPy reference implementation of a pure quantum state.
    """

    def __init__(self, amplitudes):
        psi = np.asarray(amplitudes, dtype=complex).flatten()

        if psi.size == 0:
            raise ValueError("State cannot be empty")

        norm = np.linalg.norm(psi)

        if norm == 0:
            raise ValueError("State cannot be the zero vector")

        self.amplitudes = psi / norm

    @property
    def dimension(self):
        return self.amplitudes.size

    def probabilities(self):
        return np.abs(self.amplitudes) ** 2

    def density_matrix(self):
        psi = self.amplitudes.reshape(-1, 1)
        return psi @ psi.conj().T

    def fidelity(self, other):
        overlap = np.vdot(self.amplitudes, other.amplitudes)
        return float(np.abs(overlap) ** 2)

    def entropy(self):
        p = self.probabilities()
        p = p[p > 1e-15]

        return float(-np.sum(p * np.log2(p)))

    def summary(self):
        return {
            "dimension": self.dimension,
            "norm": float(np.linalg.norm(self.amplitudes)),
            "probabilities": self.probabilities().real.tolist(),
            "entropy": self.entropy(),
        }


def tensor_to_state(tensor):
    """
    Encode an information tensor into a normalized
    quantum state vector.

    The tensor is flattened and normalized.
    """

    array = np.asarray(tensor, dtype=float)

    if array.ndim == 0:
        raise ValueError("Tensor must contain multiple values")

    values = array.flatten()

    if np.any(values < 0):
        raise ValueError("Reference encoder requires non-negative values")

    return QuantumState(values.astype(complex))
