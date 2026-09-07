import numpy as np


class InformationTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

        if self.data.ndim != 2:
            raise ValueError("InformationTensor must be 2-dimensional")

    def normalize(self):
        norm = np.linalg.norm(self.data)

        if norm == 0:
            raise ValueError("Cannot normalize zero tensor")

        return InformationTensor(self.data / norm)

    def correlation(self):
        return np.corrcoef(self.data)

    def summary(self):
        return {
            "shape": list(self.data.shape),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "mean": float(np.mean(self.data)),
            "norm": float(np.linalg.norm(self.data)),
        }


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def probability_distribution(values):
    values = np.asarray(values, dtype=float)

    total = np.sum(values)

    if total <= 0:
        raise ValueError("Values must have positive sum")

    return values / total


def zz_correlation(probabilities):
    """
    Classical computational-basis proxy for
    two-qubit ZZ correlation.

    Ordering:
        00 -> +1
        01 -> -1
        10 -> -1
        11 -> +1
    """

    p = np.asarray(probabilities, dtype=float)

    if len(p) != 4:
        raise ValueError("Expected four probabilities")

    z0 = p[0] + p[1] - p[2] - p[3]
    z1 = p[0] - p[1] + p[2] - p[3]
    zz = p[0] - p[1] - p[2] + p[3]

    return float(zz - z0 * z1)
