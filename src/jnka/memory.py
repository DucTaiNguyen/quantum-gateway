from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class NoiseRecord:
    experiment_id: str
    theta: float
    fingerprint: np.ndarray
    utility: float
    action: str
    metadata: Dict

class NoiseMemory:
    def __init__(self):
        self.records: List[NoiseRecord] = []

    def add(self, record):
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def fingerprint_matrix(self):
        if not self.records:
            return np.empty((0, 5))
        return np.vstack([r.fingerprint for r in self.records])

    def utility_vector(self):
        return np.array([r.utility for r in self.records])

    def latest(self, n=10):
        return self.records[-n:]
