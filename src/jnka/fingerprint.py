from dataclasses import dataclass, asdict
import numpy as np

from .distributions import (
    entropy,
    kl_divergence,
    js_divergence,
    total_variation,
    l1_distance,
)


@dataclass
class NoiseFingerprint:
    kl: float
    js: float
    tv: float
    entropy: float
    l1: float

    def vector(self):
        return np.array(
            [
                self.kl,
                self.js,
                self.tv,
                self.entropy,
                self.l1,
            ],
            dtype=float,
        )

    def to_dict(self):
        return asdict(self)


def compute_fingerprint(ideal_distribution, observed_distribution):
    return NoiseFingerprint(
        kl=kl_divergence(observed_distribution, ideal_distribution),
        js=js_divergence(observed_distribution, ideal_distribution),
        tv=total_variation(observed_distribution, ideal_distribution),
        entropy=entropy(observed_distribution),
        l1=l1_distance(observed_distribution, ideal_distribution),
    )
