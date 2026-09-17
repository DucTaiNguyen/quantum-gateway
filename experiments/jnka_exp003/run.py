import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath("."))

from src.jnka import compute_fingerprint


SEEDS = range(30)
THETAS = np.linspace(0.0, 2.0 * np.pi, 51)
SHOTS = 4096


def ideal_distribution(theta):
    p = np.zeros(16)

    a = np.cos(theta / 2.0) ** 2
    b = np.sin(theta / 2.0) ** 2

    p[0] = 0.35 * a
    p[7] = 0.35 * a
    p[8] = 0.15 * b
    p[15] = 0.15 * b

    return p / p.sum()


def noisy_distribution(p, strength, rng):
    uniform = np.ones_like(p) / len(p)

    q = (1.0 - strength) * p + strength * uniform

    fluctuation = rng.normal(
        0.0,
        strength * 0.01,
        size=len(p),
    )

    q = np.clip(q + fluctuation, 0.0, None)

    return q / q.sum()


def sample(p, rng):
    return rng.multinomial(SHOTS, p) / SHOTS


def evaluate(X, y):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    return float(np.mean(scores))


def main():
    bases = []
    fingerprints = []
    targets = []

    for theta in THETAS:
        for seed in SEEDS:

            rng = np.random.default_rng(seed)

            ideal = ideal_distribution(theta)

            # Noise strength is independently randomized.
            strength = rng.choice(
                [0.02, 0.08, 0.20]
            )

            observed_model = noisy_distribution(
                ideal,
                strength,
                rng,
            )

            observed = sample(
                observed_model,
                rng,
            )

            fp = compute_fingerprint(
                ideal,
                observed,
            )

            # Independent latent target.
            # It is not generated from the noise fingerprint.
            latent = (
                0.7 * np.sin(theta)
                + 0.25 * np.cos(3.0 * theta)
                + 0.30 * rng.normal()
            )

            base = [
                theta,
                np.sin(theta),
                np.cos(theta),
            ]

            bases.append(base)
            fingerprints.append(fp.vector())
            targets.append(latent)

    X_base = np.asarray(bases)
    F = np.asarray(fingerprints)
    y = np.asarray(targets)

    X_real = np.hstack([X_base, F])

    rng = np.random.default_rng(12345)

    shuffled_F = F.copy()

    # Permute each fingerprint column independently.
    for j in range(shuffled_F.shape[1]):
        rng.shuffle(shuffled_F[:, j])

    X_shuffled = np.hstack([
        X_base,
        shuffled_F,
    ])

    r2_base = evaluate(X_base, y)
    r2_real = evaluate(X_real, y)
    r2_shuffled = evaluate(X_shuffled, y)

    delta_real = r2_real - r2_base
    delta_shuffled = r2_shuffled - r2_base

    result = {
        "experiment": "JNKA-EXP003",
        "samples": int(len(y)),
        "baseline_r2": r2_base,
        "real_fingerprint_r2": r2_real,
        "shuffled_fingerprint_r2": r2_shuffled,
        "delta_real": delta_real,
        "delta_shuffled": delta_shuffled,
        "ratio_real_to_shuffled": (
            delta_real / delta_shuffled
            if abs(delta_shuffled) > 1e-12
            else None
        ),
        "interpretation": (
            "A real-vs-shuffled comparison tests whether the "
            "predictive contribution survives destruction of "
            "fingerprint sample correspondence. It does not "
            "establish causality or fundamental physical information."
        ),
    }

    os.makedirs("results", exist_ok=True)

    with open(
        "results/jnka_exp003_summary.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("====================================")
    print("        JNKA EXP-003 COMPLETE")
    print("====================================")
    print("Samples                 :", len(y))
    print("Baseline R2             :", r2_base)
    print("Real fingerprint R2     :", r2_real)
    print("Shuffled fingerprint R2 :", r2_shuffled)
    print("Delta real              :", delta_real)
    print("Delta shuffled          :", delta_shuffled)
    print()
    print("JSON:")
    print("results/jnka_exp003_summary.json")


if __name__ == "__main__":
    main()
