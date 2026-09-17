import csv
import json
import os
import sys
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath("."))

from src.jnka import compute_fingerprint


SEEDS = range(20)
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
    counts = rng.multinomial(SHOTS, p)
    return counts / SHOTS


def target(theta, rng):
    """
    Independent downstream observable.

    It is NOT constructed from the noise fingerprint.
    """
    return (
        0.8 * np.sin(theta)
        + 0.35 * np.cos(2.0 * theta)
        + rng.normal(0.0, 0.08)
    )


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

    return float(np.mean(scores)), float(np.std(scores))


def main():
    base_rows = []
    augmented_rows = []

    for theta in THETAS:
        for seed in SEEDS:

            rng = np.random.default_rng(seed)

            ideal = ideal_distribution(theta)

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

            y = target(theta, rng)

            base = [
                theta,
                np.sin(theta),
                np.cos(theta),
            ]

            noise_features = fp.vector()

            augmented = base + list(noise_features)

            base_rows.append(base)
            augmented_rows.append(augmented)

            if "ys" not in locals():
                ys = []

            ys.append(y)

    X_base = np.asarray(base_rows)
    X_augmented = np.asarray(augmented_rows)
    y = np.asarray(ys)

    base_r2, base_std = evaluate(
        X_base,
        y,
    )

    augmented_r2, augmented_std = evaluate(
        X_augmented,
        y,
    )

    delta_r2 = augmented_r2 - base_r2

    result = {
        "experiment": "JNKA-EXP002",
        "purpose": "test predictive contribution of noise fingerprint",
        "n_samples": int(len(y)),
        "shots": SHOTS,
        "base_features": [
            "theta",
            "sin(theta)",
            "cos(theta)",
        ],
        "noise_features": [
            "KL",
            "JS",
            "TV",
            "entropy",
            "L1",
        ],
        "baseline_r2": base_r2,
        "baseline_r2_std": base_std,
        "augmented_r2": augmented_r2,
        "augmented_r2_std": augmented_std,
        "delta_r2": delta_r2,
        "interpretation": (
            "Delta R2 measures incremental predictive association "
            "under this experimental design; it does not establish "
            "causality or fundamental physical information."
        ),
    }

    os.makedirs("results", exist_ok=True)

    with open(
        "results/jnka_exp002_summary.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("====================================")
    print("        JNKA EXP-002 COMPLETE")
    print("====================================")
    print("Samples       :", len(y))
    print("Baseline R2   :", base_r2)
    print("Augmented R2  :", augmented_r2)
    print("Delta R2      :", delta_r2)
    print()
    print("JSON:")
    print("results/jnka_exp002_summary.json")


if __name__ == "__main__":
    main()
