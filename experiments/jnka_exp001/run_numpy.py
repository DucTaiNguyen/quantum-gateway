import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.jnka import compute_fingerprint, compute_utility, decide


N_QUBITS = 4
N_STATES = 2 ** N_QUBITS
SHOTS = 4096
SEEDS = range(10)
THETAS = np.linspace(0.0, 2.0 * math.pi, 33)


def ideal_distribution(theta):
    """
    Four-qubit reference distribution.

    The first three qubits form a structured correlation term.
    The fourth qubit carries theta-dependent information.
    """
    p = np.zeros(N_STATES, dtype=float)

    a = math.cos(theta / 2.0) ** 2
    b = math.sin(theta / 2.0) ** 2

    # Structured states:
    # 0000, 0111, 1000, 1111
    p[0] = 0.35 * a
    p[7] = 0.35 * a
    p[8] = 0.15 * b
    p[15] = 0.15 * b

    return p / p.sum()


def noisy_distribution(p, noise_strength, rng):
    """
    Synthetic operational noise model.

    This is deliberately NOT claimed to reproduce
    physical IBM hardware noise.
    """
    uniform = np.ones_like(p) / len(p)

    # Probability redistribution.
    q = (1.0 - noise_strength) * p + noise_strength * uniform

    # Small stochastic perturbation.
    fluctuation = rng.normal(0.0, noise_strength * 0.01, size=len(p))
    q = np.clip(q + fluctuation, 0.0, None)

    return q / q.sum()


def sample_distribution(p, shots, rng):
    counts = rng.multinomial(shots, p)
    return counts / shots


def run():
    rows = []

    noise_levels = {
        "low": 0.02,
        "medium": 0.08,
        "high": 0.20,
    }

    experiment_id = 0

    for theta in THETAS:
        ideal = ideal_distribution(theta)

        for seed in SEEDS:
            for condition, strength in noise_levels.items():
                rng = np.random.default_rng(seed)

                observed_model = noisy_distribution(
                    ideal,
                    strength,
                    rng,
                )

                observed = sample_distribution(
                    observed_model,
                    SHOTS,
                    rng,
                )

                fp = compute_fingerprint(
                    ideal,
                    observed,
                )

                # EXP-001 is characterization-only.
                # No downstream learning target is used yet.
                utility = compute_utility(
                    learning_gain=0.0,
                    representation_gain=0.0,
                    cost=fp.tv,
                )

                action = decide(
                    utility=utility,
                    tv=fp.tv,
                )

                rows.append({
                    "experiment_id": experiment_id,
                    "theta": float(theta),
                    "seed": seed,
                    "condition": condition,
                    "noise_strength": strength,
                    "kl": fp.kl,
                    "js": fp.js,
                    "tv": fp.tv,
                    "entropy": fp.entropy,
                    "l1": fp.l1,
                    "utility": utility,
                    "action": action,
                })

                experiment_id += 1

    os.makedirs("results", exist_ok=True)

    csv_path = "results/jnka_exp001_numpy.csv"
    json_path = "results/jnka_exp001_numpy_summary.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {}

    for condition in noise_levels:
        subset = [
            r for r in rows
            if r["condition"] == condition
        ]

        summary[condition] = {
            "n": len(subset),
            "mean_kl": float(np.mean([r["kl"] for r in subset])),
            "mean_js": float(np.mean([r["js"] for r in subset])),
            "mean_tv": float(np.mean([r["tv"] for r in subset])),
            "mean_entropy": float(
                np.mean([r["entropy"] for r in subset])
            ),
            "mean_l1": float(np.mean([r["l1"] for r in subset])),
        }

    result = {
        "experiment": "JNKA-EXP001",
        "purpose": "operational noise fingerprint characterization",
        "backend": "numpy_synthetic",
        "n_qubits": N_QUBITS,
        "shots": SHOTS,
        "n_thetas": len(THETAS),
        "n_seeds": len(list(SEEDS)),
        "n_conditions": len(noise_levels),
        "total_observations": len(rows),
        "warning": (
            "Synthetic noise is used for local validation only; "
            "results are not hardware evidence."
        ),
        "summary": summary,
    }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("====================================")
    print("        JNKA EXP-001 COMPLETE")
    print("====================================")
    print("Observations:", len(rows))
    print()
    print("Condition   Mean TV       Mean JS       Mean Entropy")
    print("------------------------------------------------------")

    for condition in noise_levels:
        s = summary[condition]
        print(
            f"{condition:<10}"
            f"{s['mean_tv']:<14.6f}"
            f"{s['mean_js']:<14.6f}"
            f"{s['mean_entropy']:<14.6f}"
        )

    print()
    print("CSV :", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    run()
