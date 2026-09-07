import json
from pathlib import Path

import numpy as np


OUT = Path("results/qdk_012")
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURES = {
    "STRUCTURED": np.array([[1.0, 0.8], [0.8, 1.0]], dtype=float),
    "WEAK": np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    "ASYMMETRIC": np.array([[1.0, 0.2], [0.7, 1.0]], dtype=float),
}


def normalize(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return x / norm


def zz_correlation(probabilities):
    z = np.array([1.0, -1.0, -1.0, 1.0])
    return float(np.dot(probabilities, z))


def trajectory(tensor, permutation):
    values = tensor.flatten()
    encoded = values[permutation]
    encoded = normalize(encoded)

    probabilities = encoded**2
    initial = zz_correlation(probabilities)

    p_values = np.linspace(0.0, 1.0, 21)
    response = initial * (1.0 - p_values**2)

    return p_values, response


def main():
    permutations = {
        "A_ORIGINAL": np.array([0, 1, 2, 3]),
        "B_PERMUTED": np.array([0, 2, 1, 3]),
        "C_REVERSED": np.array([3, 2, 1, 0]),
    }

    results = {
        "experiment": "QDK-012",
        "title": "Cross-Encoding Structural Fingerprint Test",
        "model": (
            "Reference amplitude encoding with nonlinear "
            "effective noise q=p^2."
        ),
        "encodings": {},
        "pairwise": {},
        "warning": (
            "Reference-model robustness experiment only; "
            "not a universal physical theorem."
        ),
    }

    trajectories = {}

    for structure_name, tensor in STRUCTURES.items():
        trajectories[structure_name] = {}

        for encoding_name, permutation in permutations.items():
            p, response = trajectory(tensor, permutation)

            trajectories[structure_name][encoding_name] = {
                "p": p.tolist(),
                "response": response.tolist(),
                "initial_amplitude": float(response[0]),
            }

    for structure_name in STRUCTURES:
        results["encodings"][structure_name] = {}

        for encoding_name in permutations:
            response = np.asarray(
                trajectories[structure_name][encoding_name]["response"]
            )

            normalized_response = response / abs(response[0])

            results["encodings"][structure_name][encoding_name] = {
                "initial_amplitude": float(response[0]),
                "normalized_response": normalized_response.tolist(),
            }

    for structure_name in STRUCTURES:
        names = list(permutations)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]

                xa = np.asarray(
                    results["encodings"][structure_name][a][
                        "normalized_response"
                    ]
                )
                xb = np.asarray(
                    results["encodings"][structure_name][b][
                        "normalized_response"
                    ]
                )

                key = f"{structure_name}:{a}_vs_{b}"

                results["pairwise"][key] = {
                    "euclidean_distance": float(np.linalg.norm(xa - xb)),
                    "max_absolute_difference": float(
                        np.max(np.abs(xa - xb))
                    ),
                }

    output = OUT / "QDK-012-CROSS-ENCODING.json"

    with output.open("w") as f:
        json.dump(results, f, indent=2)

    print("QDK-012 COMPLETE")
    print()
    print("Cross-encoding normalized distances:")

    for key, value in results["pairwise"].items():
        print(f"{key}: {value['euclidean_distance']:.12f}")

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
