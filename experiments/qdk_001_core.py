import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    ),
)

from quantum_gateway import (
    InformationTensor,
    entropy,
    probability_distribution,
    zz_correlation,
)


def run(name, tensor):

    info = InformationTensor(tensor)

    normalized = info.normalize()

    flat = np.abs(normalized.data.flatten())

    probabilities = probability_distribution(flat)

    result = {
        "experiment": name,
        "input": np.asarray(tensor).tolist(),
        "normalized": normalized.data.tolist(),
        "summary": normalized.summary(),
        "probabilities": probabilities.tolist(),
        "entropy": entropy(probabilities),
    }

    if len(probabilities) == 4:
        result["zz_correlation"] = zz_correlation(probabilities)

    return result


def main():

    experiments = {
        "QDK-001-A-STRUCTURED": [
            [1.0, 0.8],
            [0.8, 1.0],
        ],
        "QDK-001-B-WEAK": [
            [1.0, 0.1],
            [0.1, 1.0],
        ],
        "QDK-001-C-ASYMMETRIC": [
            [1.0, 0.2],
            [0.7, 1.0],
        ],
    }

    os.makedirs("results/qdk_001", exist_ok=True)

    for name, tensor in experiments.items():

        result = run(name, tensor)

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(json.dumps(result, indent=2))

        path = f"results/qdk_001/{name}.json"

        with open(path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
