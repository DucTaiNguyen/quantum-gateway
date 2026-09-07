import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
        )
    ),
)

from quantum_gateway import tensor_to_state
from quantum_gateway.evolution import (
    evolve_product_state,
    entanglement_entropy,
)


def run(name, tensor, theta):

    state = tensor_to_state(tensor)

    evolved = evolve_product_state(
        state.amplitudes,
        theta=theta,
    )

    norm = np.linalg.norm(evolved)

    entropy = entanglement_entropy(evolved)

    result = {
        "experiment": name,
        "theta": theta,
        "initial_state": [
            {
                "real": float(a.real),
                "imag": float(a.imag),
            }
            for a in state.amplitudes
        ],
        "evolved_state": [
            {
                "real": float(a.real),
                "imag": float(a.imag),
            }
            for a in evolved
        ],
        "norm": float(norm),
        "entanglement_entropy": entropy,
    }

    return result


def main():

    experiments = {
        "QDK-003-A-STRUCTURED": [
            [1.0, 0.8],
            [0.8, 1.0],
        ],
        "QDK-003-B-WEAK": [
            [1.0, 0.1],
            [0.1, 1.0],
        ],
        "QDK-003-C-ASYMMETRIC": [
            [1.0, 0.2],
            [0.7, 1.0],
        ],
    }

    os.makedirs(
        "results/qdk_003",
        exist_ok=True,
    )

    for name, tensor in experiments.items():

        result = run(
            name,
            tensor,
            theta=np.pi / 4,
        )

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        with open(
            f"results/qdk_003/{name}.json",
            "w",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
