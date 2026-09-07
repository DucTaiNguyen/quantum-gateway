import json
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    ),
)

from quantum_gateway import tensor_to_state


def run(name, tensor):

    state = tensor_to_state(tensor)

    result = {
        "experiment": name,
        "input": tensor,
        "state": {
            "amplitudes": [
                {
                    "real": float(a.real),
                    "imag": float(a.imag),
                }
                for a in state.amplitudes
            ],
            "probabilities": state.probabilities().real.tolist(),
            "dimension": state.dimension,
            "entropy": state.entropy(),
            "norm": float(
                __import__("numpy").linalg.norm(
                    state.amplitudes
                )
            ),
        },
    }

    return result


def main():

    experiments = {
        "QDK-002-A-STRUCTURED": [
            [1.0, 0.8],
            [0.8, 1.0],
        ],
        "QDK-002-B-WEAK": [
            [1.0, 0.1],
            [0.1, 1.0],
        ],
        "QDK-002-C-ASYMMETRIC": [
            [1.0, 0.2],
            [0.7, 1.0],
        ],
    }

    os.makedirs(
        "results/qdk_002",
        exist_ok=True,
    )

    for name, tensor in experiments.items():

        result = run(name, tensor)

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(json.dumps(result, indent=2))

        with open(
            f"results/qdk_002/{name}.json",
            "w",
        ) as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
