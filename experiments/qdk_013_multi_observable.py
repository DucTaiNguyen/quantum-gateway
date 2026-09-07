import json
from pathlib import Path

import numpy as np


OUT = Path("results/qdk_013")
OUT.mkdir(parents=True, exist_ok=True)


STRUCTURES = {
    "STRUCTURED": np.array(
        [[1.0, 0.8], [0.8, 1.0]],
        dtype=float,
    ),
    "WEAK": np.array(
        [[1.0, 0.2], [0.2, 1.0]],
        dtype=float,
    ),
    "ASYMMETRIC": np.array(
        [[1.0, 0.2], [0.7, 1.0]],
        dtype=float,
    ),
}


PERMUTATIONS = {
    "A_ORIGINAL": np.array([0, 1, 2, 3]),
    "B_PERMUTED": np.array([0, 2, 1, 3]),
    "C_REVERSED": np.array([3, 2, 1, 0]),
}


def normalize_state(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)

    if norm == 0:
        raise ValueError("Cannot normalize zero state.")

    return values / norm


def density_matrix(state):
    state = normalize_state(state)
    return np.outer(state, state.conjugate())


def pauli_x():
    return np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )


def pauli_y():
    return np.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )


def pauli_z():
    return np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )


def kron(a, b):
    return np.kron(a, b)


I = np.eye(2, dtype=complex)
X = pauli_x()
Y = pauli_y()
Z = pauli_z()

XX = kron(X, X)
YY = kron(Y, Y)
ZZ = kron(Z, Z)


def expectation(rho, operator):
    value = np.trace(rho @ operator)
    return float(np.real_if_close(value))


def amplitude_damping(rho, p):
    p = float(np.clip(p, 0.0, 1.0))

    e0 = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(1.0 - p)],
        ],
        dtype=complex,
    )

    e1 = np.array(
        [
            [0.0, np.sqrt(p)],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    kraus = [
        kron(e0, e0),
        kron(e0, e1),
        kron(e1, e0),
        kron(e1, e1),
    ]

    result = np.zeros_like(rho, dtype=complex)

    for e in kraus:
        result += e @ rho @ e.conj().T

    return result


def phase_damping(rho, p):
    p = float(np.clip(p, 0.0, 1.0))

    e0 = np.sqrt(1.0 - p) * I
    e1 = np.sqrt(p) * Z

    kraus_single = [e0, e1]

    result = np.zeros_like(rho, dtype=complex)

    for a in kraus_single:
        for b in kraus_single:
            e = kron(a, b)
            result += 0.25 * e @ rho @ e.conj().T

    return result


def entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)

    nonzero = eigenvalues[eigenvalues > 1e-12]

    if len(nonzero) == 0:
        return 0.0

    return float(-np.sum(nonzero * np.log2(nonzero)))


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def fidelity_with_initial(rho_initial, rho):
    product = rho_initial @ rho @ rho_initial

    value = np.trace(product)

    return float(np.sqrt(max(0.0, np.real(value))))


def encode_tensor(tensor, permutation):
    values = tensor.flatten()
    values = values[permutation]

    return normalize_state(values)


def observables(rho, rho_initial):
    return {
        "XX": expectation(rho, XX),
        "YY": expectation(rho, YY),
        "ZZ": expectation(rho, ZZ),
        "entropy": entropy(rho),
        "purity": purity(rho),
        "fidelity": fidelity_with_initial(
            rho_initial,
            rho,
        ),
    }


def trajectory(tensor, permutation, channel):
    state = encode_tensor(tensor, permutation)

    rho_initial = density_matrix(state)

    p_values = np.linspace(0.0, 1.0, 21)

    rows = []

    for p in p_values:
        if channel == "amplitude_damping":
            rho = amplitude_damping(
                rho_initial,
                p,
            )
        elif channel == "phase_damping":
            rho = phase_damping(
                rho_initial,
                p,
            )
        else:
            raise ValueError(
                f"Unknown channel: {channel}"
            )

        values = observables(
            rho,
            rho_initial,
        )

        rows.append(values)

    return p_values, rows


def vectorize(rows):
    keys = [
        "XX",
        "YY",
        "ZZ",
        "entropy",
        "purity",
        "fidelity",
    ]

    return np.array(
        [
            [row[key] for key in keys]
            for row in rows
        ],
        dtype=float,
    )


def normalize_trajectory(matrix):
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale < 1e-12] = 1.0

    return matrix / scale


def trajectory_distance(a, b):
    return float(np.linalg.norm(a - b))


def main():
    results = {
        "experiment": "QDK-013",
        "title": (
            "Multi-Observable Non-Factorizable "
            "Noise Response"
        ),
        "channels": {},
        "cross_encoding": {},
        "structure_separation": {},
        "warning": (
            "Numerical reference experiment. "
            "Not a physical theorem."
        ),
    }

    for channel in [
        "amplitude_damping",
        "phase_damping",
    ]:
        results["channels"][channel] = {}

        for structure_name, tensor in STRUCTURES.items():
            results["channels"][channel][structure_name] = {}

            for encoding_name, permutation in PERMUTATIONS.items():
                p_values, rows = trajectory(
                    tensor,
                    permutation,
                    channel,
                )

                matrix = vectorize(rows)

                normalized = normalize_trajectory(
                    matrix
                )

                results["channels"][channel][
                    structure_name
                ][encoding_name] = {
                    "p": p_values.tolist(),
                    "observables": rows,
                    "normalized_matrix": normalized.tolist(),
                }

    for channel in results["channels"]:
        results["cross_encoding"][channel] = {}

        for structure_name in STRUCTURES:
            entries = results["channels"][channel][
                structure_name
            ]

            names = list(PERMUTATIONS)

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = names[i]
                    b = names[j]

                    xa = np.array(
                        entries[a]["normalized_matrix"]
                    )

                    xb = np.array(
                        entries[b]["normalized_matrix"]
                    )

                    key = (
                        f"{structure_name}:"
                        f"{a}_vs_{b}"
                    )

                    results["cross_encoding"][channel][
                        key
                    ] = {
                        "euclidean_distance": (
                            trajectory_distance(
                                xa,
                                xb,
                            )
                        ),
                        "max_absolute_difference": float(
                            np.max(
                                np.abs(xa - xb)
                            )
                        ),
                    }

    for channel in results["channels"]:
        results["structure_separation"][channel] = {}

        for encoding_name in PERMUTATIONS:
            entries = results["channels"][channel]

            names = list(STRUCTURES)

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = names[i]
                    b = names[j]

                    xa = np.array(
                        entries[a][encoding_name][
                            "normalized_matrix"
                        ]
                    )

                    xb = np.array(
                        entries[b][encoding_name][
                            "normalized_matrix"
                        ]
                    )

                    key = (
                        f"{encoding_name}:"
                        f"{a}_vs_{b}"
                    )

                    results["structure_separation"][
                        channel
                    ][key] = {
                        "euclidean_distance": (
                            trajectory_distance(
                                xa,
                                xb,
                            )
                        ),
                        "max_absolute_difference": float(
                            np.max(
                                np.abs(xa - xb)
                            )
                        ),
                    }

    output = OUT / (
        "QDK-013-MULTI-OBSERVABLE.json"
    )

    with output.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("QDK-013 COMPLETE")
    print()
    print("CROSS-ENCODING ROBUSTNESS")
    print("=" * 50)

    for channel, data in results[
        "cross_encoding"
    ].items():
        print()
        print(channel)

        for key, value in data.items():
            print(
                f"{key}: "
                f"{value['euclidean_distance']:.12f}"
            )

    print()
    print("STRUCTURE SEPARATION")
    print("=" * 50)

    for channel, data in results[
        "structure_separation"
    ].items():
        print()
        print(channel)

        for key, value in data.items():
            print(
                f"{key}: "
                f"{value['euclidean_distance']:.12f}"
            )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
