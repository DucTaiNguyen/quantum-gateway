import itertools
import json
from pathlib import Path

import numpy as np


OUT = Path("results/qdk_014")
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURES = {
    "STRUCTURED": np.array([[1.0, 0.8], [0.8, 1.0]], dtype=float),
    "WEAK": np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    "ASYMMETRIC": np.array([[1.0, 0.2], [0.7, 1.0]], dtype=float),
}

PERMUTATIONS = list(itertools.permutations(range(4)))


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
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


X = pauli_x()
Y = pauli_y()
Z = pauli_z()
I = np.eye(2, dtype=complex)

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)


def expectation(rho, operator):
    return float(np.real_if_close(np.trace(rho @ operator)))


def amplitude_damping(rho, p):
    e0 = np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]],
        dtype=complex,
    )
    e1 = np.array(
        [[0.0, np.sqrt(p)], [0.0, 0.0]],
        dtype=complex,
    )

    result = np.zeros_like(rho, dtype=complex)

    for a in (e0, e1):
        for b in (e0, e1):
            e = np.kron(a, b)
            result += e @ rho @ e.conj().T

    return result


def phase_damping(rho, p):
    # Dephasing channel:
    # E0 = sqrt(1-p) I
    # E1 = sqrt(p) Z
    e0 = np.sqrt(1.0 - p) * I
    e1 = np.sqrt(p) * Z

    result = np.zeros_like(rho, dtype=complex)

    for a in (e0, e1):
        for b in (e0, e1):
            e = np.kron(a, b)
            result += e @ rho @ e.conj().T

    return result


def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    eigvals = eigvals[eigvals > 1e-12]

    if len(eigvals) == 0:
        return 0.0

    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def fidelity_with_initial(rho0, rho):
    value = np.trace(rho0 @ rho @ rho0)
    return float(np.sqrt(max(0.0, np.real(value))))


def encode(tensor, permutation):
    values = tensor.flatten()[list(permutation)]
    return normalize_state(values)


def observable_vector(rho, rho0):
    return np.array(
        [
            expectation(rho, XX),
            expectation(rho, YY),
            expectation(rho, ZZ),
            entropy(rho),
            purity(rho),
            fidelity_with_initial(rho0, rho),
        ],
        dtype=float,
    )


def trajectory(tensor, permutation, channel):
    state = encode(tensor, permutation)
    rho0 = density_matrix(state)

    rows = []

    for p in np.linspace(0.0, 1.0, 21):
        if channel == "amplitude_damping":
            rho = amplitude_damping(rho0, p)
        elif channel == "phase_damping":
            rho = phase_damping(rho0, p)
        else:
            raise ValueError(channel)

        rows.append(observable_vector(rho, rho0))

    matrix = np.asarray(rows)

    # Column-wise scale normalization.
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale < 1e-12] = 1.0

    return matrix / scale


def distance(a, b):
    return float(np.linalg.norm(a - b))


def main():
    channels = ["amplitude_damping", "phase_damping"]

    data = {}

    for channel in channels:
        data[channel] = {}

        for structure, tensor in STRUCTURES.items():
            data[channel][structure] = {}

            for permutation in PERMUTATIONS:
                key = "".join(map(str, permutation))

                data[channel][structure][key] = trajectory(
                    tensor,
                    permutation,
                    channel,
                ).tolist()

    within = {}
    between = {}
    ratios = {}

    structure_names = list(STRUCTURES)

    for channel in channels:
        within[channel] = {}
        between[channel] = {}
        ratios[channel] = {}

        # Within-structure distances across all 24 encodings.
        for structure in STRUCTURES:
            distances = []

            names = list(data[channel][structure])

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = np.asarray(data[channel][structure][names[i]])
                    b = np.asarray(data[channel][structure][names[j]])

                    distances.append(distance(a, b))

            within[channel][structure] = {
                "min": float(np.min(distances)),
                "median": float(np.median(distances)),
                "mean": float(np.mean(distances)),
                "max": float(np.max(distances)),
                "n_pairs": len(distances),
            }

        # Between-structure distances.
        for i in range(len(structure_names)):
            for j in range(i + 1, len(structure_names)):
                a_name = structure_names[i]
                b_name = structure_names[j]

                distances = []

                for perm in PERMUTATIONS:
                    key = "".join(map(str, perm))

                    a = np.asarray(data[channel][a_name][key])
                    b = np.asarray(data[channel][b_name][key])

                    distances.append(distance(a, b))

                key = f"{a_name}_vs_{b_name}"

                between[channel][key] = {
                    "min": float(np.min(distances)),
                    "median": float(np.median(distances)),
                    "mean": float(np.mean(distances)),
                    "max": float(np.max(distances)),
                    "n_encodings": len(distances),
                }

        max_within = max(
            x["max"] for x in within[channel].values()
        )

        min_between = min(
            x["min"] for x in between[channel].values()
        )

        ratio = (
            min_between / max_within
            if max_within > 0
            else float("inf")
        )

        ratios[channel] = {
            "min_between": min_between,
            "max_within": max_within,
            "robustness_ratio": ratio,
        }

    output = {
        "experiment": "QDK-014",
        "title": "Exhaustive Encoding Robustness Test",
        "n_structures": len(STRUCTURES),
        "n_encodings": len(PERMUTATIONS),
        "n_encoding_pairs_per_structure": 276,
        "channels": channels,
        "within_structure": within,
        "between_structure": between,
        "robustness": ratios,
        "interpretation": (
            "Descriptive robustness analysis. "
            "A high robustness ratio indicates that "
            "between-structure separation exceeds "
            "within-structure encoding variation under "
            "the tested reference model."
        ),
        "warning": (
            "Numerical reference experiment only; "
            "not a physical theorem or statistical "
            "significance test."
        ),
    }

    path = OUT / "QDK-014-EXHAUSTIVE-ENCODING.json"

    with path.open("w") as f:
        json.dump(output, f, indent=2)

    print("QDK-014 COMPLETE")
    print("=" * 60)

    for channel in channels:
        print()
        print(channel)
        print("-" * 60)

        print("WITHIN-STRUCTURE")

        for structure, values in within[channel].items():
            print(
                f"{structure}: "
                f"min={values['min']:.12f}, "
                f"median={values['median']:.12f}, "
                f"max={values['max']:.12f}"
            )

        print()
        print("BETWEEN-STRUCTURE")

        for key, values in between[channel].items():
            print(
                f"{key}: "
                f"min={values['min']:.12f}, "
                f"median={values['median']:.12f}, "
                f"max={values['max']:.12f}"
            )

        print()
        print("ROBUSTNESS RATIO")

        values = ratios[channel]

        print(
            f"min_between = "
            f"{values['min_between']:.12f}"
        )

        print(
            f"max_within = "
            f"{values['max_within']:.12f}"
        )

        print(
            f"ratio = "
            f"{values['robustness_ratio']:.12f}"
        )

    print()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
