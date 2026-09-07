import itertools
import json
from pathlib import Path

import numpy as np


OUT = Path("results/qdk_015")
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURES = {
    "STRUCTURED": np.array([[1.0, 0.8], [0.8, 1.0]], dtype=float),
    "WEAK": np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    "ASYMMETRIC": np.array([[1.0, 0.2], [0.7, 1.0]], dtype=float),
}

PERMUTATIONS = list(itertools.permutations(range(4)))


def normalize_state(values):
    values = np.asarray(values, dtype=complex)
    norm = np.linalg.norm(values)

    if norm == 0:
        raise ValueError("Cannot normalize zero state.")

    return values / norm


def density_matrix(state):
    state = normalize_state(state)
    return np.outer(state, state.conjugate())


def permutation_matrix(permutation):
    dimension = len(permutation)
    P = np.zeros((dimension, dimension), dtype=complex)

    for i, j in enumerate(permutation):
        P[j, i] = 1.0

    return P


def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


X = pauli_x()
Y = pauli_y()
Z = pauli_z()

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)

OBSERVABLES = {
    "XX": XX,
    "YY": YY,
    "ZZ": ZZ,
}


def expectation(rho, operator):
    value = np.trace(rho @ operator)
    return float(np.real_if_close(value))


def encode(tensor, permutation):
    values = tensor.flatten()[list(permutation)]
    return normalize_state(values)


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
    e0 = np.sqrt(1.0 - p) * np.eye(2, dtype=complex)
    e1 = np.sqrt(p) * Z

    result = np.zeros_like(rho, dtype=complex)

    for a in (e0, e1):
        for b in (e0, e1):
            e = np.kron(a, b)
            result += e @ rho @ e.conj().T

    return result


def covariance_test(rho, P, operator):
    rho_transformed = P @ rho @ P.conj().T
    operator_transformed = P @ operator @ P.conj().T

    original = expectation(rho, operator)

    fixed = expectation(
        rho_transformed,
        operator,
    )

    covariant = expectation(
        rho_transformed,
        operator_transformed,
    )

    return {
        "original": original,
        "fixed_observable": fixed,
        "covariant_observable": covariant,
        "fixed_error": abs(fixed - original),
        "covariant_error": abs(covariant - original),
    }


def noise_covariance_test(rho, P, operator, channel, p):
    rho_transformed = P @ rho @ P.conj().T
    operator_transformed = P @ operator @ P.conj().T

    if channel == "amplitude_damping":
        noisy_original = amplitude_damping(rho, p)
        noisy_transformed = amplitude_damping(
            rho_transformed,
            p,
        )

    elif channel == "phase_damping":
        noisy_original = phase_damping(rho, p)
        noisy_transformed = phase_damping(
            rho_transformed,
            p,
        )

    else:
        raise ValueError(channel)

    original_value = expectation(
        noisy_original,
        operator,
    )

    fixed_value = expectation(
        noisy_transformed,
        operator,
    )

    covariant_value = expectation(
        noisy_transformed,
        operator_transformed,
    )

    return {
        "original": original_value,
        "fixed": fixed_value,
        "covariant": covariant_value,
        "fixed_error": abs(
            fixed_value - original_value
        ),
        "covariant_error": abs(
            covariant_value - original_value
        ),
    }


def main():
    channels = [
        "amplitude_damping",
        "phase_damping",
    ]

    results = {
        "experiment": "QDK-015",
        "title": "Representation-Covariant Fingerprint Test",
        "n_structures": len(STRUCTURES),
        "n_permutations": len(PERMUTATIONS),
        "observables": list(OBSERVABLES),
        "channels": channels,
        "covariance": {},
        "noise_covariance": {},
        "summary": {},
        "warning": (
            "Numerical reference experiment only. "
            "Covariance conclusions apply only to "
            "the tested transformations and channels."
        ),
    }

    # ---------------------------------------------------------
    # Part A: noiseless representation covariance
    # ---------------------------------------------------------

    fixed_errors = []
    covariant_errors = []

    for structure_name, tensor in STRUCTURES.items():
        results["covariance"][structure_name] = {}

        for permutation in PERMUTATIONS:
            permutation_key = "".join(map(str, permutation))

            state = encode(
                tensor,
                permutation,
            )

            rho = density_matrix(state)
            P = permutation_matrix(permutation)

            results["covariance"][structure_name][
                permutation_key
            ] = {}

            for observable_name, operator in OBSERVABLES.items():
                result = covariance_test(
                    rho,
                    P,
                    operator,
                )

                results["covariance"][structure_name][
                    permutation_key
                ][observable_name] = result

                fixed_errors.append(
                    result["fixed_error"]
                )

                covariant_errors.append(
                    result["covariant_error"]
                )

    # ---------------------------------------------------------
    # Part B: noisy covariance
    # ---------------------------------------------------------

    for channel in channels:
        results["noise_covariance"][channel] = {}

        channel_fixed_errors = []
        channel_covariant_errors = []

        for structure_name, tensor in STRUCTURES.items():
            results["noise_covariance"][channel][
                structure_name
            ] = {}

            # Use the original encoding as the physical
            # reference representation.
            permutation = (0, 1, 2, 3)

            state = encode(
                tensor,
                permutation,
            )

            rho = density_matrix(state)

            for perm in PERMUTATIONS:
                permutation_key = "".join(map(str, perm))

                P = permutation_matrix(perm)
                rho_perm = P @ rho @ P.conj().T

                results["noise_covariance"][channel][
                    structure_name
                ][permutation_key] = {}

                for p in np.linspace(0.0, 1.0, 21):
                    for observable_name, operator in OBSERVABLES.items():
                        value = noise_covariance_test(
                            rho,
                            P,
                            operator,
                            channel,
                            p,
                        )

                        key = (
                            f"{permutation_key}:"
                            f"{observable_name}:"
                            f"{p:.2f}"
                        )

                        results["noise_covariance"][channel][
                            structure_name
                        ][permutation_key][key] = value

                        channel_fixed_errors.append(
                            value["fixed_error"]
                        )

                        channel_covariant_errors.append(
                            value["covariant_error"]
                        )

        results["summary"][channel] = {
            "max_fixed_error": float(
                np.max(channel_fixed_errors)
            ),
            "mean_fixed_error": float(
                np.mean(channel_fixed_errors)
            ),
            "max_covariant_error": float(
                np.max(channel_covariant_errors)
            ),
            "mean_covariant_error": float(
                np.mean(channel_covariant_errors)
            ),
        }

    results["summary"]["noiseless"] = {
        "max_fixed_error": float(np.max(fixed_errors)),
        "mean_fixed_error": float(np.mean(fixed_errors)),
        "max_covariant_error": float(
            np.max(covariant_errors)
        ),
        "mean_covariant_error": float(
            np.mean(covariant_errors)
        ),
    }

    path = OUT / "QDK-015-REPRESENTATION-COVARIANCE.json"

    with path.open("w") as f:
        json.dump(results, f, indent=2)

    print("QDK-015 COMPLETE")
    print("=" * 60)

    print()
    print("NOISELESS REPRESENTATION COVARIANCE")
    print("-" * 60)

    print(
        "max fixed-observable error: "
        f"{results['summary']['noiseless']['max_fixed_error']:.12e}"
    )

    print(
        "max covariant-observable error: "
        f"{results['summary']['noiseless']['max_covariant_error']:.12e}"
    )

    print()
    print("NOISY REPRESENTATION COVARIANCE")
    print("-" * 60)

    for channel in channels:
        values = results["summary"][channel]

        print()
        print(channel)

        print(
            "max fixed error: "
            f"{values['max_fixed_error']:.12e}"
        )

        print(
            "mean fixed error: "
            f"{values['mean_fixed_error']:.12e}"
        )

        print(
            "max covariant error: "
            f"{values['max_covariant_error']:.12e}"
        )

        print(
            "mean covariant error: "
            f"{values['mean_covariant_error']:.12e}"
        )

    print()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
