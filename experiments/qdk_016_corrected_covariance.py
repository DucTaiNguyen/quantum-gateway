import itertools
import json
from pathlib import Path

import numpy as np


OUT = Path("results/qdk_016")
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURES = {
    "STRUCTURED": np.array([[1.0, 0.8], [0.8, 1.0]], dtype=float),
    "WEAK": np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float),
    "ASYMMETRIC": np.array([[1.0, 0.2], [0.7, 1.0]], dtype=float),
}

PERMUTATIONS = list(itertools.permutations(range(4)))


def normalize_state(x):
    x = np.asarray(x, dtype=complex)
    return x / np.linalg.norm(x)


def density_matrix(state):
    state = normalize_state(state)
    return np.outer(state, state.conjugate())


def permutation_matrix(permutation):
    n = len(permutation)
    P = np.zeros((n, n), dtype=complex)

    for i, j in enumerate(permutation):
        P[j, i] = 1.0

    return P


def X():
    return np.array([[0, 1], [1, 0]], dtype=complex)


def Y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def Z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


XX = np.kron(X(), X())
YY = np.kron(Y(), Y())
ZZ = np.kron(Z(), Z())

OBSERVABLES = {
    "XX": XX,
    "YY": YY,
    "ZZ": ZZ,
}


def expectation(rho, operator):
    return float(
        np.real_if_close(np.trace(rho @ operator))
    )


def canonical_state(tensor):
    return normalize_state(tensor.flatten())


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
    identity = np.eye(2, dtype=complex)

    e0 = np.sqrt(1.0 - p) * identity
    e1 = np.sqrt(p) * Z()

    result = np.zeros_like(rho, dtype=complex)

    for a in (e0, e1):
        for b in (e0, e1):
            e = np.kron(a, b)
            result += e @ rho @ e.conj().T

    return result


def channel(rho, p, name):
    if name == "amplitude_damping":
        return amplitude_damping(rho, p)

    if name == "phase_damping":
        return phase_damping(rho, p)

    raise ValueError(name)


def main():
    channels = [
        "amplitude_damping",
        "phase_damping",
    ]

    results = {
        "experiment": "QDK-016",
        "title": "Corrected Representation Covariance Test",
        "n_structures": len(STRUCTURES),
        "n_permutations": len(PERMUTATIONS),
        "channels": channels,
        "noiseless": {},
        "noise_covariance": {},
        "warning": (
            "Numerical reference experiment only. "
            "Not a physical theorem."
        ),
    }

    # ---------------------------------------------------------
    # Noiseless covariance
    # ---------------------------------------------------------

    for structure_name, tensor in STRUCTURES.items():
        rho0 = density_matrix(canonical_state(tensor))

        results["noiseless"][structure_name] = {}

        for permutation in PERMUTATIONS:
            key = "".join(map(str, permutation))
            P = permutation_matrix(permutation)

            rho_p = P @ rho0 @ P.conj().T

            results["noiseless"][structure_name][key] = {}

            for obs_name, O in OBSERVABLES.items():
                O_p = P @ O @ P.conj().T

                original = expectation(rho0, O)
                transformed = expectation(rho_p, O_p)

                results["noiseless"][structure_name][key][obs_name] = {
                    "original": original,
                    "covariant": transformed,
                    "error": abs(transformed - original),
                }

    # ---------------------------------------------------------
    # Noise covariance
    # ---------------------------------------------------------

    for channel_name in channels:
        results["noise_covariance"][channel_name] = {}

        fixed_errors = []
        covariant_errors = []
        channel_errors = []

        for structure_name, tensor in STRUCTURES.items():
            rho0 = density_matrix(canonical_state(tensor))

            results["noise_covariance"][channel_name][
                structure_name
            ] = {}

            for permutation in PERMUTATIONS:
                key = "".join(map(str, permutation))
                P = permutation_matrix(permutation)

                rho_p = P @ rho0 @ P.conj().T

                results["noise_covariance"][channel_name][
                    structure_name
                ][key] = {}

                for p in np.linspace(0.0, 1.0, 21):
                    rho_noisy = channel(
                        rho0,
                        p,
                        channel_name,
                    )

                    # Apply same physical channel after changing basis.
                    rho_p_noisy = channel(
                        rho_p,
                        p,
                        channel_name,
                    )

                    # Representation-transformed reference.
                    rho_reference = (
                        P
                        @ rho_noisy
                        @ P.conj().T
                    )

                    covariance_error = np.linalg.norm(
                        rho_p_noisy - rho_reference
                    )

                    channel_errors.append(
                        covariance_error
                    )

                    for obs_name, O in OBSERVABLES.items():
                        O_p = P @ O @ P.conj().T

                        original = expectation(
                            rho_noisy,
                            O,
                        )

                        fixed = expectation(
                            rho_p_noisy,
                            O,
                        )

                        covariant = expectation(
                            rho_p_noisy,
                            O_p,
                        )

                        fixed_error = abs(
                            fixed - original
                        )

                        covariant_error = abs(
                            covariant - original
                        )

                        fixed_errors.append(
                            fixed_error
                        )

                        covariant_errors.append(
                            covariant_error
                        )

                        results["noise_covariance"][
                            channel_name
                        ][structure_name][key][
                            f"{obs_name}:{p:.2f}"
                        ] = {
                            "fixed_error": fixed_error,
                            "covariant_observable_error": (
                                covariant_error
                            ),
                            "channel_covariance_error": (
                                covariance_error
                            ),
                        }

        results["noise_covariance"][
            channel_name
        ]["_summary"] = {
            "max_fixed_observable_error": float(
                np.max(fixed_errors)
            ),
            "mean_fixed_observable_error": float(
                np.mean(fixed_errors)
            ),
            "max_covariant_observable_error": float(
                np.max(covariant_errors)
            ),
            "mean_covariant_observable_error": float(
                np.mean(covariant_errors)
            ),
            "max_channel_covariance_error": float(
                np.max(channel_errors)
            ),
            "mean_channel_covariance_error": float(
                np.mean(channel_errors)
            ),
        }

    output = OUT / "QDK-016-CORRECTED-COVARIANCE.json"

    with output.open("w") as f:
        json.dump(results, f, indent=2)

    print("QDK-016 COMPLETE")
    print("=" * 60)

    print()
    print("NOISELESS COVARIANCE")
    print("-" * 60)

    errors = []

    for structure in results["noiseless"].values():
        for permutation in structure.values():
            for value in permutation.values():
                errors.append(value["error"])

    print(
        "max covariance error: "
        f"{max(errors):.12e}"
    )

    print()
    print("NOISE CHANNEL COVARIANCE")
    print("-" * 60)

    for name in channels:
        summary = results["noise_covariance"][name]["_summary"]

        print()
        print(name)

        print(
            "max fixed-observable error: "
            f"{summary['max_fixed_observable_error']:.12e}"
        )

        print(
            "max covariant-observable error: "
            f"{summary['max_covariant_observable_error']:.12e}"
        )

        print(
            "max channel covariance error: "
            f"{summary['max_channel_covariance_error']:.12e}"
        )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
