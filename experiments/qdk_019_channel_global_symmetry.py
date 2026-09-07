import json
from itertools import permutations
from pathlib import Path

import numpy as np


TOL = 1e-12


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [I, X, Y, Z]


def kron(a, b):
    return np.kron(a, b)


def operator_basis():
    return [
        kron(a, b)
        for a in PAULIS
        for b in PAULIS
    ]


def permutation_unitary(perm):
    U = np.zeros((4, 4), dtype=complex)
    for i, j in enumerate(perm):
        U[j, i] = 1.0
    return U


def amplitude_damping(rho, p):
    p = float(p)

    k0 = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(1.0 - p)],
        ],
        dtype=complex,
    )

    k1 = np.array(
        [
            [0.0, np.sqrt(p)],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    K0 = kron(k0, I)
    K1 = kron(k1, I)

    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def phase_damping(rho, p):
    p = float(p)

    k0 = np.sqrt(1.0 - p) * I
    k1 = np.sqrt(p) * Z

    K0 = kron(k0, I)
    K1 = kron(k1, I)

    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def channel_covariance_error(channel, U, p, basis):
    max_error = 0.0

    for A in basis:
        lhs = channel(U @ A @ U.conj().T, p)
        rhs = U @ channel(A, p) @ U.conj().T

        error = np.linalg.norm(lhs - rhs, ord="fro")
        max_error = max(max_error, float(error))

    return max_error


def main():
    basis = operator_basis()
    permutations_4d = list(permutations(range(4)))

    channels = {
        "amplitude_damping": amplitude_damping,
        "phase_damping": phase_damping,
    }

    p_values = np.linspace(0.0, 1.0, 21)

    results = {
        "experiment": "QDK-019",
        "title": "Channel-Global Symmetry Algebra",
        "operator_basis_dimension": len(basis),
        "permutations_tested": len(permutations_4d),
        "p_points": len(p_values),
        "tolerance": TOL,
        "channels": {},
    }

    for channel_name, channel in channels.items():
        print("=" * 70)
        print(channel_name)

        symmetry_set = []
        errors = {}

        for perm in permutations_4d:
            U = permutation_unitary(perm)

            max_error = 0.0

            for p in p_values:
                err = channel_covariance_error(
                    channel,
                    U,
                    p,
                    basis,
                )
                max_error = max(max_error, err)

            key = str(perm)
            errors[key] = max_error

            if max_error < TOL:
                symmetry_set.append(perm)

        results["channels"][channel_name] = {
            "global_symmetry_count": len(symmetry_set),
            "global_symmetries": [
                list(perm) for perm in symmetry_set
            ],
            "max_covariance_error": max(errors.values()),
            "errors": errors,
        }

        print(
            "global symmetry count:",
            len(symmetry_set),
        )

        print(
            "max covariance error:",
            f"{max(errors.values()):.12e}",
        )

        print("global symmetry set:")

        for perm in symmetry_set:
            print(" ", perm)

    common = set(
        tuple(x)
        for x in results["channels"]["amplitude_damping"][
            "global_symmetries"
        ]
    ).intersection(
        tuple(x)
        for x in results["channels"]["phase_damping"][
            "global_symmetries"
        ]
    )

    results["common_global_symmetry_count"] = len(common)
    results["common_global_symmetries"] = [
        list(x) for x in sorted(common)
    ]

    print("=" * 70)
    print(
        "COMMON GLOBAL SYMMETRIES:",
        len(common),
    )

    for perm in sorted(common):
        print(" ", perm)

    output = Path(
        "results/qdk_019/QDK-019-CHANNEL-GLOBAL-SYMMETRY.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.item()
            if hasattr(x, "item")
            else x,
        )

    print("=" * 70)
    print("QDK-019 COMPLETE")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
