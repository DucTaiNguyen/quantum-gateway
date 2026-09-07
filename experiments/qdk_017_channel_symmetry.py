"""
QDK-017 — Channel-Symmetry-Aware Fingerprint Test

Author: Tai D. Nguyen
© 2026 Tai D. Nguyen. All rights reserved.
"""

from pathlib import Path
import itertools
import json
import numpy as np


OUTPUT_DIR = Path("results/qdk_017")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOL = 1e-12


# ============================================================
# Quantum utilities
# ============================================================

def normalize_state(v):
    v = np.asarray(v, dtype=complex)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return v / norm


def density_matrix(state):
    state = normalize_state(state)
    return np.outer(state, state.conj())


def permutation_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n), dtype=complex)

    for i, j in enumerate(perm):
        P[j, i] = 1.0

    return P


def tensor_to_state(tensor):
    values = np.asarray(tensor, dtype=float).reshape(-1)
    values = np.maximum(values, 0.0)
    return normalize_state(values)


# ============================================================
# Pauli operators
# ============================================================

I = np.eye(2, dtype=complex)

X = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

Y = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

Z = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=complex,
)


def kron(a, b):
    return np.kron(a, b)


OBSERVABLES = {
    "XX": kron(X, X),
    "YY": kron(Y, Y),
    "ZZ": kron(Z, Z),
    "ZI": kron(Z, I),
    "IZ": kron(I, Z),
}


# ============================================================
# Structures
# ============================================================

STRUCTURES = {
    "STRUCTURED": np.array(
        [
            [1.0, 0.8],
            [0.8, 1.0],
        ]
    ),
    "WEAK": np.array(
        [
            [1.0, 0.2],
            [0.2, 1.0],
        ]
    ),
    "ASYMMETRIC": np.array(
        [
            [1.0, 0.2],
            [0.7, 1.0],
        ]
    ),
}


# ============================================================
# Noise channels
# ============================================================

def amplitude_damping(rho, p):
    p = float(p)

    E0 = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(1.0 - p)],
        ],
        dtype=complex,
    )

    E1 = np.array(
        [
            [0.0, np.sqrt(p)],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    kraus = [
        np.kron(E0, E0),
        np.kron(E0, E1),
        np.kron(E1, E0),
        np.kron(E1, E1),
    ]

    out = np.zeros_like(rho, dtype=complex)

    for K in kraus:
        out += K @ rho @ K.conj().T

    return out


def phase_damping(rho, p):
    p = float(p)

    K0 = np.sqrt(1.0 - p / 2.0) * I
    K1 = np.sqrt(p / 2.0) * Z

    kraus = [
        np.kron(K0, K0),
        np.kron(K0, K1),
        np.kron(K1, K0),
        np.kron(K1, K1),
    ]

    out = np.zeros_like(rho, dtype=complex)

    for K in kraus:
        out += K @ rho @ K.conj().T

    return out


CHANNELS = {
    "amplitude_damping": amplitude_damping,
    "phase_damping": phase_damping,
}


# ============================================================
# Diagnostics
# ============================================================

def representation_covariance_error(rho, observable, U):
    rho_u = U @ rho @ U.conj().T
    observable_u = U @ observable @ U.conj().T

    lhs = np.trace(rho_u @ observable_u)
    rhs = np.trace(rho @ observable)

    return float(abs(lhs - rhs))


def channel_covariance_error(channel, rho, U, p):
    rho_u = U @ rho @ U.conj().T

    lhs = channel(rho_u, p)
    rhs = U @ channel(rho, p) @ U.conj().T

    return float(np.linalg.norm(lhs - rhs, ord="fro"))


def fingerprint(rho, observables, channel, p):
    noisy = channel(rho, p)

    return np.array(
        [
            np.real(np.trace(noisy @ O))
            for O in observables.values()
        ],
        dtype=float,
    )


def covariant_fingerprint(rho, observables, channel, p, U):
    rho_u = U @ rho @ U.conj().T

    transformed_observables = {
        name: U @ O @ U.conj().T
        for name, O in observables.items()
    }

    noisy = channel(rho_u, p)

    return np.array(
        [
            np.real(np.trace(noisy @ O))
            for O in transformed_observables.values()
        ],
        dtype=float,
    )


# ============================================================
# Main experiment
# ============================================================

def run():

    permutations = list(itertools.permutations(range(4)))
    p_values = np.linspace(0.0, 1.0, 21)

    results = {
        "experiment": "QDK-017",
        "title": "Channel-Symmetry-Aware Fingerprint Test",
        "author": "Tai D. Nguyen",
        "tolerance": TOL,
        "n_permutations": len(permutations),
        "p_points": len(p_values),
        "structures": {},
    }

    # --------------------------------------------------------
    # Noiseless representation covariance
    # --------------------------------------------------------

    noiseless_max = 0.0

    for tensor in STRUCTURES.values():

        rho = density_matrix(
            tensor_to_state(tensor)
        )

        for perm in permutations:

            U = permutation_matrix(perm)

            for O in OBSERVABLES.values():

                err = representation_covariance_error(
                    rho,
                    O,
                    U,
                )

                noiseless_max = max(
                    noiseless_max,
                    err,
                )

    results["noiseless_representation_covariance"] = {
        "max_error": noiseless_max,
        "pass": noiseless_max < TOL,
    }

    print("=" * 70)
    print("QDK-017 CHANNEL-SYMMETRY-AWARE FINGERPRINT TEST")
    print("=" * 70)

    print()
    print("NOISELESS REPRESENTATION COVARIANCE")
    print("-" * 70)
    print(f"max error: {noiseless_max:.16e}")
    print(f"PASS: {noiseless_max < TOL}")

    # --------------------------------------------------------
    # Noise channel symmetry
    # --------------------------------------------------------

    for channel_name, channel in CHANNELS.items():

        results["structures"][channel_name] = {}

        print()
        print(channel_name)
        print("-" * 70)

        for structure_name, tensor in STRUCTURES.items():

            rho = density_matrix(
                tensor_to_state(tensor)
            )

            channel_results = []

            for perm in permutations:

                U = permutation_matrix(perm)

                max_channel_error = 0.0
                max_fingerprint_error = 0.0

                for p in p_values:

                    error = channel_covariance_error(
                        channel,
                        rho,
                        U,
                        p,
                    )

                    max_channel_error = max(
                        max_channel_error,
                        error,
                    )

                    f0 = fingerprint(
                        rho,
                        OBSERVABLES,
                        channel,
                        p,
                    )

                    fu = covariant_fingerprint(
                        rho,
                        OBSERVABLES,
                        channel,
                        p,
                        U,
                    )

                    fp_error = np.linalg.norm(
                        f0 - fu
                    )

                    max_fingerprint_error = max(
                        max_fingerprint_error,
                        fp_error,
                    )

                channel_results.append(
                    {
                        "permutation": list(perm),
                        "max_channel_error": max_channel_error,
                        "max_fingerprint_error": max_fingerprint_error,
                        "channel_symmetry": (
                            max_channel_error < TOL
                        ),
                    }
                )

            results["structures"][channel_name][
                structure_name
            ] = channel_results

            symmetry_count = sum(
                r["channel_symmetry"]
                for r in channel_results
            )

            max_error = max(
                r["max_channel_error"]
                for r in channel_results
            )

            print(
                f"{structure_name:12s} "
                f"channel symmetries: "
                f"{symmetry_count:2d}/24 "
                f"max error: "
                f"{max_error:.6e}"
            )

    # --------------------------------------------------------
    # Common channel symmetries
    # --------------------------------------------------------

    results["common_symmetries"] = {}

    for channel_name in CHANNELS:

        common = None

        for structure_name in STRUCTURES:

            rows = results["structures"][
                channel_name
            ][structure_name]

            current = {
                tuple(r["permutation"])
                for r in rows
                if r["channel_symmetry"]
            }

            if common is None:
                common = current
            else:
                common &= current

        common = sorted(common)

        results["common_symmetries"][
            channel_name
        ] = [
            list(p)
            for p in common
        ]

        print()
        print(
            f"{channel_name.upper()} "
            "COMMON SYMMETRIES"
        )
        print("-" * 70)
        print(
            f"common channel symmetries: "
            f"{len(common)}/24"
        )

        for perm in common:
            print(f"  {perm}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        OUTPUT_DIR
        / "QDK-017-CHANNEL-SYMMETRY.json"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("QDK-017 COMPLETE")
    print("=" * 70)
    print(f"Saved: {output}")


if __name__ == "__main__":
    run()
