"""
QDK-018 — Physical Symmetry Group Validation

Tests:
1. Identity
2. Closure
3. Inverse
4. Channel covariance
5. Fingerprint covariance

Author: Tai D. Nguyen
© 2026 Tai D. Nguyen. All rights reserved.
"""

from pathlib import Path
import json
import numpy as np


OUTPUT_DIR = Path("results/qdk_018")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOL = 1e-12

# ------------------------------------------------------------
# The 8 common symmetries identified by QDK-017
# ------------------------------------------------------------

SYMMETRIES = [
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (1, 0, 3, 2),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
    (2, 3, 0, 1),
    (3, 1, 2, 0),
    (3, 2, 1, 0),
]


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

def normalize(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


def density_matrix(state):
    state = normalize(state)
    return np.outer(state, state.conj())


def permutation_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n), dtype=complex)

    for i, j in enumerate(perm):
        P[j, i] = 1.0

    return P


def compose(p, q):
    """
    Composition p o q.
    """
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    inv = [0] * len(p)

    for i, j in enumerate(p):
        inv[j] = i

    return tuple(inv)


# ------------------------------------------------------------
# Structures
# ------------------------------------------------------------

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


def tensor_to_state(tensor):
    x = np.asarray(tensor, dtype=float).reshape(-1)
    x = np.maximum(x, 0.0)
    return normalize(x)


# ------------------------------------------------------------
# Pauli operators
# ------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

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
    "ZI": kron(Z, I2),
    "IZ": kron(I2, Z),
}


# ------------------------------------------------------------
# Noise channels
# ------------------------------------------------------------

def amplitude_damping(rho, p):
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
    K0 = np.sqrt(1.0 - p / 2.0) * I2
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


# ------------------------------------------------------------
# Fingerprint
# ------------------------------------------------------------

def fingerprint(rho, channel, p):
    noisy = channel(rho, p)

    return np.array(
        [
            np.real(np.trace(noisy @ O))
            for O in OBSERVABLES.values()
        ]
    )


def transformed_fingerprint(rho, channel, p, U):
    rho_u = U @ rho @ U.conj().T

    noisy = channel(rho_u, p)

    values = []

    for O in OBSERVABLES.values():
        O_u = U @ O @ U.conj().T
        values.append(
            np.real(np.trace(noisy @ O_u))
        )

    return np.array(values)


# ------------------------------------------------------------
# Group tests
# ------------------------------------------------------------

def test_identity():
    identity = (0, 1, 2, 3)
    return identity in SYMMETRIES


def test_closure():
    G = set(SYMMETRIES)

    failures = []

    for a in SYMMETRIES:
        for b in SYMMETRIES:

            c = compose(a, b)

            if c not in G:
                failures.append(
                    {
                        "a": list(a),
                        "b": list(b),
                        "composition": list(c),
                    }
                )

    return failures


def test_inverses():
    G = set(SYMMETRIES)

    failures = []

    for a in SYMMETRIES:

        inv = inverse(a)

        if inv not in G:
            failures.append(
                {
                    "element": list(a),
                    "inverse": list(inv),
                }
            )

    return failures


# ------------------------------------------------------------
# Channel validation
# ------------------------------------------------------------

def validate_channel(channel):

    p_values = np.linspace(0.0, 1.0, 21)

    max_channel_error = 0.0
    max_fingerprint_error = 0.0

    per_structure = {}

    for structure_name, tensor in STRUCTURES.items():

        rho = density_matrix(
            tensor_to_state(tensor)
        )

        structure_channel_error = 0.0
        structure_fingerprint_error = 0.0

        for perm in SYMMETRIES:

            U = permutation_matrix(perm)

            for p in p_values:

                rho_u = U @ rho @ U.conj().T

                lhs = channel(rho_u, p)

                rhs = (
                    U
                    @ channel(rho, p)
                    @ U.conj().T
                )

                channel_error = np.linalg.norm(
                    lhs - rhs,
                    ord="fro",
                )

                f0 = fingerprint(
                    rho,
                    channel,
                    p,
                )

                fu = transformed_fingerprint(
                    rho,
                    channel,
                    p,
                    U,
                )

                fp_error = np.linalg.norm(
                    f0 - fu
                )

                structure_channel_error = max(
                    structure_channel_error,
                    channel_error,
                )

                structure_fingerprint_error = max(
                    structure_fingerprint_error,
                    fp_error,
                )

        per_structure[structure_name] = {
            "max_channel_error": structure_channel_error,
            "max_fingerprint_error": structure_fingerprint_error,
        }

        max_channel_error = max(
            max_channel_error,
            structure_channel_error,
        )

        max_fingerprint_error = max(
            max_fingerprint_error,
            structure_fingerprint_error,
        )

    return {
        "max_channel_error": max_channel_error,
        "max_fingerprint_error": max_fingerprint_error,
        "per_structure": per_structure,
        "channel_covariance_pass": bool(
            max_channel_error < TOL
        ),
        "fingerprint_covariance_pass": bool(
            max_fingerprint_error < TOL
        ),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("QDK-018 PHYSICAL SYMMETRY GROUP VALIDATION")
    print("=" * 70)

    results = {
        "experiment": "QDK-018",
        "title": "Physical Symmetry Group Validation",
        "author": "Tai D. Nguyen",
        "tolerance": TOL,
        "symmetry_count": len(SYMMETRIES),
        "symmetries": [
            list(x) for x in SYMMETRIES
        ],
    }

    # --------------------------------------------------------
    # Group structure
    # --------------------------------------------------------

    identity_pass = test_identity()

    closure_failures = test_closure()
    inverse_failures = test_inverses()

    group_pass = (
        identity_pass
        and len(closure_failures) == 0
        and len(inverse_failures) == 0
    )

    results["group_tests"] = {
        "identity": identity_pass,
        "closure": len(closure_failures) == 0,
        "inverse": len(inverse_failures) == 0,
        "closure_failures": closure_failures,
        "inverse_failures": inverse_failures,
        "group_pass": bool(group_pass),
    }

    print()
    print("GROUP STRUCTURE")
    print("-" * 70)
    print(f"elements: {len(SYMMETRIES)}")
    print(f"identity: {identity_pass}")
    print(
        f"closure: "
        f"{len(closure_failures) == 0}"
    )
    print(
        f"inverse: "
        f"{len(inverse_failures) == 0}"
    )
    print(f"GROUP PASS: {group_pass}")

    # --------------------------------------------------------
    # Channel tests
    # --------------------------------------------------------

    results["channels"] = {}

    for channel_name, channel in CHANNELS.items():

        validation = validate_channel(channel)

        results["channels"][channel_name] = validation

        print()
        print(channel_name.upper())
        print("-" * 70)

        print(
            "max channel covariance error: "
            f"{validation['max_channel_error']:.12e}"
        )

        print(
            "max fingerprint covariance error: "
            f"{validation['max_fingerprint_error']:.12e}"
        )

        print(
            "channel covariance PASS: "
            f"{validation['channel_covariance_pass']}"
        )

        print(
            "fingerprint covariance PASS: "
            f"{validation['fingerprint_covariance_pass']}"
        )

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    overall_pass = (
        group_pass
        and all(
            x["channel_covariance_pass"]
            for x in results["channels"].values()
        )
    )

    results["overall"] = {
        "physical_symmetry_group_pass": bool(overall_pass),
        "interpretation": (
            "The identified transformations form a valid "
            "permutation group and preserve the tested "
            "reference noise channels for the tested states."
            if overall_pass
            else
            "The identified transformations do not satisfy "
            "all physical symmetry requirements under the "
            "tested reference channels."
        ),
    }

    output = (
        OUTPUT_DIR
        / "QDK-018-PHYSICAL-SYMMETRY.json"
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
    print("QDK-018 COMPLETE")
    print("=" * 70)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
