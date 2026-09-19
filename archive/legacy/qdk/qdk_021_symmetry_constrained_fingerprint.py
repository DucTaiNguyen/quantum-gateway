import json
from pathlib import Path

import numpy as np


TOL = 1e-12
P_VALUES = np.linspace(0.0, 1.0, 21)


I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1], [1, 0]],
    dtype=complex,
)

Y = np.array(
    [[0, -1j], [1j, 0]],
    dtype=complex,
)

Z = np.array(
    [[1, 0], [0, -1]],
    dtype=complex,
)


def kron(a, b):
    return np.kron(a, b)


# ------------------------------------------------------------
# Reference structures
# ------------------------------------------------------------

STRUCTURES = {
    "STRUCTURED": np.array(
        [[1.0, 0.8],
         [0.8, 1.0]],
        dtype=float,
    ),
    "WEAK": np.array(
        [[1.0, 0.2],
         [0.2, 1.0]],
        dtype=float,
    ),
    "ASYMMETRIC": np.array(
        [[1.0, 0.2],
         [0.7, 1.0]],
        dtype=float,
    ),
}


def tensor_to_state(tensor):
    v = np.asarray(
        tensor,
        dtype=float,
    ).reshape(-1)

    v = np.abs(v)

    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError(
            "Cannot encode zero tensor."
        )

    return v / norm


def density_matrix(psi):
    return np.outer(
        psi,
        psi.conj(),
    )


# ------------------------------------------------------------
# Physical symmetry from QDK-020
#
# permutation (1,0,3,2)
#
# In computational basis this is:
#
# |00> <-> |01>
# |10> <-> |11>
#
# = I ⊗ X
# ------------------------------------------------------------

U_STAR = kron(I2, X)


# ------------------------------------------------------------
# Channels
# ------------------------------------------------------------

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

    K0 = kron(k0, I2)
    K1 = kron(k1, I2)

    return (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )


def phase_damping(rho, p):
    p = float(p)

    k0 = np.sqrt(1.0 - p) * I2
    k1 = np.sqrt(p) * Z

    K0 = kron(k0, I2)
    K1 = kron(k1, I2)

    return (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )


# ------------------------------------------------------------
# Observables
# ------------------------------------------------------------

OBSERVABLES = {
    "XX": kron(X, X),
    "YY": kron(Y, Y),
    "ZZ": kron(Z, Z),
    "ZI": kron(Z, I2),
    "IZ": kron(I2, Z),
}


def observable_vector(rho):
    return np.array(
        [
            np.real(
                np.trace(O @ rho)
            )
            for O in OBSERVABLES.values()
        ],
        dtype=float,
    )


def fingerprint_trajectory(
    rho,
    channel,
):
    trajectory = []

    for p in P_VALUES:
        noisy = channel(
            rho,
            p,
        )

        trajectory.append(
            observable_vector(noisy)
        )

    return np.asarray(
        trajectory,
        dtype=float,
    )


# ------------------------------------------------------------
# Column normalization
# ------------------------------------------------------------

def normalize_columns(trajectory):
    scale = np.max(
        np.abs(trajectory),
        axis=0,
    )

    scale[scale < TOL] = 1.0

    return trajectory / scale


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def trajectory_distance(A, B):
    return float(
        np.linalg.norm(
            A - B,
        )
    )


def max_difference(A, B):
    return float(
        np.max(
            np.abs(A - B)
        )
    )


def apply_symmetry(rho):
    return (
        U_STAR
        @ rho
        @ U_STAR.conj().T
    )


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():

    results = {
        "experiment": "QDK-021",
        "title": (
            "Symmetry-Constrained "
            "Structural Fingerprint"
        ),
        "symmetry": {
            "permutation": [1, 0, 3, 2],
            "operator": "I tensor X",
            "local_unitary": True,
        },
        "p_points": len(P_VALUES),
        "observables": list(
            OBSERVABLES.keys()
        ),
        "channels": {},
    }

    for channel_name, channel in {
        "phase_damping": phase_damping,
        "amplitude_damping": amplitude_damping,
    }.items():

        print("=" * 70)
        print(channel_name)
        print("=" * 70)

        channel_result = {
            "structures": {},
            "symmetry_covariance": {},
            "between_structure_distances": {},
        }

        fingerprints = {}

        # ----------------------------------------------------
        # Construct fingerprints
        # ----------------------------------------------------

        for name, tensor in STRUCTURES.items():

            psi = tensor_to_state(tensor)
            rho = density_matrix(psi)

            original = fingerprint_trajectory(
                rho,
                channel,
            )

            transformed_rho = apply_symmetry(
                rho
            )

            transformed = fingerprint_trajectory(
                transformed_rho,
                channel,
            )

            original_norm = normalize_columns(
                original
            )

            transformed_norm = normalize_columns(
                transformed
            )

            covariance_error = max_difference(
                original_norm,
                transformed_norm,
            )

            fingerprints[name] = original_norm

            channel_result[
                "structures"
            ][name] = {
                "symmetry_covariance_error": (
                    covariance_error
                )
            }

            print(
                name,
                "symmetry covariance error:",
                f"{covariance_error:.12e}",
            )

        # ----------------------------------------------------
        # Between-structure fingerprint distances
        # ----------------------------------------------------

        names = list(STRUCTURES.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a = names[i]
                b = names[j]

                d = trajectory_distance(
                    fingerprints[a],
                    fingerprints[b],
                )

                key = f"{a}_vs_{b}"

                channel_result[
                    "between_structure_distances"
                ][key] = d

                print(
                    key,
                    "fingerprint distance:",
                    f"{d:.12e}",
                )

        # ----------------------------------------------------
        # Symmetry orbit comparison
        # ----------------------------------------------------

        orbit_tests = {}

        for name, tensor in STRUCTURES.items():

            psi = tensor_to_state(tensor)
            rho = density_matrix(psi)

            rho_sym = apply_symmetry(rho)

            # State-level orbit equivalence
            state_error = np.linalg.norm(
                rho_sym
                - apply_symmetry(rho),
                ord="fro",
            )

            orbit_tests[name] = {
                "state_transform_consistency": float(
                    state_error
                )
            }

        channel_result[
            "symmetry_orbit_tests"
        ] = orbit_tests

        results[
            "channels"
        ][channel_name] = channel_result

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    all_phase_errors = [
        x[
            "symmetry_covariance_error"
        ]
        for x in results[
            "channels"
        ][
            "phase_damping"
        ][
            "structures"
        ].values()
    ]

    all_amp_errors = [
        x[
            "symmetry_covariance_error"
        ]
        for x in results[
            "channels"
        ][
            "amplitude_damping"
        ][
            "structures"
        ].values()
    ]

    results["summary"] = {
        "phase_damping_covariance_pass": bool(
            max(all_phase_errors) < TOL
        ),
        "amplitude_damping_covariance_pass": bool(
            max(all_amp_errors) < TOL
        ),
        "phase_damping_max_error": float(
            max(all_phase_errors)
        ),
        "amplitude_damping_max_error": float(
            max(all_amp_errors)
        ),
    }

    output = Path(
        "results/qdk_021/"
        "QDK-021-SYMMETRY-CONSTRAINED-FINGERPRINT.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    print("=" * 70)
    print("QDK-021 COMPLETE")
    print("=" * 70)
    print(
        "phase damping covariance PASS:",
        results[
            "summary"
        ][
            "phase_damping_covariance_pass"
        ],
    )
    print(
        "amplitude damping covariance PASS:",
        results[
            "summary"
        ][
            "amplitude_damping_covariance_pass"
        ],
    )
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
