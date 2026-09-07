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


OBSERVABLES = {
    "XX": kron(X, X),
    "YY": kron(Y, Y),
    "ZZ": kron(Z, Z),
    "ZI": kron(Z, I2),
    "IZ": kron(I2, Z),
}


# QDK-020 physical local symmetry
# permutation (1,0,3,2) = I tensor X
U_STAR = kron(I2, X)


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


def transform_state(rho, U):
    return (
        U
        @ rho
        @ U.conj().T
    )


def transform_observable(O, U):
    return (
        U
        @ O
        @ U.conj().T
    )


def fingerprint(
    rho,
    channel,
    observables,
):
    rows = []

    for p in P_VALUES:

        noisy = channel(
            rho,
            p,
        )

        row = []

        for O in observables.values():

            value = np.trace(
                O @ noisy
            )

            row.append(
                float(np.real(value))
            )

        rows.append(row)

    return np.asarray(
        rows,
        dtype=float,
    )


def max_difference(A, B):
    return float(
        np.max(
            np.abs(A - B)
        )
    )


def euclidean_distance(A, B):
    return float(
        np.linalg.norm(
            A - B
        )
    )


def normalize_columns(A):
    scale = np.max(
        np.abs(A),
        axis=0,
    )

    scale[scale < TOL] = 1.0

    return A / scale


def main():

    channels = {
        "phase_damping": phase_damping,
        "amplitude_damping": amplitude_damping,
    }

    results = {
        "experiment": "QDK-022",
        "title": (
            "Corrected Symmetry-Covariant "
            "Structural Fingerprint"
        ),
        "symmetry": {
            "permutation": [1, 0, 3, 2],
            "operator": "I tensor X",
            "physical_local_unitary": True,
        },
        "observables": list(
            OBSERVABLES.keys()
        ),
        "p_points": len(P_VALUES),
        "channels": {},
    }

    for channel_name, channel in channels.items():

        print("=" * 70)
        print(channel_name)
        print("=" * 70)

        channel_result = {
            "structures": {},
            "between_structure_distances": {},
            "max_covariance_error": 0.0,
        }

        fingerprints = {}

        for name, tensor in STRUCTURES.items():

            psi = tensor_to_state(tensor)
            rho = density_matrix(psi)

            rho_transformed = transform_state(
                rho,
                U_STAR,
            )

            observables_transformed = {
                key: transform_observable(
                    O,
                    U_STAR,
                )
                for key, O in OBSERVABLES.items()
            }

            F = fingerprint(
                rho,
                channel,
                OBSERVABLES,
            )

            F_transformed = fingerprint(
                rho_transformed,
                channel,
                observables_transformed,
            )

            covariance_error = max_difference(
                F,
                F_transformed,
            )

            normalized_F = normalize_columns(F)
            normalized_F_transformed = (
                normalize_columns(
                    F_transformed
                )
            )

            normalized_error = max_difference(
                normalized_F,
                normalized_F_transformed,
            )

            channel_result[
                "structures"
            ][name] = {
                "raw_covariance_error": (
                    covariance_error
                ),
                "normalized_covariance_error": (
                    normalized_error
                ),
            }

            fingerprints[name] = normalized_F

            channel_result[
                "max_covariance_error"
            ] = max(
                channel_result[
                    "max_covariance_error"
                ],
                covariance_error,
            )

            print(
                name,
                "raw covariance error:",
                f"{covariance_error:.12e}",
            )

            print(
                name,
                "normalized covariance error:",
                f"{normalized_error:.12e}",
            )

        names = list(
            STRUCTURES.keys()
        )

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a = names[i]
                b = names[j]

                distance = euclidean_distance(
                    fingerprints[a],
                    fingerprints[b],
                )

                key = f"{a}_vs_{b}"

                channel_result[
                    "between_structure_distances"
                ][key] = distance

                print(
                    key,
                    "normalized fingerprint distance:",
                    f"{distance:.12e}",
                )

        channel_result[
            "covariance_pass"
        ] = bool(
            channel_result[
                "max_covariance_error"
            ] < TOL
        )

        results[
            "channels"
        ][channel_name] = channel_result

    results["summary"] = {
        "phase_damping_pass": bool(
            results[
                "channels"
            ][
                "phase_damping"
            ][
                "covariance_pass"
            ]
        ),
        "amplitude_damping_pass": bool(
            results[
                "channels"
            ][
                "amplitude_damping"
            ][
                "covariance_pass"
            ]
        ),
    }

    output = Path(
        "results/qdk_022/"
        "QDK-022-CORRECTED-COVARIANT-FINGERPRINT.json"
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
    print("QDK-022 COMPLETE")
    print("=" * 70)

    print(
        "phase damping PASS:",
        results[
            "summary"
        ][
            "phase_damping_pass"
        ],
    )

    print(
        "amplitude damping PASS:",
        results[
            "summary"
        ][
            "amplitude_damping_pass"
        ],
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
