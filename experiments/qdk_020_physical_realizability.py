import json
from pathlib import Path

import numpy as np


TOL = 1e-12


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def permutation_unitary(perm):
    U = np.zeros((4, 4), dtype=complex)

    for i, j in enumerate(perm):
        U[j, i] = 1.0

    return U


def partial_trace_B(rho):
    return np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)


def partial_trace_A(rho):
    return np.trace(rho.reshape(2, 2, 2, 2), axis1=0, axis2=2)


def operator_schmidt_rank(U):
    """
    Operator Schmidt rank of a 2-qubit operator.

    U_{ij,kl} is reshaped into a 4x4 matrix:
        (i,k) x (j,l)

    The matrix rank gives the operator Schmidt rank.
    A product operator A⊗B has rank 1.
    """

    tensor = U.reshape(2, 2, 2, 2)
    reshaped = np.transpose(
        tensor,
        (0, 2, 1, 3),
    ).reshape(4, 4)

    singular_values = np.linalg.svd(
        reshaped,
        compute_uv=False,
    )

    rank = int(
        np.sum(singular_values > TOL)
    )

    return rank, singular_values


def extract_product_factors(U):
    """
    If operator Schmidt rank is 1, reconstruct A⊗B
    up to a global complex scale.
    """

    tensor = U.reshape(2, 2, 2, 2)

    reshaped = np.transpose(
        tensor,
        (0, 2, 1, 3),
    ).reshape(4, 4)

    u, s, vh = np.linalg.svd(reshaped)

    if s[1] > TOL:
        return None, None

    A = (
        np.sqrt(s[0])
        * u[:, 0]
        .reshape(2, 2)
    )

    B = (
        np.sqrt(s[0])
        * vh[0, :]
        .reshape(2, 2)
    )

    return A, B


def unitary_error(U):
    return float(
        np.linalg.norm(
            U.conj().T @ U - np.eye(4),
            ord="fro",
        )
    )


def product_reconstruction_error(U, A, B):
    if A is None or B is None:
        return float("inf")

    V = np.kron(A, B)

    norm_u = np.linalg.norm(U)

    if norm_u == 0:
        return float("inf")

    overlap = np.vdot(V, U)

    if abs(overlap) == 0:
        return float(
            np.linalg.norm(U - V)
        )

    phase = overlap / abs(overlap)

    V_aligned = phase * V

    return float(
        np.linalg.norm(
            U - V_aligned,
            ord="fro",
        )
    )


def main():
    target_perm = (1, 0, 3, 2)

    U = permutation_unitary(target_perm)

    rank, singular_values = operator_schmidt_rank(U)

    A, B = extract_product_factors(U)

    result = {
        "experiment": "QDK-020",
        "title": "Physical Realizability of Global Symmetry",
        "target_permutation": list(target_perm),
        "matrix_dimension": 4,
        "unitarity_error": unitary_error(U),
        "operator_schmidt_rank": rank,
        "operator_schmidt_singular_values": [
            float(x) for x in singular_values
        ],
        "local_product_candidate": bool(
            rank == 1
        ),
    }

    if A is not None and B is not None:
        reconstruction_error = (
            product_reconstruction_error(
                U,
                A,
                B,
            )
        )

        result[
            "product_reconstruction_error"
        ] = reconstruction_error

        result[
            "physical_local_unitary"
        ] = bool(
            rank == 1
            and reconstruction_error < TOL
            and unitary_error(U) < TOL
        )

        result["factor_A"] = [
            [
                {
                    "real": float(z.real),
                    "imag": float(z.imag),
                }
                for z in row
            ]
            for row in A
        ]

        result["factor_B"] = [
            [
                {
                    "real": float(z.real),
                    "imag": float(z.imag),
                }
                for z in row
            ]
            for row in B
        ]
    else:
        result[
            "product_reconstruction_error"
        ] = None

        result[
            "physical_local_unitary"
        ] = False

    print("=" * 70)
    print("QDK-020")
    print("=" * 70)

    print(
        "target permutation:",
        target_perm,
    )

    print(
        "unitarity error:",
        f"{result['unitarity_error']:.12e}",
    )

    print(
        "operator Schmidt rank:",
        rank,
    )

    print(
        "singular values:",
        [
            f"{x:.12e}"
            for x in singular_values
        ],
    )

    print(
        "local product candidate:",
        result[
            "local_product_candidate"
        ],
    )

    print(
        "physical local unitary:",
        result[
            "physical_local_unitary"
        ],
    )

    print("=" * 70)

    output = Path(
        "results/qdk_020/"
        "QDK-020-PHYSICAL-REALIZABILITY.json"
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
            result,
            f,
            indent=2,
        )

    print("QDK-020 COMPLETE")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
