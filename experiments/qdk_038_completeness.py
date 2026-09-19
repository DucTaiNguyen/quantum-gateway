import json
from pathlib import Path

import numpy as np


TOLERANCE = 1e-10

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
}

OBS = {}

for a_name, a in PAULIS.items():
    for b_name, b in PAULIS.items():
        name = a_name + b_name

        if name != "II":
            OBS[name] = np.kron(a, b)

P_GRID = np.linspace(0.0, 1.0, 21)


def depolarizing(rho, p):
    return (
        (1.0 - p) * rho
        + p * np.eye(4, dtype=complex) / 4.0
    )


def nonlinear_depolarizing(rho, p):
    q = p**2

    return (
        (1.0 - q) * rho
        + q * np.eye(4, dtype=complex) / 4.0
    )


def phase_damping(rho, p):
    g = np.sqrt(max(0.0, 1.0 - p))

    D = np.diag([1.0, g, g, 1.0])

    return D @ rho @ D


def nonlinear_phase(rho, p):
    q = p**2
    g = np.sqrt(max(0.0, 1.0 - q))

    D = np.diag([1.0, g, g, 1.0])

    return D @ rho @ D


CHANNELS = {
    "depolarizing": depolarizing,
    "nonlinear_depolarizing": nonlinear_depolarizing,
    "phase_damping": phase_damping,
    "nonlinear_phase": nonlinear_phase,
}


def basis():
    result = []

    for a_name, a in PAULIS.items():
        for b_name, b in PAULIS.items():
            result.append(
                (a_name + b_name, np.kron(a, b))
            )

    return result


BASIS = basis()


def fingerprint(rho, channel):
    values = []

    for p in P_GRID:
        rp = channel(rho, p)

        for O in OBS.values():
            values.append(
                float(
                    np.real(
                        np.trace(O @ rp)
                    )
                )
            )

    return np.asarray(values)


def build_matrix(channel):
    columns = []

    # Full Pauli basis, including identity.
    # The identity component is removed because
    # density matrices have fixed trace.
    for name, B in BASIS:

        if name == "II":
            continue

        columns.append(
            fingerprint(B, channel)
        )

    return np.asarray(columns).T


def main():
    print("QDK-038 FINGERPRINT COMPLETENESS")
    print("=" * 50)
    print(f"Observables: {len(OBS)}")
    print(f"Noise points: {len(P_GRID)}")

    results = {}

    for name, channel in CHANNELS.items():

        M = build_matrix(channel)

        singular_values = np.linalg.svd(
            M,
            compute_uv=False,
        )

        rank = int(
            np.sum(
                singular_values
                > TOLERANCE
            )
        )

        nullity = (
            M.shape[1] - rank
        )

        max_rank = 15

        complete = (
            rank == max_rank
        )

        results[name] = {
            "matrix_shape": list(M.shape),
            "rank": rank,
            "maximum_possible_rank": max_rank,
            "nullity": nullity,
            "complete": complete,
            "minimum_singular_value": float(
                singular_values[-1]
            ),
            "maximum_singular_value": float(
                singular_values[0]
            ),
        }

        print()
        print(f"CHANNEL: {name}")
        print(
            f"Matrix shape: {M.shape}"
        )
        print(
            f"Rank: {rank}/{max_rank}"
        )
        print(
            f"Nullity: {nullity}"
        )
        print(
            f"Complete: {complete}"
        )
        print(
            "Minimum singular value: "
            f"{singular_values[-1]:.6e}"
        )

    result = {
        "experiment": "QDK-038",
        "title": (
            "Quantum Noise Fingerprint "
            "Completeness and Rank"
        ),
        "observable_count": len(OBS),
        "observables": list(OBS.keys()),
        "noise_points": len(P_GRID),
        "maximum_possible_rank": 15,
        "results": results,
        "scientific_note": (
            "For two-qubit trace-one Hermitian states, "
            "the physical state space has 15 independent "
            "real parameters. Rank 15 indicates that the "
            "tested fingerprint map is informationally "
            "complete within the modeled operator space."
        ),
    }

    out = Path("results/qdk_038")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-038-COMPLETENESS.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("QDK-038 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
