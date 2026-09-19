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

OBS_NAMES = list(OBS.keys())

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


BASIS = [
    item
    for item in basis()
    if item[0] != "II"
]


def fingerprint_basis(
    B,
    channel,
    obs_names,
):
    values = []

    for p in P_GRID:
        rp = channel(B, p)

        for name in obs_names:
            values.append(
                float(
                    np.real(
                        np.trace(
                            OBS[name] @ rp
                        )
                    )
                )
            )

    return np.asarray(values)


def build_observable_columns(channel):
    columns = {}

    for name in OBS_NAMES:

        column = []

        for _, B in BASIS:
            column.append(
                fingerprint_basis(
                    B,
                    channel,
                    [name],
                )
            )

        columns[name] = np.asarray(
            column
        ).T

    return columns


def matrix_rank(M):
    if M.size == 0:
        return 0

    singular_values = np.linalg.svd(
        M,
        compute_uv=False,
    )

    return int(
        np.sum(
            singular_values
            > TOLERANCE
        )
    )


def greedy_selection(channel):
    columns = build_observable_columns(
        channel
    )

    selected = []
    current = np.empty(
        (0, 15),
        dtype=float,
    )

    current_rank = 0

    while current_rank < 15:

        best_name = None
        best_matrix = None
        best_rank = current_rank

        for name in OBS_NAMES:

            if name in selected:
                continue

            candidate = np.vstack(
                [
                    current,
                    columns[name],
                ]
            )

            rank = matrix_rank(
                candidate
            )

            if rank > best_rank:
                best_rank = rank
                best_name = name
                best_matrix = candidate

        if best_name is None:
            break

        selected.append(best_name)
        current = best_matrix
        current_rank = best_rank

    return selected, current_rank


def verify_removal_minimality(
    channel,
    selected,
):
    failures = []

    for removed in selected:

        subset = [
            name
            for name in selected
            if name != removed
        ]

        columns = build_observable_columns(
            channel
        )

        M = np.vstack(
            [
                columns[name]
                for name in subset
            ]
        )

        rank = matrix_rank(M)

        if rank == 15:
            failures.append(
                removed
            )

    return failures


def main():

    print(
        "QDK-039 MINIMAL COMPLETENESS SEARCH"
    )
    print("=" * 50)
    print(
        f"Target rank: 15"
    )
    print(
        f"Total observables: "
        f"{len(OBS_NAMES)}"
    )

    results = {}

    for name, channel in CHANNELS.items():

        print()
        print(
            f"CHANNEL: {name}"
        )

        selected, rank = greedy_selection(
            channel
        )

        print(
            "Greedy subset:"
            f" {selected}"
        )

        print(
            f"Greedy rank: "
            f"{rank}/15"
        )

        removable = (
            verify_removal_minimality(
                channel,
                selected,
            )
        )

        minimal = (
            rank == 15
            and len(removable) == 0
        )

        print(
            f"Removal preserving rank-15: "
            f"{removable}"
        )

        print(
            f"Locally minimal: "
            f"{minimal}"
        )

        results[name] = {
            "selected_observables": selected,
            "rank": rank,
            "target_rank": 15,
            "locally_minimal": minimal,
            "removable_without_rank_loss": (
                removable
            ),
        }

    result = {
        "experiment": "QDK-039",
        "title": (
            "Minimal Observable Set for "
            "Quantum Fingerprint Completeness"
        ),
        "target_rank": 15,
        "observable_basis": OBS_NAMES,
        "noise_points": len(P_GRID),
        "results": results,
        "scientific_note": (
            "Greedy rank-increment search followed by "
            "single-removal minimality verification. "
            "Local minimality does not by itself prove "
            "global minimum cardinality."
        ),
    }

    out = Path("results/qdk_039")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-039-MINIMAL-COMPLETENESS.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("QDK-039 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
