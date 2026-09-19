import itertools
import json
from pathlib import Path

import numpy as np


TOLERANCE = 1e-10

I = np.eye(2, dtype=complex)

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

P_GRID = np.linspace(
    0.0,
    1.0,
    21,
)


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
    g = np.sqrt(
        max(
            0.0,
            1.0 - p,
        )
    )

    D = np.diag(
        [1.0, g, g, 1.0]
    )

    return D @ rho @ D


def nonlinear_phase(rho, p):
    q = p**2

    g = np.sqrt(
        max(
            0.0,
            1.0 - q,
        )
    )

    D = np.diag(
        [1.0, g, g, 1.0]
    )

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
                (
                    a_name + b_name,
                    np.kron(a, b),
                )
            )

    return result


BASIS = basis()


def fingerprint(rho, channel, observable_names):
    values = []

    for p in P_GRID:
        rp = channel(
            rho,
            p,
        )

        for name in observable_names:
            O = OBS[name]

            values.append(
                float(
                    np.real(
                        np.trace(
                            O @ rp
                        )
                    )
                )
            )

    return np.asarray(
        values,
        dtype=float,
    )


def build_full_matrix(channel):
    """
    Exact QDK-038 measurement matrix.

    Rows:
        noise point x observable

    Columns:
        15 non-identity Pauli basis directions.
    """

    columns = []

    for name, B in BASIS:

        if name == "II":
            continue

        columns.append(
            fingerprint(
                B,
                channel,
                OBS_NAMES,
            )
        )

    return np.asarray(
        columns,
        dtype=float,
    ).T


def observable_row_indices(
    subset_indices,
):
    """
    Map a subset of observables to the corresponding
    rows in the QDK-038 fingerprint matrix.

    QDK-038 ordering is:

        p0: obs0 ... obs14
        p1: obs0 ... obs14
        ...
        p20: obs0 ... obs14
    """

    indices = []

    n_obs = len(OBS_NAMES)

    for p_index in range(
        len(P_GRID)
    ):
        base = (
            p_index
            * n_obs
        )

        for obs_index in subset_indices:
            indices.append(
                base + obs_index
            )

    return np.asarray(
        indices,
        dtype=int,
    )


def subset_rank(
    full_matrix,
    subset_indices,
):
    rows = observable_row_indices(
        subset_indices
    )

    M = full_matrix[rows, :]

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

    return rank


def exhaustive_search(
    channel_name,
    channel,
):
    print()
    print(
        f"CHANNEL: {channel_name}"
    )

    full_matrix = build_full_matrix(
        channel
    )

    full_rank = int(
        np.linalg.matrix_rank(
            full_matrix,
            tol=TOLERANCE,
        )
    )

    print(
        "Full fingerprint matrix:",
        full_matrix.shape,
    )

    print(
        "Full rank:",
        f"{full_rank}/15",
    )

    if full_rank != 15:
        raise RuntimeError(
            "QDK-040B consistency failure: "
            "full QDK-038 matrix does not have rank 15."
        )

    tested = 0

    rank_histogram = {}

    globally_minimal = []

    for k in range(
        1,
        len(OBS_NAMES) + 1,
    ):
        subset_count = 1
        for i in range(k):
            subset_count = subset_count * (15 - i) // (i + 1)

        print(
            f"  Searching k={k} "
            f"({subset_count} subsets)..."
        )

        count_rank_15 = 0

        for subset in itertools.combinations(
            range(len(OBS_NAMES)),
            k,
        ):
            tested += 1

            rank = subset_rank(
                full_matrix,
                subset,
            )

            rank_histogram.setdefault(
                str(k),
                {},
            )

            rank_histogram[
                str(k)
            ][str(rank)] = (
                rank_histogram[
                    str(k)
                ].get(
                    str(rank),
                    0,
                )
                + 1
            )

            if rank == 15:
                count_rank_15 += 1

                globally_minimal.append(
                    [
                        OBS_NAMES[i]
                        for i in subset
                    ]
                )

        if count_rank_15 > 0:
            return {
                "k_min": k,
                "globally_minimal_subset_count":
                    count_rank_15,
                "globally_minimal_subsets":
                    globally_minimal,
                "tested_subsets": tested,
                "rank_histogram":
                    rank_histogram,
                "full_matrix_shape":
                    list(full_matrix.shape),
                "full_rank":
                    full_rank,
            }

    return {
        "k_min": None,
        "globally_minimal_subset_count": 0,
        "globally_minimal_subsets": [],
        "tested_subsets": tested,
        "rank_histogram": rank_histogram,
        "full_matrix_shape":
            list(full_matrix.shape),
        "full_rank":
            full_rank,
    }


def main():

    print(
        "QDK-040B "
        "GLOBAL MINIMALITY"
    )
    print(
        "=" * 70
    )

    print(
        "Observables:",
        len(OBS_NAMES),
    )

    print(
        "Noise points:",
        len(P_GRID),
    )

    print(
        "Total non-empty subsets:",
        2 ** len(OBS_NAMES) - 1,
    )

    results = {}

    for channel_name, channel in CHANNELS.items():

        result = exhaustive_search(
            channel_name,
            channel,
        )

        results[
            channel_name
        ] = result

        print(
            "  GLOBAL MINIMUM k:",
            result["k_min"],
        )

        print(
            "  Globally minimal subsets:",
            result[
                "globally_minimal_subset_count"
            ],
        )

        if result[
            "globally_minimal_subsets"
        ]:
            for subset in result[
                "globally_minimal_subsets"
            ][:20]:
                print(
                    "   ",
                    subset,
                )

        print(
            "  Subsets tested:",
            result[
                "tested_subsets"
            ],
        )

    output = {
        "experiment": "QDK-040B",
        "title": (
            "Global Minimality of "
            "Quantum Noise Fingerprint"
        ),
        "method": (
            "Exact exhaustive subset search "
            "using the QDK-038 fingerprint "
            "matrix construction."
        ),
        "observable_count":
            len(OBS_NAMES),
        "observables":
            OBS_NAMES,
        "noise_points":
            len(P_GRID),
        "maximum_possible_rank":
            15,
        "total_nonempty_subsets":
            2 ** len(OBS_NAMES) - 1,
        "tolerance":
            TOLERANCE,
        "results":
            results,
        "scientific_scope": (
            "Global minimality is established "
            "only for the explicitly defined "
            "QDK-038 modeled fingerprint map. "
            "It does not by itself establish "
            "a universal physical law."
        ),
    }

    out = Path(
        "results/qdk_040b"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-040B-GLOBAL-MINIMALITY.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "QDK-040B COMPLETE"
    )

    print(
        f"Saved: {path}"
    )


if __name__ == "__main__":
    main()
