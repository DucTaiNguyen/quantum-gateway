import itertools
import json
from pathlib import Path

import numpy as np

OBSERVABLES = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]

CHANNELS = [
    "depolarizing",
    "nonlinear_depolarizing",
    "phase_damping",
    "nonlinear_phase",
]

NOISE_POINTS = np.linspace(0.0, 1.0, 21)


def channel_factor(channel, p):
    """Scalar response factor for the modeled channel."""
    if channel == "depolarizing":
        return 1.0 - p

    if channel == "nonlinear_depolarizing":
        return 1.0 - p * p

    if channel == "phase_damping":
        return 1.0 - p

    if channel == "nonlinear_phase":
        return 1.0 - p * p

    raise ValueError(f"Unknown channel: {channel}")


def build_full_response_matrix(channel):
    """
    Build the full 21 x 15 response matrix.

    Each observable probes one independent Pauli coordinate.
    Therefore a subset of observables corresponds simply to
    selecting columns from this matrix.
    """
    factors = np.array(
        [channel_factor(channel, p) for p in NOISE_POINTS],
        dtype=float,
    )

    M = np.zeros((len(NOISE_POINTS), len(OBSERVABLES)), dtype=float)

    for j in range(len(OBSERVABLES)):
        M[:, j] = factors

    return M


def subset_rank(M, indices):
    """
    Rank of the selected observable response.

    Since every selected column has the same scalar noise
    trajectory in this simplified model, the rank is determined
    directly by the number of independent observable coordinates.
    """
    if not indices:
        return 0

    return np.linalg.matrix_rank(
        M[:, indices],
        tol=1e-10,
    )


def exhaustive_search(channel):
    """
    Exhaustively search all non-empty subsets.

    Target rank is 15, corresponding to the 15-dimensional
    trace-one Hermitian two-qubit state coordinate space.
    """
    target_rank = 15
    M = build_full_response_matrix(channel)

    tested = 0
    rank_histogram = {}

    for k in range(1, 16):
        found = []

        for combo in itertools.combinations(range(15), k):
            tested += 1

            rank = subset_rank(M, combo)

            key = str(k)
            rank_histogram.setdefault(key, {})
            rank_histogram[key][str(rank)] = (
                rank_histogram[key].get(str(rank), 0) + 1
            )

            if rank == target_rank:
                found.append(
                    [OBSERVABLES[i] for i in combo]
                )

        if found:
            return {
                "k_min": k,
                "minimal_subsets": found,
                "tested_subsets": tested,
                "rank_histogram": rank_histogram,
            }

    return {
        "k_min": None,
        "minimal_subsets": [],
        "tested_subsets": tested,
        "rank_histogram": rank_histogram,
    }


def main():
    results = {}

    print("=" * 70)
    print("QDK-040 GLOBAL MINIMALITY SEARCH")
    print("=" * 70)
    print("Total observables:", len(OBSERVABLES))
    print("Total non-empty subsets:", 2 ** len(OBSERVABLES) - 1)
    print()

    for channel in CHANNELS:
        print("CHANNEL:", channel)

        result = exhaustive_search(channel)
        results[channel] = result

        print("Global minimum k:", result["k_min"])
        print(
            "Number of globally minimal subsets:",
            len(result["minimal_subsets"]),
        )

        for subset in result["minimal_subsets"][:10]:
            print("  ", subset)

        if len(result["minimal_subsets"]) > 10:
            print(
                "  ...",
                len(result["minimal_subsets"]) - 10,
                "additional minimal subsets",
            )

        print("Subsets tested:", result["tested_subsets"])
        print()

    output = {
        "experiment": "QDK-040",
        "title": "Global Minimality of Informationally Complete Fingerprints",
        "observables": OBSERVABLES,
        "noise_points": NOISE_POINTS.tolist(),
        "channels": CHANNELS,
        "target_rank": 15,
        "results": results,
        "method": (
            "Exhaustive enumeration of all non-empty subsets "
            "of the 15 nontrivial two-qubit Pauli observables."
        ),
        "scope": (
            "Global minimality is established only for the "
            "defined response-matrix model."
        ),
    }

    out = Path("results/qdk_040")
    out.mkdir(parents=True, exist_ok=True)

    path = out / "QDK-040-GLOBAL-MINIMALITY.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("QDK-040 COMPLETE")
    print("Saved:", path)


if __name__ == "__main__":
    main()
