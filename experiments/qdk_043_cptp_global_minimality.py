import json
from pathlib import Path
from itertools import combinations
import numpy as np

TOL = 1e-10

OUT = Path("results/qdk_043")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Pauli basis
# ============================================================

I = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

PAULIS = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
}

LABELS = ["I", "X", "Y", "Z"]

BASIS = []

for a in LABELS:
    for b in LABELS:
        BASIS.append(
            (a + b, np.kron(PAULIS[a], PAULIS[b]))
        )

OBS = [
    (name, op)
    for name, op in BASIS
    if name != "II"
]

P_GRID = np.linspace(0.0, 1.0, 21)

I4 = np.eye(4, dtype=complex)

# ============================================================
# CPTP phase damping
# ============================================================

def phase_kraus(p):

    K0 = np.sqrt(1.0 - p / 2.0) * I
    K1 = np.sqrt(p / 2.0) * Z

    return [K0, K1]


def apply_single_qubit_phase(rho, p, qubit):

    result = np.zeros_like(
        rho,
        dtype=complex
    )

    for K in phase_kraus(p):

        if qubit == 0:
            K4 = np.kron(K, I)
        else:
            K4 = np.kron(I, K)

        result += (
            K4
            @ rho
            @ K4.conj().T
        )

    return result


def phase_damping_cptp(rho, p):

    result = apply_single_qubit_phase(
        rho,
        p,
        0
    )

    result = apply_single_qubit_phase(
        result,
        p,
        1
    )

    return result


def nonlinear_phase_cptp(rho, p):

    return phase_damping_cptp(
        rho,
        p ** 2
    )


CHANNELS = {
    "phase_damping_cptp": phase_damping_cptp,
    "nonlinear_phase_cptp": nonlinear_phase_cptp,
}

# ============================================================
# Fingerprint
# ============================================================

def fingerprint(
    rho,
    channel,
    observable_indices
):

    values = []

    for p in P_GRID:

        rp = channel(
            rho,
            p
        )

        for idx in observable_indices:

            O = OBS[idx][1]

            values.append(
                float(
                    np.real(
                        np.trace(O @ rp)
                    )
                )
            )

    return np.asarray(values)


# ============================================================
# Build complete 15-coordinate matrix
# ============================================================

def build_full_matrix(channel):

    columns = []

    for _, B in OBS:

        columns.append(
            fingerprint(
                B,
                channel,
                range(len(OBS))
            )
        )

    return np.asarray(columns).T


# ============================================================
# Select observable rows
# ============================================================

def observable_rows(
    observable_indices
):

    rows = []

    n_obs = len(OBS)

    for p_index in range(len(P_GRID)):

        base = p_index * n_obs

        for obs_index in observable_indices:

            rows.append(
                base + obs_index
            )

    return rows


# ============================================================
# Rank
# ============================================================

def matrix_rank(A):

    s = np.linalg.svd(
        A,
        compute_uv=False
    )

    if len(s) == 0:
        return 0

    threshold = (
        max(A.shape)
        * np.max(s)
        * np.finfo(float).eps
    )

    return int(
        np.sum(s > threshold)
    )


# ============================================================
# Exhaustive search
# ============================================================

results = {
    "experiment": "QDK-043",
    "title": (
        "Global Minimality of CPTP Quantum Noise "
        "Fingerprints"
    ),
    "state_space_dimension": 15,
    "observable_count": 15,
    "noise_points": len(P_GRID),
    "channels": {}
}

print("=" * 70)
print("QDK-043 CPTP GLOBAL MINIMALITY")
print("=" * 70)

print(
    "Total possible non-empty subsets:",
    2 ** len(OBS) - 1
)

for channel_name, channel in CHANNELS.items():

    print()
    print("=" * 70)
    print("CHANNEL:", channel_name)
    print("=" * 70)

    A = build_full_matrix(channel)

    full_rank = matrix_rank(A)

    print(
        "Full matrix:",
        A.shape
    )

    print(
        "Full rank:",
        full_rank,
        "/ 15"
    )

    channel_result = {
        "full_matrix_shape": list(A.shape),
        "full_rank": int(full_rank),
        "global_minimum_k": None,
        "minimal_subsets": [],
        "subsets_tested": 0,
    }

    if full_rank < 15:

        print(
            "WARNING: full fingerprint is not complete."
        )

        results["channels"][channel_name] = channel_result

        continue

    found = False

    for k in range(1, len(OBS) + 1):

        count_k = 0
        successful = []

        print(
            f"Searching k={k} "
            f"({len(list(combinations(range(15), k)))} subsets)..."
        )

        for subset in combinations(
            range(len(OBS)),
            k
        ):

            count_k += 1
            channel_result["subsets_tested"] += 1

            rows = observable_rows(
                subset
            )

            submatrix = A[rows, :]

            rank = matrix_rank(
                submatrix
            )

            if rank == 15:

                successful.append(
                    [OBS[i][0] for i in subset]
                )

        if successful:

            found = True

            channel_result[
                "global_minimum_k"
            ] = k

            channel_result[
                "minimal_subsets"
            ] = successful

            print()
            print(
                "GLOBAL MINIMUM k:",
                k
            )

            print(
                "Number of globally minimal subsets:",
                len(successful)
            )

            for subset in successful:
                print(
                    " ",
                    subset
                )

            print(
                "Subsets tested:",
                channel_result[
                    "subsets_tested"
                ]
            )

            break

    results["channels"][
        channel_name
    ] = channel_result


# ============================================================
# Save
# ============================================================

outfile = (
    OUT /
    "QDK-043-CPTP-GLOBAL-MINIMALITY.json"
)

with open(outfile, "w") as f:

    json.dump(
        results,
        f,
        indent=2
    )

print()
print("=" * 70)
print("QDK-043 COMPLETE")
print("Saved:", outfile)
print("=" * 70)
