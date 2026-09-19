import json
from pathlib import Path
import numpy as np

TOL = 1e-10

OUT = Path("results/qdk_042")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Pauli matrices
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
# CPTP single-qubit phase damping
#
# E(rho) =
# K0 rho K0^\dagger + K1 rho K1^\dagger
#
# K0 = sqrt(1-p/2) I
# K1 = sqrt(p/2) Z
#
# This gives:
# off-diagonal -> (1-p) off-diagonal
#
# Completeness:
# K0^\dagger K0 + K1^\dagger K1 = I
# ============================================================

def phase_kraus(p):
    K0 = np.sqrt(1.0 - p / 2.0) * I
    K1 = np.sqrt(p / 2.0) * Z

    return [K0, K1]


def apply_single_qubit_phase(rho, p, qubit):
    """
    Apply CPTP phase damping to one qubit.

    qubit = 0 or 1.
    """

    Ks = phase_kraus(p)

    result = np.zeros_like(rho, dtype=complex)

    for K in Ks:

        if qubit == 0:
            K4 = np.kron(K, I)
        else:
            K4 = np.kron(I, K)

        result += K4 @ rho @ K4.conj().T

    return result


def phase_damping_cptp(rho, p):
    """
    Apply phase damping independently to both qubits.
    """

    result = apply_single_qubit_phase(rho, p, 0)
    result = apply_single_qubit_phase(result, p, 1)

    return result


def nonlinear_phase_cptp(rho, p):
    """
    Same physical CPTP channel but nonlinear parameterization q=p^2.
    """

    q = p ** 2

    return phase_damping_cptp(rho, q)


# ============================================================
# Fingerprint
# ============================================================

def fingerprint(rho, channel, observable_indices=None):

    if observable_indices is None:
        selected = OBS
    else:
        selected = [
            OBS[i]
            for i in observable_indices
        ]

    values = []

    for p in P_GRID:

        rp = channel(rho, p)

        for _, O in selected:

            values.append(
                float(
                    np.real(
                        np.trace(O @ rp)
                    )
                )
            )

    return np.asarray(values)


# ============================================================
# Fingerprint matrix
# ============================================================

def build_matrix(channel, observable_indices=None):

    columns = []

    for _, B in OBS:

        columns.append(
            fingerprint(
                B,
                channel,
                observable_indices
            )
        )

    return np.asarray(columns).T


def numerical_rank(A):

    s = np.linalg.svd(
        A,
        compute_uv=False
    )

    threshold = (
        max(A.shape)
        * np.max(s)
        * np.finfo(float).eps
    )

    rank = int(
        np.sum(s > threshold)
    )

    return rank, s


# ============================================================
# CPTP validation
# ============================================================

def validate_kraus():

    max_tp_error = 0.0

    for p in P_GRID:

        for K in phase_kraus(p):

            pass

        Ks = phase_kraus(p)

        completeness = np.zeros(
            (2, 2),
            dtype=complex
        )

        for K in Ks:
            completeness += (
                K.conj().T @ K
            )

        error = np.linalg.norm(
            completeness - I
        )

        max_tp_error = max(
            max_tp_error,
            error
        )

    return max_tp_error


# ============================================================
# State validation
# ============================================================

def validate_trace_and_positivity(channel):

    rng = np.random.default_rng(42)

    max_trace_error = 0.0
    min_eigenvalue = np.inf

    for _ in range(100):

        A = (
            rng.normal(size=(4, 4))
            + 1j * rng.normal(size=(4, 4))
        )

        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        for p in P_GRID:

            rp = channel(rho, p)

            trace_error = abs(
                np.trace(rp) - 1.0
            )

            eig = np.linalg.eigvalsh(rp)

            max_trace_error = max(
                max_trace_error,
                trace_error
            )

            min_eigenvalue = min(
                min_eigenvalue,
                np.min(eig)
            )

    return {
        "max_trace_error": float(max_trace_error),
        "minimum_eigenvalue": float(min_eigenvalue),
        "trace_preserving": bool(
            max_trace_error < TOL
        ),
        "positive": bool(
            min_eigenvalue >= -TOL
        )
    }


# ============================================================
# Main analysis
# ============================================================

CHANNELS = {
    "phase_damping_cptp": phase_damping_cptp,
    "nonlinear_phase_cptp": nonlinear_phase_cptp,
}

results = {
    "experiment": "QDK-042",
    "title": "CPTP Phase-Damping Validation",
    "state_space_dimension": 15,
    "noise_points": len(P_GRID),
    "observables": [x[0] for x in OBS],
    "kraus_completeness_error": validate_kraus(),
    "channels": {}
}

print("=" * 70)
print("QDK-042 CPTP PHASE-DAMPING VALIDATION")
print("=" * 70)

print(
    "Maximum Kraus completeness error:",
    results["kraus_completeness_error"]
)

for name, channel in CHANNELS.items():

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    physical = validate_trace_and_positivity(channel)

    A = build_matrix(channel)

    rank, singular_values = numerical_rank(A)

    nullity = 15 - rank

    print("Matrix:", A.shape)
    print("Rank:", rank)
    print("Nullity:", nullity)
    print(
        "Minimum singular value:",
        singular_values[-1]
    )
    print(
        "Condition number:",
        singular_values[0] /
        singular_values[-1]
    )

    print(
        "Max trace error:",
        physical["max_trace_error"]
    )

    print(
        "Minimum eigenvalue:",
        physical["minimum_eigenvalue"]
    )

    print(
        "Injective:",
        rank == 15
    )

    results["channels"][name] = {
        "matrix_shape": list(A.shape),
        "rank": int(rank),
        "nullity": int(nullity),
        "injective": bool(rank == 15),
        "minimum_singular_value": float(
            singular_values[-1]
        ),
        "maximum_singular_value": float(
            singular_values[0]
        ),
        "condition_number": float(
            singular_values[0] /
            singular_values[-1]
        ),
        "physical_validation": physical,
    }


# ============================================================
# Restricted 11-observable test
# ============================================================

restricted_names = [
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
]

restricted_indices = [
    next(
        i
        for i, (name, _) in enumerate(OBS)
        if name == target
    )
    for target in restricted_names
]

A11 = build_matrix(
    phase_damping_cptp,
    restricted_indices
)

rank11, singular11 = numerical_rank(A11)

print()
print("=" * 70)
print("RESTRICTED CPTP PHASE-DAMPING")
print("=" * 70)

print("Observables:", restricted_names)
print("Matrix:", A11.shape)
print("Rank:", rank11)
print("Nullity:", 15 - rank11)

results["restricted_11"] = {
    "observables": restricted_names,
    "matrix_shape": list(A11.shape),
    "rank": int(rank11),
    "nullity": int(15 - rank11),
    "minimum_singular_value": float(
        singular11[-1]
    ),
    "injective": bool(rank11 == 15),
}

# ============================================================
# Save
# ============================================================

outfile = (
    OUT /
    "QDK-042-CPTP-PHASE-DAMPING.json"
)

with open(outfile, "w") as f:
    json.dump(
        results,
        f,
        indent=2
    )

print()
print("=" * 70)
print("QDK-042 COMPLETE")
print("Saved:", outfile)
print("=" * 70)
