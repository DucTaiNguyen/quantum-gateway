import json
from pathlib import Path
import numpy as np
from itertools import combinations

TOL = 1e-10
OUT = Path("results/qdk_041")
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
        BASIS.append((a + b, np.kron(PAULIS[a], PAULIS[b])))

OBS = [(name, op) for name, op in BASIS if name != "II"]

P_GRID = np.linspace(0.0, 1.0, 21)

# ============================================================
# Noise maps
# ============================================================

def depolarizing(rho, p):
    return (1.0 - p) * rho + p * I4 / 4.0


def nonlinear_depolarizing(rho, p):
    q = p ** 2
    return (1.0 - q) * rho + q * I4 / 4.0


def phase_damping_model(rho, p):
    """
    Exact model used in QDK-038.

    IMPORTANT:
    This is the same simplified map used in QDK-038.
    It is retained here for reproducibility, but it should
    not be interpreted as a general CPTP physical channel.
    """
    g = np.sqrt(max(0.0, 1.0 - p))

    D = np.diag([
        1.0,
        g,
        g,
        1.0
    ]).astype(complex)

    return D @ rho @ D


def nonlinear_phase_model(rho, p):
    q = p ** 2
    g = np.sqrt(max(0.0, 1.0 - q))

    D = np.diag([
        1.0,
        g,
        g,
        1.0
    ]).astype(complex)

    return D @ rho @ D


I4 = np.eye(4, dtype=complex)

CHANNELS = {
    "depolarizing": depolarizing,
    "nonlinear_depolarizing": nonlinear_depolarizing,
    "phase_damping_model": phase_damping_model,
    "nonlinear_phase_model": nonlinear_phase_model,
}

# ============================================================
# Fingerprint matrix
# ============================================================

def fingerprint(rho, channel, observable_indices=None):
    values = []

    selected = OBS if observable_indices is None else [
        OBS[i] for i in observable_indices
    ]

    for p in P_GRID:
        rp = channel(rho, p)

        for _, O in selected:
            values.append(float(np.real(np.trace(O @ rp))))

    return np.asarray(values)


def build_matrix(channel, observable_indices=None):
    """
    Columns correspond to the 15 non-identity Pauli
    coordinates of the traceless Hermitian state space.
    """

    columns = []

    for name, B in OBS:
        rho = B.copy()
        columns.append(
            fingerprint(rho, channel, observable_indices)
        )

    return np.asarray(columns).T


def numerical_rank(A):
    s = np.linalg.svd(A, compute_uv=False)

    if len(s) == 0:
        return 0, s

    threshold = max(A.shape) * np.max(s) * np.finfo(float).eps

    return int(np.sum(s > threshold)), s


# ============================================================
# Null-space construction
# ============================================================

def nullspace(A):
    U, S, Vh = np.linalg.svd(A, full_matrices=True)

    rank = numerical_rank(A)[0]

    if rank == A.shape[1]:
        return np.empty((A.shape[1], 0))

    return Vh[rank:].conj().T


def pauli_operator_from_coordinates(v):
    """
    Reconstruct traceless Hermitian operator
    from Pauli coordinates.
    """

    Delta = np.zeros((4, 4), dtype=complex)

    for coeff, (_, B) in zip(v, OBS):
        Delta += coeff * B

    return Delta


def make_collision(A):
    """
    Construct rho_plus and rho_minus from a null direction.

    We normalize Delta and choose epsilon adaptively
    so positivity is preserved.
    """

    N = nullspace(A)

    if N.shape[1] == 0:
        return None

    v = np.real(N[:, 0])

    Delta = pauli_operator_from_coordinates(v)

    # Normalize operator norm
    eig = np.linalg.eigvalsh(Delta)

    scale = np.max(np.abs(eig))

    if scale < TOL:
        return None

    Delta = Delta / scale

    # I/4 +/- epsilon Delta >= 0
    # Conservative epsilon.
    epsilon = 0.20

    rho_plus = I4 / 4.0 + epsilon * Delta
    rho_minus = I4 / 4.0 - epsilon * Delta

    eig_plus = np.linalg.eigvalsh(rho_plus)
    eig_minus = np.linalg.eigvalsh(rho_minus)

    if np.min(eig_plus) < -TOL or np.min(eig_minus) < -TOL:
        # Find maximum safe epsilon.
        epsilon_max = 0.0

        for candidate in np.linspace(1e-4, 0.24, 2400):
            rp = I4 / 4.0 + candidate * Delta
            rm = I4 / 4.0 - candidate * Delta

            if (
                np.min(np.linalg.eigvalsh(rp)) >= -TOL
                and
                np.min(np.linalg.eigvalsh(rm)) >= -TOL
            ):
                epsilon_max = candidate

        if epsilon_max == 0:
            return None

        epsilon = 0.5 * epsilon_max

        rho_plus = I4 / 4.0 + epsilon * Delta
        rho_minus = I4 / 4.0 - epsilon * Delta

    return rho_plus, rho_minus, Delta, epsilon


# ============================================================
# Analysis
# ============================================================

results = {
    "experiment": "QDK-041",
    "title": "Formal Injectivity and Identifiability Boundary",
    "state_space_dimension": 15,
    "noise_points": len(P_GRID),
    "observables": [x[0] for x in OBS],
    "channels": {},
}

for channel_name, channel in CHANNELS.items():

    print()
    print("=" * 70)
    print(channel_name)
    print("=" * 70)

    A = build_matrix(channel)

    rank, singular_values = numerical_rank(A)
    nullity = A.shape[1] - rank

    print("Fingerprint matrix:", A.shape)
    print("Rank:", rank)
    print("Nullity:", nullity)
    print("Minimum singular value:", singular_values[-1])

    entry = {
        "matrix_shape": list(A.shape),
        "rank": int(rank),
        "nullity": int(nullity),
        "injective": bool(rank == 15),
        "minimum_singular_value": float(singular_values[-1]),
        "maximum_singular_value": float(singular_values[0]),
        "condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
    }

    # Exact full-map injectivity check
    if rank == 15:
        print("RESULT: INJECTIVE")
        print("Kernel dimension = 0")

        entry["theorem_status"] = (
            "For the modeled linear fingerprint map, "
            "rank 15 implies trivial kernel and injectivity "
            "on the 15-dimensional traceless Hermitian space."
        )

    else:
        print("RESULT: NON-INJECTIVE")

        collision = make_collision(A)

        if collision is not None:
            rho_plus, rho_minus, Delta, epsilon = collision

            fp_plus = fingerprint(rho_plus, channel)
            fp_minus = fingerprint(rho_minus, channel)

            state_distance = np.linalg.norm(
                rho_plus - rho_minus
            )

            fingerprint_distance = np.linalg.norm(
                fp_plus - fp_minus
            )

            print("Constructive collision:")
            print("  epsilon =", epsilon)
            print("  state distance =", state_distance)
            print("  fingerprint distance =", fingerprint_distance)

            entry["constructive_collision"] = {
                "epsilon": float(epsilon),
                "state_distance": float(state_distance),
                "fingerprint_distance": float(
                    fingerprint_distance
                ),
                "trace_plus": float(np.real(np.trace(rho_plus))),
                "trace_minus": float(np.real(np.trace(rho_minus))),
                "min_eigenvalue_plus": float(
                    np.min(np.linalg.eigvalsh(rho_plus))
                ),
                "min_eigenvalue_minus": float(
                    np.min(np.linalg.eigvalsh(rho_minus))
                ),
            }

    results["channels"][channel_name] = entry


# ============================================================
# Explicit restricted-map boundary
# ============================================================

# Example: phase-damping model with 11 observables.
# Since QDK-040B found k_min = 12, an 11-observable
# restriction must be rank-deficient.

phase_index = list(range(len(OBS)))

restricted_indices = phase_index[:11]

A_restricted = build_matrix(
    CHANNELS["phase_damping_model"],
    restricted_indices
)

r_restricted, s_restricted = numerical_rank(A_restricted)

print()
print("=" * 70)
print("RESTRICTED PHASE-DAMPING MAP")
print("=" * 70)
print("Observables:", [OBS[i][0] for i in restricted_indices])
print("Matrix:", A_restricted.shape)
print("Rank:", r_restricted)
print("Nullity:", 15 - r_restricted)

restricted_collision = make_collision(A_restricted)

if restricted_collision is not None:

    rp, rm, Delta, eps = restricted_collision

    fp = fingerprint(
        rp,
        CHANNELS["phase_damping_model"],
        restricted_indices
    )

    fm = fingerprint(
        rm,
        CHANNELS["phase_damping_model"],
        restricted_indices
    )

    results["restricted_boundary"] = {
        "channel": "phase_damping_model",
        "observable_count": len(restricted_indices),
        "rank": int(r_restricted),
        "nullity": int(15 - r_restricted),
        "collision": {
            "epsilon": float(eps),
            "state_distance": float(np.linalg.norm(rp - rm)),
            "fingerprint_distance": float(np.linalg.norm(fp - fm)),
            "min_eigenvalue_plus": float(
                np.min(np.linalg.eigvalsh(rp))
            ),
            "min_eigenvalue_minus": float(
                np.min(np.linalg.eigvalsh(rm))
            ),
        }
    }

    print(
        "Collision fingerprint distance:",
        np.linalg.norm(fp - fm)
    )

# ============================================================
# Save
# ============================================================

out_file = OUT / "QDK-041-FORMAL-INJECTIVITY.json"

with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 70)
print("QDK-041 COMPLETE")
print("Saved:", out_file)
print("=" * 70)
