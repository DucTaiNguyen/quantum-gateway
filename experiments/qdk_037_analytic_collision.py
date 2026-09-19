import json
from pathlib import Path

import numpy as np


SEED = 20260907
EPSILON = 0.05
TOLERANCE = 1e-10

rng = np.random.default_rng(SEED)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Restricted observable family deliberately used here.
OBS = {
    "XX": np.kron(X, X),
    "YY": np.kron(Y, Y),
    "ZZ": np.kron(Z, Z),
    "ZI": np.kron(Z, I),
    "IZ": np.kron(I, Z),
}

P_GRID = np.linspace(0.0, 1.0, 21)


def hermitian_basis():
    basis = []

    for a in [I, X, Y, Z]:
        for b in [I, X, Y, Z]:
            basis.append(np.kron(a, b))

    return basis


BASIS = hermitian_basis()


def depolarizing(rho, p):
    return (
        (1.0 - p) * rho
        + p * np.eye(4, dtype=complex) / 4.0
    )


def phase_damping(rho, p):
    g = np.sqrt(max(0.0, 1.0 - p))

    D = np.diag([1.0, g, g, 1.0])

    return D @ rho @ D


def nonlinear_depolarizing(rho, p):
    q = p**2

    return (
        (1.0 - q) * rho
        + q * np.eye(4, dtype=complex) / 4.0
    )


def nonlinear_phase(rho, p):
    q = p**2
    g = np.sqrt(max(0.0, 1.0 - q))

    D = np.diag([1.0, g, g, 1.0])

    return D @ rho @ D


CHANNELS = {
    "depolarizing": depolarizing,
    "phase_damping": phase_damping,
    "nonlinear_depolarizing": nonlinear_depolarizing,
    "nonlinear_phase": nonlinear_phase,
}


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


def vectorize_hermitian(H):
    return np.asarray(
        [
            float(
                np.real(
                    np.trace(B.conj().T @ H)
                )
            )
            for B in BASIS
        ]
    )


def reconstruct(v):
    H = np.zeros(
        (4, 4),
        dtype=complex,
    )

    for coefficient, B in zip(v, BASIS):
        H += coefficient * B / 4.0

    return H


def build_measurement_matrix(channel):
    columns = []

    for B in BASIS:
        fp = fingerprint(B, channel)
        columns.append(fp)

    return np.asarray(columns).T


def random_density_matrix():
    A = (
        rng.normal(size=(4, 4))
        + 1j * rng.normal(size=(4, 4))
    )

    rho = A @ A.conj().T

    return rho / np.trace(rho)


def main():
    print("QDK-037 ANALYTIC COLLISION CONSTRUCTION")
    print("=" * 50)

    results = {}

    for channel_name, channel in CHANNELS.items():

        M = build_measurement_matrix(channel)

        u, s, vh = np.linalg.svd(M)

        rank = int(
            np.sum(s > TOLERANCE)
        )

        nullity = M.shape[1] - rank

        print()
        print(f"CHANNEL: {channel_name}")
        print(f"Measurement matrix shape: {M.shape}")
        print(f"Rank: {rank}")
        print(f"Nullity: {nullity}")

        collision_found = False
        best_fp_error = float("inf")
        best_delta_norm = 0.0

        if nullity > 0:

            null_vector = vh[-1]

            Delta = reconstruct(
                null_vector
            )

            Delta = (
                Delta
                + Delta.conj().T
            ) / 2.0

            # Remove trace component.
            Delta -= (
                np.trace(Delta)
                / 4.0
            ) * np.eye(
                4,
                dtype=complex,
            )

            delta_norm = np.linalg.norm(
                Delta,
                ord="fro",
            )

            if delta_norm > TOLERANCE:
                Delta /= delta_norm

            rho = random_density_matrix()

            # Scale perturbation until positivity is preserved.
            eps = EPSILON

            for _ in range(20):

                rho_a = (
                    rho
                    + eps * Delta
                )

                rho_b = (
                    rho
                    - eps * Delta
                )

                min_a = np.min(
                    np.linalg.eigvalsh(rho_a)
                )

                min_b = np.min(
                    np.linalg.eigvalsh(rho_b)
                )

                if (
                    min_a >= -TOLERANCE
                    and min_b >= -TOLERANCE
                ):
                    break

                eps *= 0.5

            if (
                np.min(np.linalg.eigvalsh(rho_a))
                >= -TOLERANCE
                and
                np.min(np.linalg.eigvalsh(rho_b))
                >= -TOLERANCE
            ):

                fa = fingerprint(
                    rho_a,
                    channel,
                )

                fb = fingerprint(
                    rho_b,
                    channel,
                )

                fp_error = float(
                    np.linalg.norm(fa - fb)
                )

                state_distance = float(
                    np.linalg.norm(
                        rho_a - rho_b,
                        ord="fro",
                    )
                )

                best_fp_error = fp_error
                best_delta_norm = state_distance

                collision_found = (
                    fp_error < TOLERANCE
                    and state_distance > TOLERANCE
                )

                print(
                    "Fingerprint difference: "
                    f"{fp_error:.12e}"
                )

                print(
                    "State distance: "
                    f"{state_distance:.12e}"
                )

                print(
                    "Constructive collision: "
                    f"{collision_found}"
                )

        results[channel_name] = {
            "measurement_matrix_rank": rank,
            "measurement_matrix_nullity": nullity,
            "fingerprint_difference": best_fp_error,
            "state_distance": best_delta_norm,
            "constructive_collision": collision_found,
        }

    result = {
        "experiment": "QDK-037",
        "title": (
            "Analytic Collision Construction "
            "via Fingerprint Null Space"
        ),
        "seed": SEED,
        "epsilon": EPSILON,
        "tolerance": TOLERANCE,
        "observables": list(OBS.keys()),
        "noise_points": len(P_GRID),
        "channels": list(CHANNELS.keys()),
        "results": results,
        "scientific_note": (
            "A nonzero null space demonstrates that the "
            "restricted fingerprint map cannot be injective "
            "over the full operator space. A positive "
            "semidefinite constructive perturbation provides "
            "an explicit collision."
        ),
    }

    out = Path("results/qdk_037")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-037-ANALYTIC-COLLISION.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("QDK-037 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
