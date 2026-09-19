import json
from pathlib import Path

import numpy as np


SEED = 20260907
TRIALS = 500
TOLERANCE = 1e-8

rng = np.random.default_rng(SEED)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

OBS = [
    np.kron(X, X),
    np.kron(Y, Y),
    np.kron(Z, Z),
    np.kron(Z, I),
    np.kron(I, Z),
]

SPECTRUM = np.array(
    [0.50, 0.25, 0.15, 0.10],
    dtype=float,
)

P_GRID = np.linspace(0.0, 1.0, 21)


def random_unitary(n):
    A = (
        rng.normal(size=(n, n))
        + 1j * rng.normal(size=(n, n))
    )

    Q, R = np.linalg.qr(A)

    d = np.diag(R)

    phase = np.where(
        np.abs(d) > 1e-15,
        d / np.abs(d),
        1.0,
    )

    return Q @ np.diag(np.conjugate(phase))


def random_state():
    U = random_unitary(4)

    return (
        U
        @ np.diag(SPECTRUM)
        @ U.conj().T
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


def fingerprint(channel, rho):
    values = []

    for p in P_GRID:
        rp = channel(rho, p)

        for O in OBS:
            values.append(
                float(
                    np.real(
                        np.trace(O @ rp)
                    )
                )
            )

    return np.asarray(values)


def spectral_error(rho):
    eig = np.linalg.eigvalsh(rho)

    return float(
        np.max(
            np.abs(
                np.sort(eig)[::-1]
                - SPECTRUM
            )
        )
    )


def state_distance(rho_a, rho_b):
    return float(
        np.linalg.norm(rho_a - rho_b)
    )


def main():
    print("QDK-036 CONSTRUCTIVE COLLISION SEARCH")
    print("=" * 50)
    print(f"Trials: {TRIALS}")
    print(f"Tolerance: {TOLERANCE}")

    states = [
        random_state()
        for _ in range(TRIALS)
    ]

    max_spectral_error = max(
        spectral_error(rho)
        for rho in states
    )

    print(
        f"Maximum spectral error: "
        f"{max_spectral_error:.6e}"
    )

    results = {}

    for channel_name, channel in CHANNELS.items():

        print()
        print(f"CHANNEL: {channel_name}")

        fingerprints = [
            fingerprint(channel, rho)
            for rho in states
        ]

        fingerprints = np.asarray(
            fingerprints
        )

        best_fp_distance = float("inf")
        best_state_distance = 0.0
        best_pair = None

        collision_count = 0
        near_collision_count = 0

        for i in range(TRIALS):
            for j in range(i + 1, TRIALS):

                fp_distance = float(
                    np.linalg.norm(
                        fingerprints[i]
                        - fingerprints[j]
                    )
                )

                rho_distance = state_distance(
                    states[i],
                    states[j],
                )

                if (
                    fp_distance < TOLERANCE
                    and rho_distance > TOLERANCE
                ):
                    collision_count += 1

                if (
                    fp_distance < 1e-3
                    and rho_distance > 1e-2
                ):
                    near_collision_count += 1

                if fp_distance < best_fp_distance:
                    best_fp_distance = fp_distance
                    best_state_distance = rho_distance
                    best_pair = (i, j)

        exact_collision = collision_count > 0

        results[channel_name] = {
            "best_fingerprint_distance": (
                best_fp_distance
            ),
            "corresponding_state_distance": (
                best_state_distance
            ),
            "best_pair": list(best_pair),
            "exact_collision_found": (
                exact_collision
            ),
            "near_collision_count": (
                near_collision_count
            ),
            "collision_count": collision_count,
        }

        print(
            "Best fingerprint distance: "
            f"{best_fp_distance:.12e}"
        )

        print(
            "Corresponding state distance: "
            f"{best_state_distance:.12e}"
        )

        print(
            "Near collisions (<1e-3): "
            f"{near_collision_count}"
        )

        print(
            "Exact collisions: "
            f"{collision_count}"
        )

        print(
            "Counterexample found: "
            f"{exact_collision}"
        )

    result = {
        "experiment": "QDK-036",
        "title": (
            "Constructive Quantum Fingerprint "
            "Collision Search"
        ),
        "seed": SEED,
        "trials": TRIALS,
        "tolerance": TOLERANCE,
        "spectrum": SPECTRUM.tolist(),
        "noise_points": len(P_GRID),
        "observables": [
            "XX",
            "YY",
            "ZZ",
            "ZI",
            "IZ",
        ],
        "channels": list(CHANNELS.keys()),
        "maximum_spectral_error": (
            max_spectral_error
        ),
        "results": results,
        "scientific_note": (
            "Failure to find a collision in a finite "
            "random search does not establish injectivity. "
            "An exact collision would provide a constructive "
            "counterexample to injectivity for the tested "
            "fingerprint definition."
        ),
    }

    out = Path("results/qdk_036")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-036-COLLISION-SEARCH.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("QDK-036 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
