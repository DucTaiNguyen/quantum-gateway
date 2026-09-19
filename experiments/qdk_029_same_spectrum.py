import json
from pathlib import Path
import numpy as np

SEED = 20260907
TRIALS = 100

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

P_GRID = np.linspace(0.0, 1.0, 21)

SPECTRUM = np.array(
    [0.50, 0.25, 0.15, 0.10],
    dtype=float,
)


def random_unitary(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)

    d = np.diag(R)
    d = np.where(
        np.abs(d) > 1e-15,
        d / np.abs(d),
        1.0,
    )

    return Q @ np.diag(np.conjugate(d))


def random_state():
    U = random_unitary(4)

    return U @ np.diag(SPECTRUM) @ U.conj().T


def depolarize(rho, p):
    return (
        (1.0 - p) * rho
        + p * np.eye(4, dtype=complex) / 4.0
    )


def fingerprint(rho):
    values = []

    for p in P_GRID:
        rho_p = depolarize(rho, p)

        for O in OBS:
            values.append(
                float(
                    np.real(
                        np.trace(O @ rho_p)
                    )
                )
            )

    return np.asarray(values)


def spectral_error(rho):
    eig = np.sort(
        np.real(
            np.linalg.eigvalsh(rho)
        )
    )

    target = np.sort(SPECTRUM)

    return float(
        np.max(np.abs(eig - target))
    )


def main():
    states = [
        random_state()
        for _ in range(TRIALS)
    ]

    spectral_errors = [
        spectral_error(rho)
        for rho in states
    ]

    fingerprint_distances = []

    state_distances = []

    for i in range(TRIALS):
        for j in range(i + 1, TRIALS):
            fp_i = fingerprint(states[i])
            fp_j = fingerprint(states[j])

            fingerprint_distances.append(
                float(
                    np.linalg.norm(fp_i - fp_j)
                )
            )

            state_distances.append(
                float(
                    np.linalg.norm(
                        states[i] - states[j],
                        ord="fro",
                    )
                )
            )

    result = {
        "experiment": "QDK-029",
        "title": "Same-Spectrum Structural Test",
        "seed": SEED,
        "trials": TRIALS,
        "fixed_spectrum": SPECTRUM.tolist(),
        "maximum_spectral_error": max(spectral_errors),
        "minimum_state_distance": min(state_distances),
        "maximum_state_distance": max(state_distances),
        "minimum_fingerprint_distance": min(
            fingerprint_distances
        ),
        "median_fingerprint_distance": float(
            np.median(fingerprint_distances)
        ),
        "maximum_fingerprint_distance": max(
            fingerprint_distances
        ),
        "interpretation": (
            "States with identical spectra can nevertheless "
            "produce different noise-response fingerprints "
            "under the tested observable/channel model."
        ),
    }

    out = Path("results/qdk_029")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-029-SAME-SPECTRUM.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-029 SAME-SPECTRUM STRUCTURAL TEST")
    print("=" * 50)
    print(
        "Maximum spectral error:",
        f"{max(spectral_errors):.12e}",
    )
    print(
        "Minimum state distance:",
        f"{min(state_distances):.12e}",
    )
    print(
        "Minimum fingerprint distance:",
        f"{min(fingerprint_distances):.12e}",
    )
    print(
        "Median fingerprint distance:",
        f"{np.median(fingerprint_distances):.12e}",
    )
    print(
        "Maximum fingerprint distance:",
        f"{max(fingerprint_distances):.12e}",
    )
    print("QDK-029 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
