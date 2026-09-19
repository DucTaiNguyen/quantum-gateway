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

SPECTRUM = np.array(
    [0.50, 0.25, 0.15, 0.10]
)

P_GRID = np.linspace(0.01, 1.0, 20)


def random_unitary(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)

    d = np.diag(R)

    d = np.where(
        np.abs(d) > 1e-15,
        d / np.abs(d),
        1.0,
    )

    return Q @ np.diag(
        np.conjugate(d)
    )


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
        + p * np.eye(4, dtype=complex) / 4
    )


def nonlinear_depolarizing(rho, p):
    q = p ** 2

    return (
        (1.0 - q) * rho
        + q * np.eye(4, dtype=complex) / 4
    )


def phase_damping(rho, p):
    g = np.sqrt(
        max(0.0, 1.0 - p)
    )

    D = np.diag(
        [1.0, g, g, 1.0]
    )

    return D @ rho @ D


def nonlinear_phase(rho, p):
    q = p ** 2

    g = np.sqrt(
        max(0.0, 1.0 - q)
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


def response(channel, rho, p):
    rp = channel(rho, p)

    return np.asarray([
        float(
            np.real(
                np.trace(O @ rp)
            )
        )
        for O in OBS
    ])


def fingerprint(channel, rho):
    return np.concatenate([
        response(channel, rho, p)
        for p in P_GRID
    ])


def main():
    states = [
        random_state()
        for _ in range(TRIALS)
    ]

    results = {}

    for name, channel in CHANNELS.items():

        distances = []

        for i in range(TRIALS):
            for j in range(i + 1, TRIALS):

                fi = fingerprint(
                    channel,
                    states[i],
                )

                fj = fingerprint(
                    channel,
                    states[j],
                )

                d = float(
                    np.linalg.norm(fi - fj)
                )

                distances.append(d)

        distances = np.asarray(
            distances
        )

        results[name] = {
            "minimum": float(
                np.min(distances)
            ),
            "median": float(
                np.median(distances)
            ),
            "mean": float(
                np.mean(distances)
            ),
            "maximum": float(
                np.max(distances)
            ),
            "nonzero_fraction": float(
                np.mean(distances > 1e-12)
            ),
        }

    result = {
        "experiment": "QDK-035",
        "title": (
            "Structure-Response Invariance "
            "Under Noise Parameterization"
        ),
        "seed": SEED,
        "trials": TRIALS,
        "fixed_spectrum": SPECTRUM.tolist(),
        "noise_points": len(P_GRID),
        "results": results,
        "interpretation": (
            "Tests whether distinct same-spectrum quantum "
            "states remain distinguishable by their complete "
            "noise-response fingerprints under different "
            "noise parameterizations."
        ),
    }

    out = Path("results/qdk_035")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-035-STRUCTURE-RESPONSE.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-035 STRUCTURE-RESPONSE INVARIANCE")
    print("=" * 50)

    for name, r in results.items():
        print(
            f"{name}: "
            f"min={r['minimum']:.6f}, "
            f"median={r['median']:.6f}, "
            f"mean={r['mean']:.6f}, "
            f"max={r['maximum']:.6f}"
        )

    print("QDK-035 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
