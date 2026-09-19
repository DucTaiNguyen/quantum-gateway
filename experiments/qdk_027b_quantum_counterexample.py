import json
from pathlib import Path
import numpy as np

SEED = 20260907
TRIALS = 300
P_GRID = np.linspace(0.0, 1.0, 11)

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

EIG = np.array([0.50, 0.25, 0.15, 0.10], dtype=float)


def random_unitary(n, rng):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    phases = np.diag(R)
    phases = np.where(np.abs(phases) > 1e-15,
                      phases / np.abs(phases), 1.0)
    return Q @ np.diag(np.conjugate(phases))


def random_density_matrix():
    U = random_unitary(4, rng)
    return U @ np.diag(EIG) @ U.conj().T


def depolarize(rho, p):
    return (1.0 - p) * rho + p * np.eye(4, dtype=complex) / 4.0


def fingerprint(rho):
    values = []
    for p in P_GRID:
        rp = depolarize(rho, p)
        values.extend(
            float(np.real(np.trace(o @ rp)))
            for o in OBS
        )
    return np.asarray(values, dtype=float)


def state_distance(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))


def fingerprint_distance(a, b):
    return float(np.linalg.norm(a - b))


def main():
    states = []
    fingerprints = []

    for _ in range(TRIALS):
        rho = random_density_matrix()
        states.append(rho)
        fingerprints.append(fingerprint(rho))

    best = None
    best_fp = float("inf")

    for i in range(TRIALS):
        for j in range(i + 1, TRIALS):
            sd = state_distance(states[i], states[j])

            if sd < 1e-2:
                continue

            fd = fingerprint_distance(
                fingerprints[i],
                fingerprints[j],
            )

            if fd < best_fp:
                best_fp = fd
                best = {
                    "i": i,
                    "j": j,
                    "state_distance_frobenius": sd,
                    "fingerprint_distance": fd,
                }

    counterexample = (
        best is not None
        and best["state_distance_frobenius"] > 1e-2
        and best["fingerprint_distance"] < 1e-8
    )

    result = {
        "experiment": "QDK-027B",
        "title": "Quantum Counterexample Search",
        "seed": SEED,
        "trials": TRIALS,
        "noise_model": "two-qubit depolarizing channel",
        "noise_points": len(P_GRID),
        "observables": ["XX", "YY", "ZZ", "ZI", "IZ"],
        "fixed_spectrum": EIG.tolist(),
        "best_pair": best,
        "counterexample_found": bool(counterexample),
        "interpretation": (
            "A random-search collision was found."
            if counterexample
            else
            "No exact fingerprint collision was found in this finite random search; "
            "this does not establish absence of counterexamples."
        ),
    }

    out = Path("results/qdk_027b")
    out.mkdir(parents=True, exist_ok=True)

    path = out / "QDK-027B-QUANTUM-COUNTEREXAMPLE.json"
    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-027B QUANTUM COUNTEREXAMPLE SEARCH")
    print("=" * 50)
    print(f"Trials: {TRIALS}")
    print(f"Best fingerprint distance: {best_fp:.12e}")
    print(f"Counterexample found: {counterexample}")
    print("QDK-027B COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
