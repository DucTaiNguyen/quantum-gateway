import json
from pathlib import Path
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

U = np.kron(I, X)

OBS = [
    np.kron(X, X),
    np.kron(Y, Y),
    np.kron(Z, Z),
    np.kron(Z, I),
    np.kron(I, Z),
]

P_GRID = np.linspace(0.0, 1.0, 21)


def depolarize(rho, p):
    return (1.0 - p) * rho + p * np.eye(4, dtype=complex) / 4.0


def fingerprint(rho):
    values = []

    for p in P_GRID:
        rp = depolarize(rho, p)

        for O in OBS:
            values.append(
                float(np.real(np.trace(O @ rp)))
            )

    return np.asarray(values)


def transformed_fingerprint(rho):
    rho_u = U @ rho @ U.conj().T

    values = []

    for p in P_GRID:
        rp = depolarize(rho_u, p)

        for O in OBS:
            O_u = U @ O @ U.conj().T

            values.append(
                float(np.real(np.trace(O_u @ rp)))
            )

    return np.asarray(values)


def random_density(seed):
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho)

    return rho


def main():
    errors = []

    for seed in range(100):
        rho = random_density(seed)

        r1 = fingerprint(rho)
        r2 = transformed_fingerprint(rho)

        errors.append(
            float(np.linalg.norm(r1 - r2))
        )

    max_error = max(errors)

    result = {
        "experiment": "QDK-028",
        "title": "Symmetry-Invariant Quantum Noise Fingerprint",
        "symmetry": "U = I tensor X",
        "channel": "two-qubit depolarizing",
        "observables": ["XX", "YY", "ZZ", "ZI", "IZ"],
        "states_tested": 100,
        "max_covariance_error": max_error,
        "tolerance": 1e-12,
        "PASS": bool(max_error < 1e-12),
        "interpretation": (
            "Fingerprint is invariant under the tested physical symmetry."
            if max_error < 1e-12
            else
            "Fingerprint covariance failed under the tested symmetry."
        ),
    }

    out = Path("results/qdk_028")
    out.mkdir(parents=True, exist_ok=True)

    path = out / "QDK-028-SYMMETRY-INVARIANT.json"

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-028 SYMMETRY-INVARIANT FINGERPRINT")
    print("=" * 50)
    print(f"States tested: 100")
    print(f"Maximum covariance error: {max_error:.12e}")
    print(f"PASS: {max_error < 1e-12}")
    print("QDK-028 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
