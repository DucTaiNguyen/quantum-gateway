import json
from pathlib import Path
import numpy as np

SEED = 20260907
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

SPECTRUM = np.array([0.50, 0.25, 0.15, 0.10])

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

    return Q @ np.diag(np.conjugate(d))


def random_state():
    U = random_unitary(4)
    return U @ np.diag(SPECTRUM) @ U.conj().T


def depolarizing(rho, p):
    return (
        (1.0 - p) * rho
        + p * np.eye(4, dtype=complex) / 4.0
    )


def phase_damping(rho, p):
    D = np.diag([
        1.0,
        np.sqrt(max(0.0, 1.0 - p)),
        np.sqrt(max(0.0, 1.0 - p)),
        1.0,
    ])

    return D @ rho @ D


def amplitude_damping(rho, p):
    g = np.sqrt(max(0.0, 1.0 - p))

    K0 = np.array(
        [[1.0, 0.0], [0.0, g]],
        dtype=complex,
    )

    K1 = np.array(
        [[0.0, np.sqrt(p)], [0.0, 0.0]],
        dtype=complex,
    )

    K0_2 = np.kron(K0, I)
    K1_2 = np.kron(K1, I)

    return (
        K0_2 @ rho @ K0_2.conj().T
        + K1_2 @ rho @ K1_2.conj().T
    )


CHANNELS = {
    "depolarizing": depolarizing,
    "phase_damping": phase_damping,
    "amplitude_damping": amplitude_damping,
}


def response(channel, rho, p):
    rp = channel(rho, p)

    return np.asarray([
        float(np.real(np.trace(O @ rp)))
        for O in OBS
    ])


def fit_scaling(channel, rho_a, rho_b):
    a0 = response(channel, rho_a, 0.0)
    b0 = response(channel, rho_b, 0.0)

    distances = []

    for p in P_GRID:
        da = response(channel, rho_a, p) - a0
        db = response(channel, rho_b, p) - b0

        distances.append(
            float(np.linalg.norm(da - db))
        )

    distances = np.asarray(distances)

    mask = distances > 1e-14

    log_p = np.log(P_GRID[mask])
    log_d = np.log(distances[mask])

    alpha, intercept = np.polyfit(
        log_p,
        log_d,
        1,
    )

    predicted = alpha * log_p + intercept

    ss_res = np.sum(
        (log_d - predicted) ** 2
    )

    ss_tot = np.sum(
        (log_d - np.mean(log_d)) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return {
        "alpha": float(alpha),
        "R2": float(r2),
        "minimum_distance": float(np.min(distances)),
        "maximum_distance": float(np.max(distances)),
        "distances": distances.tolist(),
    }


def main():
    rho_a = random_state()
    rho_b = random_state()

    results = {}

    for name, channel in CHANNELS.items():
        results[name] = fit_scaling(
            channel,
            rho_a,
            rho_b,
        )

    result = {
        "experiment": "QDK-031",
        "title": "Open-System Scaling Universality",
        "seed": SEED,
        "channels": list(CHANNELS.keys()),
        "noise_points": len(P_GRID),
        "results": results,
        "interpretation": (
            "Scaling exponents are compared across multiple "
            "open-system noise channels. Agreement across "
            "channels is evidence for model robustness, not "
            "proof of a universal physical law."
        ),
    }

    out = Path("results/qdk_031")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-031-OPEN-SYSTEM-UNIVERSALITY.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-031 OPEN-SYSTEM UNIVERSALITY")
    print("=" * 50)

    for name, r in results.items():
        print(
            f"{name}: "
            f"alpha={r['alpha']:.12e}, "
            f"R2={r['R2']:.12e}"
        )

    print("QDK-031 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
