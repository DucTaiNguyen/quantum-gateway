import json
from pathlib import Path
import numpy as np

SEED = 20260907
PAIRS = 30

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
    g = np.sqrt(max(0.0, 1.0 - p))

    D = np.diag(
        [1.0, g, g, 1.0]
    )

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

    A0 = np.kron(K0, I)
    A1 = np.kron(K1, I)

    return (
        A0 @ rho @ A0.conj().T
        + A1 @ rho @ A1.conj().T
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


def fit_pair(channel, rho_a, rho_b):
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

    x = np.log(P_GRID[mask])
    y = np.log(distances[mask])

    alpha, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = alpha * x + intercept

    ss_res = np.sum(
        (y - predicted) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return float(alpha), float(r2)


def main():
    results = {
        name: []
        for name in CHANNELS
    }

    for _ in range(PAIRS):
        rho_a = random_state()
        rho_b = random_state()

        for name, channel in CHANNELS.items():
            alpha, r2 = fit_pair(
                channel,
                rho_a,
                rho_b,
            )

            results[name].append(
                {
                    "alpha": alpha,
                    "R2": r2,
                }
            )

    summary = {}

    for name, values in results.items():
        alphas = np.array(
            [v["alpha"] for v in values]
        )

        r2s = np.array(
            [v["R2"] for v in values]
        )

        summary[name] = {
            "mean_alpha": float(np.mean(alphas)),
            "std_alpha": float(np.std(alphas)),
            "min_alpha": float(np.min(alphas)),
            "max_alpha": float(np.max(alphas)),
            "mean_R2": float(np.mean(r2s)),
            "min_R2": float(np.min(r2s)),
            "max_R2": float(np.max(r2s)),
        }

    result = {
        "experiment": "QDK-032",
        "title": "Ensemble Scaling Stability",
        "seed": SEED,
        "pairs": PAIRS,
        "noise_points": len(P_GRID),
        "summary": summary,
        "interpretation": (
            "This experiment tests whether the empirical "
            "scaling exponent is stable across independent "
            "same-spectrum state pairs. Stability across "
            "pairs is required before stronger universality "
            "claims can be considered."
        ),
    }

    out = Path("results/qdk_032")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-032-ENSEMBLE-SCALING.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-032 ENSEMBLE SCALING STABILITY")
    print("=" * 50)

    for name, s in summary.items():
        print(
            f"{name}: "
            f"mean_alpha={s['mean_alpha']:.8f}, "
            f"std_alpha={s['std_alpha']:.8f}, "
            f"min={s['min_alpha']:.8f}, "
            f"max={s['max_alpha']:.8f}, "
            f"mean_R2={s['mean_R2']:.8f}"
        )

    print("QDK-032 COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
