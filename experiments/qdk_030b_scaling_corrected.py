import json
from pathlib import Path
import numpy as np

SEED = 20260907
rng = np.random.default_rng(SEED)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1],], dtype=complex)

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


def depolarize(rho, p):
    return (
        (1.0 - p) * rho
        + p * np.eye(4, dtype=complex) / 4.0
    )


def response(rho, p):
    rp = depolarize(rho, p)

    return np.asarray([
        float(np.real(np.trace(O @ rp)))
        for O in OBS
    ])


def main():
    rho_a = random_state()
    rho_b = random_state()

    r_a0 = response(rho_a, 0.0)
    r_b0 = response(rho_b, 0.0)

    distances = []

    for p in P_GRID:
        da = response(rho_a, p) - r_a0
        db = response(rho_b, p) - r_b0

        distances.append(
            float(np.linalg.norm(da - db))
        )

    distances = np.asarray(distances)

    mask = distances > 1e-14

    log_p = np.log(P_GRID[mask])
    log_d = np.log(distances[mask])

    slope, intercept = np.polyfit(
        log_p,
        log_d,
        1,
    )

    predicted = (
        slope * log_p + intercept
    )

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

    result = {
        "experiment": "QDK-030B",
        "title": "Corrected Noise-Induced Scaling Test",
        "seed": SEED,
        "channel": "depolarizing",
        "noise_grid": P_GRID.tolist(),
        "noise_induced_distances": distances.tolist(),
        "power_law_exponent": float(slope),
        "log_intercept": float(intercept),
        "log_log_R2": float(r2),
        "interpretation": (
            "The fit tests scaling of the noise-induced "
            "fingerprint difference after subtracting the "
            "zero-noise baseline. This is an empirical fit "
            "and does not establish a universal law."
        ),
    }

    out = Path("results/qdk_030b")
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out
        / "QDK-030B-CORRECTED-SCALING.json"
    )

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("QDK-030B CORRECTED SCALING")
    print("=" * 50)
    print(
        "Scaling exponent alpha:",
        f"{slope:.12e}",
    )
    print(
        "Log-log R2:",
        f"{r2:.12e}",
    )
    print(
        "Minimum induced distance:",
        f"{np.min(distances):.12e}",
    )
    print(
        "Maximum induced distance:",
        f"{np.max(distances):.12e}",
    )
    print("QDK-030B COMPLETE")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
