import json
from pathlib import Path

import numpy as np


SEED = 20260907
N_REPLICATES = 100
N_BOOTSTRAP = 2000
TOL = 1e-12

P_VALUES = np.linspace(0.0, 1.0, 21)
SHOT_LEVELS = [1024, 4096, 16384]

rng = np.random.default_rng(SEED)

I2 = np.eye(2, dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    return np.kron(a, b)


STRUCTURES = {
    "STRUCTURED": np.array(
        [[1.0, 0.8], [0.8, 1.0]],
        dtype=float,
    ),
    "WEAK": np.array(
        [[1.0, 0.2], [0.2, 1.0]],
        dtype=float,
    ),
    "ASYMMETRIC": np.array(
        [[1.0, 0.2], [0.7, 1.0]],
        dtype=float,
    ),
}


OBSERVABLES = {
    "XX": kron(X, X),
    "YY": kron(Y, Y),
    "ZZ": kron(Z, Z),
    "ZI": kron(Z, I2),
    "IZ": kron(I2, Z),
}


OBSERVABLE_SETS = {
    "ALL": [
        "XX",
        "YY",
        "ZZ",
        "ZI",
        "IZ",
    ],
    "CORRELATION": [
        "XX",
        "YY",
        "ZZ",
    ],
    "LOCAL": [
        "ZI",
        "IZ",
    ],
    "MIXED": [
        "XX",
        "ZZ",
        "IZ",
    ],
}


def tensor_to_state(tensor):
    v = np.asarray(
        tensor,
        dtype=float,
    ).reshape(-1)

    v = np.abs(v)

    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Zero tensor.")

    return v / norm


def density_matrix(psi):
    return np.outer(
        psi,
        psi.conj(),
    )


def amplitude_damping(rho, p):
    k0 = np.array(
        [
            [1.0, 0.0],
            [0.0, np.sqrt(1.0 - p)],
        ],
        dtype=complex,
    )

    k1 = np.array(
        [
            [0.0, np.sqrt(p)],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    K0 = kron(k0, I2)
    K1 = kron(k1, I2)

    return (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )


def phase_damping(rho, p):
    k0 = np.sqrt(1.0 - p) * I2
    k1 = np.sqrt(p) * Z

    K0 = kron(k0, I2)
    K1 = kron(k1, I2)

    return (
        K0 @ rho @ K0.conj().T
        + K1 @ rho @ K1.conj().T
    )


CHANNELS = {
    "phase_damping": phase_damping,
    "amplitude_damping": amplitude_damping,
}


def fingerprint(rho, channel, observable_names):
    rows = []

    for p in P_VALUES:

        noisy = channel(rho, p)

        row = []

        for name in observable_names:

            O = OBSERVABLES[name]

            value = np.trace(
                O @ noisy
            )

            row.append(
                float(
                    np.real(value)
                )
            )

        rows.append(row)

    return np.asarray(
        rows,
        dtype=float,
    )


def normalize_columns(F):
    scale = np.max(
        np.abs(F),
        axis=0,
    )

    scale[scale < TOL] = 1.0

    return F / scale


def finite_shot_sample(F, shots):

    clipped = np.clip(
        F,
        -1.0,
        1.0,
    )

    variance = np.maximum(
        0.0,
        (
            1.0
            - clipped**2
        )
        / shots,
    )

    noise = rng.normal(
        0.0,
        np.sqrt(variance),
        size=F.shape,
    )

    return F + noise


def generate_replicates(F, shots):

    return np.asarray(
        [
            normalize_columns(
                finite_shot_sample(
                    F,
                    shots,
                )
            )
            for _ in range(N_REPLICATES)
        ]
    )


def distance(A, B):

    return float(
        np.linalg.norm(
            A - B
        )
    )


def within_distances(R):

    values = []

    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            values.append(
                distance(
                    R[i],
                    R[j],
                )
            )

    return np.asarray(values)


def between_distances(A, B):

    return np.asarray(
        [
            distance(a, b)
            for a in A
            for b in B
        ]
    )


def bootstrap_mean(values):

    values = np.asarray(values)

    means = np.empty(
        N_BOOTSTRAP
    )

    for i in range(
        N_BOOTSTRAP
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        means[i] = np.mean(
            sample
        )

    return {
        "mean": float(
            np.mean(values)
        ),
        "ci95_low": float(
            np.percentile(
                means,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                means,
                97.5,
            )
        ),
    }


def run_condition(
    channel_name,
    observable_set_name,
    shots,
):

    channel = CHANNELS[
        channel_name
    ]

    observable_names = (
        OBSERVABLE_SETS[
            observable_set_name
        ]
    )

    replicates = {}

    for structure_name, tensor in STRUCTURES.items():

        psi = tensor_to_state(
            tensor
        )

        rho = density_matrix(
            psi
        )

        F = fingerprint(
            rho,
            channel,
            observable_names,
        )

        F = normalize_columns(
            F
        )

        replicates[
            structure_name
        ] = generate_replicates(
            F,
            shots,
        )

    names = list(
        STRUCTURES.keys()
    )

    within = []

    for name in names:
        within.extend(
            within_distances(
                replicates[name]
            )
        )

    between = []

    pair_results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a = names[i]
            b = names[j]

            d = between_distances(
                replicates[a],
                replicates[b],
            )

            key = f"{a}_vs_{b}"

            pair_results[key] = {
                "mean": float(
                    np.mean(d)
                ),
                "ci95": bootstrap_mean(
                    d
                ),
            }

            between.extend(d)

    within = np.asarray(within)
    between = np.asarray(between)

    within_mean = float(
        np.mean(within)
    )

    between_mean = float(
        np.mean(between)
    )

    ratio = (
        between_mean
        / max(
            within_mean,
            TOL,
        )
    )

    return {
        "within_mean": within_mean,
        "within_ci95": bootstrap_mean(
            within
        ),
        "between_mean": between_mean,
        "between_ci95": bootstrap_mean(
            between
        ),
        "between_within_ratio": float(
            ratio
        ),
        "pairwise": pair_results,
    }


def main():

    results = {
        "experiment": "QDK-024",
        "title": (
            "Cross-Channel / "
            "Cross-Observable "
            "Generalization"
        ),
        "seed": SEED,
        "replicates": N_REPLICATES,
        "bootstrap_samples": N_BOOTSTRAP,
        "shots": SHOT_LEVELS,
        "observable_sets": OBSERVABLE_SETS,
        "conditions": {},
    }

    ratios = []

    for channel_name in CHANNELS:

        for observable_set_name in OBSERVABLE_SETS:

            for shots in SHOT_LEVELS:

                key = (
                    f"{channel_name}"
                    f"__{observable_set_name}"
                    f"__shots_{shots}"
                )

                print("=" * 70)
                print(
                    "CHANNEL:",
                    channel_name,
                    "| OBSERVABLES:",
                    observable_set_name,
                    "| SHOTS:",
                    shots,
                )
                print("=" * 70)

                condition = run_condition(
                    channel_name,
                    observable_set_name,
                    shots,
                )

                results[
                    "conditions"
                ][key] = condition

                ratio = condition[
                    "between_within_ratio"
                ]

                ratios.append(ratio)

                print(
                    "between/within ratio:",
                    f"{ratio:.6f}",
                )

    min_ratio = float(
        np.min(ratios)
    )

    median_ratio = float(
        np.median(ratios)
    )

    mean_ratio = float(
        np.mean(ratios)
    )

    robust = min_ratio > 1.0

    results["summary"] = {
        "number_of_conditions": len(
            ratios
        ),
        "minimum_ratio": min_ratio,
        "median_ratio": median_ratio,
        "mean_ratio": mean_ratio,
        "all_conditions_ratio_gt_1": bool(
            robust
        ),
    }

    output = Path(
        "results/qdk_024/"
        "QDK-024-CROSS-CHANNEL-GENERALIZATION.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print("=" * 70)
    print("QDK-024 COMPLETE")
    print("=" * 70)

    print(
        "conditions:",
        len(ratios),
    )

    print(
        "minimum ratio:",
        f"{min_ratio:.6f}",
    )

    print(
        "median ratio:",
        f"{median_ratio:.6f}",
    )

    print(
        "mean ratio:",
        f"{mean_ratio:.6f}",
    )

    print(
        "ALL CONDITIONS > 1:",
        robust,
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
