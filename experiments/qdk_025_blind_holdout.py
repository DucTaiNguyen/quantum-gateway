import json
from pathlib import Path

import numpy as np


SEED = 20260907
N_REPLICATES = 200
TRAIN_FRACTION = 0.5
TOL = 1e-12

P_VALUES = np.linspace(0.0, 1.0, 21)

rng = np.random.default_rng(SEED)

I2 = np.eye(2, dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    return np.kron(a, b)


STRUCTURES = {
    "STRUCTURED": np.array(
        [[1.0, 0.8],
         [0.8, 1.0]],
        dtype=float,
    ),
    "WEAK": np.array(
        [[1.0, 0.2],
         [0.2, 1.0]],
        dtype=float,
    ),
    "ASYMMETRIC": np.array(
        [[1.0, 0.2],
         [0.7, 1.0]],
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
    "ALL": ["XX", "YY", "ZZ", "ZI", "IZ"],
    "CORRELATION": ["XX", "YY", "ZZ"],
    "LOCAL": ["ZI", "IZ"],
    "MIXED": ["XX", "ZZ", "IZ"],
}


SHOT_LEVELS = [
    1024,
    4096,
    16384,
]


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


def fingerprint(
    rho,
    channel,
    observable_names,
):

    rows = []

    for p in P_VALUES:

        noisy = channel(
            rho,
            p,
        )

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


def finite_shot_sample(
    F,
    shots,
):

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


def make_replicates(
    F,
    shots,
):

    return np.asarray(
        [
            normalize_columns(
                finite_shot_sample(
                    F,
                    shots,
                )
            )
            for _ in range(
                N_REPLICATES
            )
        ]
    )


def flatten_fingerprint(F):

    return F.reshape(-1)


def train_centroids(
    training_data,
):

    return {
        name: np.mean(
            reps,
            axis=0,
        )
        for name, reps
        in training_data.items()
    }


def predict(
    sample,
    centroids,
):

    distances = {
        name: float(
            np.linalg.norm(
                sample - centroid
            )
        )
        for name, centroid
        in centroids.items()
    }

    return min(
        distances,
        key=distances.get,
    )


def confusion_matrix(
    labels,
    predictions,
):

    names = list(
        STRUCTURES.keys()
    )

    matrix = {
        true_name: {
            predicted_name: 0
            for predicted_name in names
        }
        for true_name in names
    }

    for true_name, predicted_name in zip(
        labels,
        predictions,
    ):

        matrix[
            true_name
        ][
            predicted_name
        ] += 1

    return matrix


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

    all_data = {}

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

        F = normalize_columns(F)

        reps = make_replicates(
            F,
            shots,
        )

        all_data[
            structure_name
        ] = reps

    train = {}
    holdout = {}

    n_train = int(
        N_REPLICATES
        * TRAIN_FRACTION
    )

    for name, reps in all_data.items():

        indices = rng.permutation(
            N_REPLICATES
        )

        train[name] = reps[
            indices[:n_train]
        ]

        holdout[name] = reps[
            indices[n_train:]
        ]

    centroids = train_centroids(
        train
    )

    labels = []
    predictions = []

    for true_name, reps in holdout.items():

        for sample in reps:

            prediction = predict(
                sample,
                centroids,
            )

            labels.append(
                true_name
            )

            predictions.append(
                prediction
            )

    correct = sum(
        a == b
        for a, b in zip(
            labels,
            predictions,
        )
    )

    accuracy = (
        correct
        / len(labels)
    )

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    per_class = {}

    for name in STRUCTURES:

        total = sum(
            matrix[name].values()
        )

        correct_class = matrix[
            name
        ][name]

        per_class[name] = (
            correct_class / total
            if total > 0
            else 0.0
        )

    return {
        "accuracy": float(
            accuracy
        ),
        "chance_accuracy": 1.0 / 3.0,
        "correct": int(
            correct
        ),
        "total": int(
            len(labels)
        ),
        "confusion_matrix": matrix,
        "per_class_accuracy": per_class,
    }


def main():

    results = {
        "experiment": "QDK-025",
        "title": (
            "Blind Holdout Validation"
        ),
        "seed": SEED,
        "replicates": N_REPLICATES,
        "train_fraction": TRAIN_FRACTION,
        "chance_accuracy": 1.0 / 3.0,
        "conditions": {},
    }

    accuracies = []

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
                    "| OBS:",
                    observable_set_name,
                    "| SHOTS:",
                    shots,
                )
                print("=" * 70)

                result = run_condition(
                    channel_name,
                    observable_set_name,
                    shots,
                )

                results[
                    "conditions"
                ][key] = result

                accuracies.append(
                    result["accuracy"]
                )

                print(
                    "accuracy:",
                    f"{result['accuracy']:.6f}",
                )

                print(
                    "chance:",
                    f"{1/3:.6f}",
                )

    results["summary"] = {
        "number_of_conditions": len(
            accuracies
        ),
        "minimum_accuracy": float(
            np.min(accuracies)
        ),
        "median_accuracy": float(
            np.median(accuracies)
        ),
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "maximum_accuracy": float(
            np.max(accuracies)
        ),
        "all_conditions_above_chance": bool(
            np.all(
                np.asarray(accuracies)
                > 1.0 / 3.0
            )
        ),
    }

    output = Path(
        "results/qdk_025/"
        "QDK-025-BLIND-HOLDOUT.json"
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
    print("QDK-025 COMPLETE")
    print("=" * 70)

    print(
        "conditions:",
        len(accuracies),
    )

    print(
        "minimum accuracy:",
        f"{np.min(accuracies):.6f}",
    )

    print(
        "median accuracy:",
        f"{np.median(accuracies):.6f}",
    )

    print(
        "mean accuracy:",
        f"{np.mean(accuracies):.6f}",
    )

    print(
        "maximum accuracy:",
        f"{np.max(accuracies):.6f}",
    )

    print(
        "ALL CONDITIONS ABOVE CHANCE:",
        bool(
            np.all(
                np.asarray(accuracies)
                > 1.0 / 3.0
            )
        ),
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
