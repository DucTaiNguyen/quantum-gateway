import json
import math
import random
from pathlib import Path
from itertools import combinations

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "qdk_009"
    / "QDK-009-STATISTICAL-VALIDATION.json"
)

OUT = ROOT / "results" / "qdk_011"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260907
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 5000


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    with open(INPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_structure_names(data):
    return [
        "STRUCTURED",
        "WEAK",
        "ASYMMETRIC",
    ]


def get_reference_distances(data):
    return data["between_structure_distance"]


def get_within_variation(data):
    return data["within_structure_variation"]


def descriptive_statistics(data):
    distances = get_reference_distances(data)
    within = get_within_variation(data)

    values = np.array(list(distances.values()), dtype=float)

    minimum_between = float(np.min(values))

    within_means = np.array(
        [
            float(within[name]["mean"])
            for name in extract_structure_names(data)
        ],
        dtype=float,
    )

    maximum_within = float(np.max(within_means))

    ratio = minimum_between / maximum_within

    return {
        "pairwise_distances": {
            k: float(v)
            for k, v in distances.items()
        },
        "minimum_between_distance": minimum_between,
        "within_structure_means": {
            name: float(within[name]["mean"])
            for name in extract_structure_names(data)
        },
        "maximum_within_mean_variation": maximum_within,
        "separation_ratio": float(ratio),
    }


def bootstrap_ratio(data, rng):
    distances = np.array(
        list(get_reference_distances(data).values()),
        dtype=float,
    )

    within = get_within_variation(data)

    within_values = np.array(
        [
            float(within[name]["mean"])
            for name in extract_structure_names(data)
        ],
        dtype=float,
    )

    ratios = []

    for _ in range(N_BOOTSTRAP):
        d_sample = rng.choice(
            distances,
            size=len(distances),
            replace=True,
        )

        w_sample = rng.choice(
            within_values,
            size=len(within_values),
            replace=True,
        )

        denominator = np.max(w_sample)

        if denominator > 0:
            ratios.append(
                np.min(d_sample) / denominator
            )

    ratios = np.asarray(ratios, dtype=float)

    return {
        "n_bootstrap": int(len(ratios)),
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "lower_95": float(np.percentile(ratios, 2.5)),
        "upper_95": float(np.percentile(ratios, 97.5)),
    }


def permutation_test(data, rng):
    """
    Permutation test on the observed pairwise-distance structure.

    Test statistic:
        T = minimum between-structure distance

    Null construction:
        random permutation of structure labels applied to
        a pooled collection of distance observations.

    This is a reference permutation diagnostic, not a substitute
    for raw replicate-level experimental data.
    """

    distances = get_reference_distances(data)

    observed = float(
        np.mean(list(distances.values()))
    )

    labels = [
        "STRUCTURED",
        "WEAK",
        "ASYMMETRIC",
    ]

    # Construct pairwise observations.
    observed_pairs = [
        float(distances["STRUCTURED_vs_WEAK"]),
        float(distances["STRUCTURED_vs_ASYMMETRIC"]),
        float(distances["WEAK_vs_ASYMMETRIC"]),
    ]

    permuted_statistics = []

    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(observed_pairs)

        # Mean is invariant to permutation, so use
        # max-min contrast as a permutation diagnostic.
        statistic = float(
            np.max(permuted) - np.min(permuted)
        )

        permuted_statistics.append(statistic)

    observed_contrast = (
        max(observed_pairs) - min(observed_pairs)
    )

    null = np.asarray(permuted_statistics)

    p_value = (
        np.sum(null >= observed_contrast) + 1
    ) / (len(null) + 1)

    return {
        "test": "pairwise-distance permutation diagnostic",
        "observed_mean_distance": observed,
        "observed_contrast": float(observed_contrast),
        "n_permutations": N_PERMUTATIONS,
        "p_value": float(p_value),
        "warning": (
            "This permutation diagnostic is limited because "
            "QDK-009 summary data contain only three pairwise "
            "distances. A valid replicate-level permutation test "
            "requires the individual trajectory observations."
        ),
    }


def centroid_classification(data):
    """
    Reference nearest-centroid classifier using the three
    structure-level distance signatures.

    With only summary pairwise distances available, this is a
    structural diagnostic rather than an independent prediction test.
    """

    distances = get_reference_distances(data)

    matrix = np.array(
        [
            [
                0.0,
                distances["STRUCTURED_vs_WEAK"],
                distances["STRUCTURED_vs_ASYMMETRIC"],
            ],
            [
                distances["STRUCTURED_vs_WEAK"],
                0.0,
                distances["WEAK_vs_ASYMMETRIC"],
            ],
            [
                distances["STRUCTURED_vs_ASYMMETRIC"],
                distances["WEAK_vs_ASYMMETRIC"],
                0.0,
            ],
        ]
    )

    labels = extract_structure_names(data)

    # Leave-one-structure-out nearest-neighbor diagnostic.
    predictions = []
    correct = 0

    for i, true_label in enumerate(labels):
        candidates = [
            (matrix[i, j], labels[j])
            for j in range(len(labels))
            if j != i
        ]

        prediction = min(
            candidates,
            key=lambda x: x[0]
        )[1]

        predictions.append(
            {
                "true": true_label,
                "predicted_reference_neighbor": prediction,
                "distance": float(
                    min(x[0] for x in candidates)
                ),
            }
        )

    # This is NOT conventional classification accuracy because
    # no independent sample-level test set exists.
    return {
        "method": "nearest-reference diagnostic",
        "predictions": predictions,
        "warning": (
            "Not an independent classifier accuracy estimate. "
            "QDK-009 summary contains structure-level distances "
            "rather than independent trajectory samples."
        ),
    }


def save_json(result):
    path = OUT / "QDK-011-STATISTICAL-SIGNIFICANCE.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved: {path}")


def main():
    print("=" * 60)
    print("QDK-011 — STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    data = load_data()

    descriptive = descriptive_statistics(data)

    print("\nObserved separation:")
    print(
        f"Minimum between distance: "
        f"{descriptive['minimum_between_distance']:.12f}"
    )

    print(
        f"Maximum within variation: "
        f"{descriptive['maximum_within_mean_variation']:.12f}"
    )

    print(
        f"Separation ratio: "
        f"{descriptive['separation_ratio']:.12f}"
    )

    bootstrap = bootstrap_ratio(data, rng)

    print("\nBootstrap separation ratio:")
    print(
        f"Mean: {bootstrap['mean']:.6f}"
    )
    print(
        f"95% interval: "
        f"[{bootstrap['lower_95']:.6f}, "
        f"{bootstrap['upper_95']:.6f}]"
    )

    permutation = permutation_test(data, rng)

    print("\nPermutation diagnostic:")
    print(
        f"Observed contrast: "
        f"{permutation['observed_contrast']:.6f}"
    )
    print(
        f"p-value: {permutation['p_value']:.6f}"
    )

    classification = centroid_classification(data)

    result = {
        "experiment": "QDK-011",
        "source": "QDK-009",
        "seed": SEED,
        "descriptive_statistics": descriptive,
        "bootstrap": bootstrap,
        "permutation_test": permutation,
        "classification_diagnostic": classification,
        "scientific_status": (
            "computational statistical diagnostic; "
            "not physical hardware validation"
        ),
    }

    save_json(result)

    print("\n" + "=" * 60)
    print("QDK-011 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
