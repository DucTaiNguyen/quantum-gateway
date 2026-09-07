import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SOURCE = Path("results/qdk_009/QDK-009-STATISTICAL-VALIDATION.json")
OUT = Path("results/qdk_011b")
OUT.mkdir(parents=True, exist_ok=True)


STRUCTURES = ["STRUCTURED", "WEAK", "ASYMMETRIC"]


def find_trajectory_objects(obj):
    found = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in STRUCTURES and isinstance(value, dict):
                if "mean" in value and isinstance(value["mean"], list):
                    found[key] = value

            nested = find_trajectory_objects(value)
            found.update(nested)

    elif isinstance(obj, list):
        for value in obj:
            nested = find_trajectory_objects(value)
            found.update(nested)

    return found


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return float("nan")

    return float(1.0 - np.dot(a, b) / (na * nb))


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    with SOURCE.open() as f:
        data = json.load(f)

    trajectories = find_trajectory_objects(data)

    missing = [s for s in STRUCTURES if s not in trajectories]
    if missing:
        raise RuntimeError(
            f"Could not find trajectory statistics for: {missing}"
        )

    means = {}
    stds = {}
    ci95 = {}

    for s in STRUCTURES:
        means[s] = np.asarray(
            trajectories[s]["mean"], dtype=float
        )

        stds[s] = np.asarray(
            trajectories[s].get("std", np.zeros_like(means[s])),
            dtype=float,
        )

        ci95[s] = np.asarray(
            trajectories[s].get(
                "ci95_half_width",
                np.zeros_like(means[s])
            ),
            dtype=float,
        )

    lengths = {len(v) for v in means.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"Trajectory lengths differ: {lengths}")

    n = len(next(iter(means.values())))
    p = np.linspace(0.0, 1.0, n)

    # Normalize each trajectory by its initial amplitude.
    normalized = {}

    for s in STRUCTURES:
        x = means[s]

        if abs(x[0]) < 1e-15:
            raise RuntimeError(
                f"Initial amplitude for {s} is too close to zero."
            )

        normalized[s] = x / abs(x[0])

    pair_names = [
        ("STRUCTURED", "WEAK"),
        ("STRUCTURED", "ASYMMETRIC"),
        ("WEAK", "ASYMMETRIC"),
    ]

    raw_distances = {}
    normalized_distances = {}
    cosine_distances = {}

    for a, b in pair_names:
        name = f"{a}_vs_{b}"

        raw_distances[name] = euclidean(
            means[a], means[b]
        )

        normalized_distances[name] = euclidean(
            normalized[a], normalized[b]
        )

        cosine_distances[name] = cosine_distance(
            normalized[a], normalized[b]
        )

    # Integrated absolute area between normalized trajectories.
    area_distances = {}

    for a, b in pair_names:
        name = f"{a}_vs_{b}"

        area_distances[name] = float(
            np.trapezoid(
                np.abs(normalized[a] - normalized[b]),
                p,
            )
        )

    # Descriptive CI-band non-overlap.
    ci_nonoverlap = {}

    for a, b in pair_names:
        lower_a = means[a] - ci95[a]
        upper_a = means[a] + ci95[a]

        lower_b = means[b] - ci95[b]
        upper_b = means[b] + ci95[b]

        overlap = np.maximum(lower_a, lower_b) <= np.minimum(
            upper_a, upper_b
        )

        ci_nonoverlap[f"{a}_vs_{b}"] = {
            "points_nonoverlapping": int(np.sum(~overlap)),
            "total_points": int(n),
            "fraction_nonoverlapping": float(
                np.mean(~overlap)
            ),
        }

    max_raw = max(raw_distances.values())
    min_raw = min(raw_distances.values())

    max_normalized = max(normalized_distances.values())
    min_normalized = min(normalized_distances.values())

    raw_to_normalized = {
        name: (
            raw_distances[name] /
            normalized_distances[name]
            if normalized_distances[name] > 0
            else float("inf")
        )
        for name in raw_distances
    }

    result = {
        "experiment": "QDK-011B",
        "title": "Scale-Normalized Noise-Trajectory Robustness",
        "source": str(SOURCE),
        "trajectory_points": n,
        "normalization": (
            "Each mean trajectory is divided by the absolute "
            "initial amplitude |C(0)|."
        ),
        "raw_euclidean_distance": raw_distances,
        "normalized_euclidean_distance": normalized_distances,
        "normalized_cosine_distance": cosine_distances,
        "normalized_area_between_curves": area_distances,
        "raw_to_normalized_distance_ratio": raw_to_normalized,
        "ci95_band_nonoverlap": ci_nonoverlap,
        "min_raw_distance": min_raw,
        "max_raw_distance": max_raw,
        "min_normalized_distance": min_normalized,
        "max_normalized_distance": max_normalized,
        "interpretation": {
            "stronger_shape_evidence": (
                "Normalized trajectory distances remain materially "
                "nonzero, indicating separation beyond simple "
                "initial-amplitude differences."
            ),
            "amplitude_dominated": (
                "Normalized trajectory distances collapse toward zero, "
                "indicating that raw separation is primarily explained "
                "by amplitude scaling."
            ),
            "warning": (
                "This is a robustness/descriptive analysis, not a "
                "formal hypothesis test or p-value."
            ),
        },
    }

    out_json = OUT / "QDK-011B-SCALE-ROBUSTNESS.json"

    with out_json.open("w") as f:
        json.dump(result, f, indent=2)

    # Raw trajectory plot.
    plt.figure(figsize=(8, 5))

    for s in STRUCTURES:
        plt.plot(p, means[s], label=s)

    plt.xlabel("Noise parameter p")
    plt.ylabel("ZZ correlation response")
    plt.title("QDK-011B Raw Noise Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT / "QDK-011B-raw-trajectories.png",
        dpi=180,
    )
    plt.close()

    # Normalized trajectory plot.
    plt.figure(figsize=(8, 5))

    for s in STRUCTURES:
        plt.plot(p, normalized[s], label=s)

    plt.xlabel("Noise parameter p")
    plt.ylabel("Normalized trajectory")
    plt.title("QDK-011B Scale-Normalized Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT / "QDK-011B-normalized-trajectories.png",
        dpi=180,
    )
    plt.close()

    # Difference plot.
    plt.figure(figsize=(8, 5))

    for a, b in pair_names:
        plt.plot(
            p,
            np.abs(normalized[a] - normalized[b]),
            label=f"{a} vs {b}",
        )

    plt.xlabel("Noise parameter p")
    plt.ylabel("Absolute normalized difference")
    plt.title("QDK-011B Shape Separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT / "QDK-011B-shape-separation.png",
        dpi=180,
    )
    plt.close()

    print("QDK-011B COMPLETE")
    print()
    print("Raw Euclidean distances:")
    for k, v in raw_distances.items():
        print(f"  {k}: {v:.12f}")

    print()
    print("Normalized Euclidean distances:")
    for k, v in normalized_distances.items():
        print(f"  {k}: {v:.12f}")

    print()
    print("Normalized cosine distances:")
    for k, v in cosine_distances.items():
        print(f"  {k}: {v:.12f}")

    print()
    print("Normalized area distances:")
    for k, v in area_distances.items():
        print(f"  {k}: {v:.12f}")

    print()
    print("95% CI descriptive non-overlap:")
    for k, v in ci_nonoverlap.items():
        print(
            f"  {k}: "
            f"{v['points_nonoverlapping']}/"
            f"{v['total_points']} "
            f"({v['fraction_nonoverlapping']:.3f})"
        )

    print()
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
