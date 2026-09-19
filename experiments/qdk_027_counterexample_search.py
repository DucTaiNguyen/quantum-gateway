import json
from pathlib import Path

import numpy as np


OUT_DIR = Path("results/qdk_027")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def fingerprint_distance(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(np.linalg.norm(a - b))


def random_search(seed=20260907, trials=10000):
    rng = np.random.default_rng(seed)

    best = {
        "distance": float("inf"),
        "a": None,
        "b": None,
    }

    for _ in range(trials):
        a = rng.normal(size=12)
        b = rng.normal(size=12)

        d = fingerprint_distance(a, b)

        if d < best["distance"]:
            best = {
                "distance": d,
                "a": a.tolist(),
                "b": b.tolist(),
            }

    return best


def main():
    print("QDK-027 COUNTEREXAMPLE SEARCH")
    print("=" * 50)

    trials = 10000
    best = random_search(trials=trials)

    result = {
        "experiment": "QDK-027",
        "title": "Counterexample Search",
        "seed": 20260907,
        "trials": trials,
        "search_type": "random_fingerprint_collision_search",
        "best_distance": best["distance"],
        "counterexample_found": best["distance"] < 1e-6,
        "interpretation": (
            "A candidate fingerprint collision was found."
            if best["distance"] < 1e-6
            else
            "No exact fingerprint collision was found in this random search."
        ),
    }

    output = OUT_DIR / "QDK-027-COUNTEREXAMPLE-SEARCH.json"

    with output.open("w") as f:
        json.dump(result, f, indent=2)

    print("Trials:", trials)
    print("Best fingerprint distance:", best["distance"])
    print("Counterexample found:", result["counterexample_found"])
    print("\nQDK-027 COMPLETE")
    print("Saved:", output)


if __name__ == "__main__":
    main()
