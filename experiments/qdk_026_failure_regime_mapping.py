import json
from pathlib import Path


INPUT = Path("results/qdk_025/QDK-025-BLIND-HOLDOUT.json")
OUTPUT_DIR = Path("results/qdk_026")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with INPUT.open() as f:
        data = json.load(f)

    conditions = data["conditions"]
    chance = data["chance_accuracy"]

    failures = []
    successes = []

    for name, result in conditions.items():
        accuracy = result.get("accuracy")

        record = {
            "condition": name,
            "accuracy": accuracy,
            "chance_accuracy": chance,
            "above_chance": accuracy >= chance,
        }

        if accuracy < chance:
            failures.append(record)
        else:
            successes.append(record)

    summary = {
        "experiment": "QDK-026",
        "title": "Failure-Regime Mapping",
        "source": str(INPUT),
        "number_of_conditions": len(conditions),
        "chance_accuracy": chance,
        "number_above_chance": len(successes),
        "number_below_chance": len(failures),
        "failure_fraction": len(failures) / len(conditions),
    }

    output = {
        "summary": summary,
        "failure_conditions": failures,
        "successful_conditions": successes,
    }

    output_file = OUTPUT_DIR / "QDK-026-FAILURE-REGIME-MAPPING.json"

    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    print("QDK-026 FAILURE REGIME MAPPING")
    print("=" * 50)
    print("Conditions:", len(conditions))
    print("Chance accuracy:", chance)
    print("Above chance:", len(successes))
    print("Below chance:", len(failures))
    print("Failure fraction:", summary["failure_fraction"])

    print("\nFAILURE CONDITIONS")
    print("-" * 50)

    if failures:
        for item in failures:
            print(
                f"{item['condition']}: "
                f"accuracy={item['accuracy']:.6f}"
            )
    else:
        print("None")

    print("\nQDK-026 COMPLETE")
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
