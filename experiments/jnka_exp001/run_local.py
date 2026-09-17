import csv
import json
import os
import sys

import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    ),
)

from src.jnka.fingerprint import compute_fingerprint
from src.jnka.policy import decide


N_QUBITS = 4
SHOTS = 4096
THETA_VALUES = np.linspace(0.0, 2.0 * np.pi, 33)
SEEDS = range(10)

RESULT_DIR = "results"
CSV_FILE = os.path.join(
    RESULT_DIR,
    "jnka_exp001.csv",
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    "jnka_exp001_summary.json",
)


def create_circuit(theta):
    qc = QuantumCircuit(
        N_QUBITS,
        N_QUBITS,
    )

    for q in range(N_QUBITS):
        qc.h(q)

    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)

    qc.ry(theta, 0)

    qc.measure(
        range(N_QUBITS),
        range(N_QUBITS),
    )

    return qc


def create_noise_model(level):
    noise = NoiseModel()

    if level == "ideal":
        return noise

    strengths = {
        "low": 0.001,
        "medium": 0.005,
        "high": 0.02,
    }

    p = strengths[level]

    one_qubit = depolarizing_error(
        p,
        1,
    )

    two_qubit = depolarizing_error(
        p * 2,
        2,
    )

    noise.add_all_qubit_quantum_error(
        one_qubit,
        ["h", "ry"],
    )

    noise.add_all_qubit_quantum_error(
        two_qubit,
        ["cx"],
    )

    readout_p = min(
        0.01 + 5.0 * p,
        0.25,
    )

    readout = ReadoutError([
        [1.0 - readout_p, readout_p],
        [readout_p, 1.0 - readout_p],
    ])

    noise.add_all_qubit_readout_error(readout)

    return noise


def run_counts(circuit, noise_model, seed):
    simulator = AerSimulator(
        noise_model=noise_model,
    )

    job = simulator.run(
        circuit,
        shots=SHOTS,
        seed_simulator=seed,
    )

    return job.result().get_counts()


def counts_to_probability(counts):
    dimension = 2 ** N_QUBITS
    p = np.zeros(dimension)

    total = sum(counts.values())

    for bitstring, count in counts.items():
        p[int(bitstring, 2)] = count / total

    return p


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    rows = []

    print("JNKA-EXP-001")
    print("=" * 60)
    print(f"qubits : {N_QUBITS}")
    print(f"shots  : {SHOTS}")
    print(f"theta  : {len(THETA_VALUES)}")
    print(f"seeds  : {len(list(SEEDS))}")
    print()

    for seed in SEEDS:
        print(f"Seed {seed}")

        for theta in THETA_VALUES:

            circuit = create_circuit(theta)

            ideal_counts = run_counts(
                circuit,
                create_noise_model("ideal"),
                seed,
            )

            ideal_p = counts_to_probability(
                ideal_counts
            )

            for condition in [
                "low",
                "medium",
                "high",
            ]:

                noisy_counts = run_counts(
                    circuit,
                    create_noise_model(condition),
                    seed,
                )

                noisy_p = counts_to_probability(
                    noisy_counts
                )

                fingerprint = compute_fingerprint(
                    ideal_p,
                    noisy_p,
                )

                vector = fingerprint.vector()

                # EXP-001 intentionally uses a
                # neutral utility placeholder.
                utility = 0.0

                action = decide(
                    utility=utility,
                    tv=fingerprint.tv,
                )

                rows.append({
                    "experiment": "JNKA-EXP-001",
                    "seed": seed,
                    "theta": float(theta),
                    "condition": condition,
                    "shots": SHOTS,
                    "kl": fingerprint.kl,
                    "js": fingerprint.js,
                    "tv": fingerprint.tv,
                    "entropy": fingerprint.entropy,
                    "l1": fingerprint.l1,
                    "utility": utility,
                    "action": action,
                })

    fieldnames = list(rows[0].keys())

    with open(
        CSV_FILE,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "JNKA-EXP-001",
        "qubits": N_QUBITS,
        "shots": SHOTS,
        "theta_points": len(THETA_VALUES),
        "seeds": len(list(SEEDS)),
        "conditions": [
            "ideal",
            "low",
            "medium",
            "high",
        ],
        "observations": len(rows),
        "metrics": [
            "KL",
            "JS",
            "TV",
            "entropy",
            "L1",
        ],
        "status": "LOCAL_COMPLETE",
    }

    with open(
        SUMMARY_FILE,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("COMPLETE")
    print(f"CSV: {CSV_FILE}")
    print(f"JSON: {SUMMARY_FILE}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
