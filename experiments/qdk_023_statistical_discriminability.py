import json
from pathlib import Path

import numpy as np


SEED = 20260907
N_REPLICATES = 200
N_SHOTS = 4096
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000
TOL = 1e-12

P_VALUES = np.linspace(0.0, 1.0, 21)

rng = np.random.default_rng(SEED)

I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1], [1, 0]],
    dtype=complex,
)

Y = np.array(
    [[0, -1j], [1j, 0]],
    dtype=complex,
)

Z = np.array(
    [[1, 0], [0, -1]],
    dtype=complex,
)


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


# ------------------------------------------------------------
# Physical local symmetry identified in QDK-020
# permutation (1,0,3,2) = I tensor X
# ------------------------------------------------------------

U_STAR = kron(I2, X)


def tensor_to_state(tensor):
    v = np.asarray(
        tensor,
        dtype=float,
    ).reshape(-1)

    v = np.abs(v)

    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError(
            "Cannot encode zero tensor."
        )

    return v / norm


def density_matrix(psi):
    return np.outer(
        psi,
        psi.conj(),
    )


# ------------------------------------------------------------
# Reference channels
# ------------------------------------------------------------

def amplitude_damping(rho, p):
    p = float(p)

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
    p = float(p)

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


# ------------------------------------------------------------
# Deterministic fingerprint
# ------------------------------------------------------------

def fingerprint(rho, channel):
    rows = []

    for p in P_VALUES:
        noisy = channel(rho, p)

        row = []

        for O in OBSERVABLES.values():
            value = np.trace(O @ noisy)
            row.append(float(np.real(value)))

        rows.append(row)

    return np.asarray(
        rows,
        dtype=float,
    )


# ------------------------------------------------------------
# Finite-shot measurement model
#
# For Pauli expectation m:
#
#     Var(m_hat) ~= (1 - m^2) / N
#
# This is a statistical measurement model, not IBM
# device noise.
# ------------------------------------------------------------

def finite_shot_sample(F):
    variance = np.maximum(
        0.0,
        (1.0 - np.clip(F, -1.0, 1.0) ** 2)
        / N_SHOTS,
    )

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(variance),
        size=F.shape,
    )

    return F + noise


# ------------------------------------------------------------
# Column normalization
# ------------------------------------------------------------

def normalize_columns(F):
    scale = np.max(
        np.abs(F),
        axis=0,
    )

    scale[scale < TOL] = 1.0

    return F / scale


# ------------------------------------------------------------
# Replicate generation
# ------------------------------------------------------------

def generate_replicates(F):
    reps = []

    for _ in range(N_REPLICATES):
        measured = finite_shot_sample(F)
        measured = normalize_columns(measured)
        reps.append(measured)

    return np.asarray(reps)


# ------------------------------------------------------------
# Distances
# ------------------------------------------------------------

def distance(A, B):
    return float(
        np.linalg.norm(A - B)
    )


def pairwise_within(replicates):
    distances = []

    n = len(replicates)

    for i in range(n):
        for j in range(i + 1, n):
            distances.append(
                distance(
                    replicates[i],
                    replicates[j],
                )
            )

    return np.asarray(distances)


def pairwise_between(A, B):
    distances = []

    for a in A:
        for b in B:
            distances.append(
                distance(a, b)
            )

    return np.asarray(distances)


# ------------------------------------------------------------
# Bootstrap confidence interval
# ------------------------------------------------------------

def bootstrap_mean_ci(values):
    values = np.asarray(values)

    if len(values) == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    boot_means = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means[i] = np.mean(sample)

    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(
            np.percentile(
                boot_means,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                boot_means,
                97.5,
            )
        ),
    }


# ------------------------------------------------------------
# Welch-style standardized effect
# ------------------------------------------------------------

def standardized_effect(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    mean_A = np.mean(A)
    mean_B = np.mean(B)

    var_A = np.var(A, ddof=1)
    var_B = np.var(B, ddof=1)

    pooled = np.sqrt(
        0.5 * (var_A + var_B)
    )

    if pooled < TOL:
        return float("inf")

    return float(
        abs(mean_A - mean_B) / pooled
    )


# ------------------------------------------------------------
# Replicate-level permutation test
#
# Null hypothesis:
# the two replicate populations are exchangeable.
#
# Statistic = absolute difference in mean distance
# to the pooled centroid.
# ------------------------------------------------------------

def centroid_distance_stat(A, B):
    pooled = np.concatenate(
        [A, B],
        axis=0,
    )

    centroid = np.mean(
        pooled,
        axis=0,
    )

    dA = np.linalg.norm(
        A - centroid,
        axis=1,
    )

    dB = np.linalg.norm(
        B - centroid,
        axis=1,
    )

    return float(
        abs(
            np.mean(dA)
            - np.mean(dB)
        )
    )


def permutation_test(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    observed = centroid_distance_stat(
        A,
        B,
    )

    pooled = np.concatenate(
        [A, B],
        axis=0,
    )

    nA = len(A)

    count = 0

    for _ in range(N_PERMUTATIONS):

        indices = rng.permutation(
            len(pooled)
        )

        group_A = pooled[
            indices[:nA]
        ]

        group_B = pooled[
            indices[nA:]
        ]

        stat = centroid_distance_stat(
            group_A,
            group_B,
        )

        if stat >= observed:
            count += 1

    p_value = (
        count + 1
    ) / (
        N_PERMUTATIONS + 1
    )

    return {
        "observed_statistic": float(
            observed
        ),
        "permutation_p_value": float(
            p_value
        ),
        "permutations": N_PERMUTATIONS,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    results = {
        "experiment": "QDK-023",
        "title": (
            "Statistical Fingerprint "
            "Discriminability"
        ),
        "seed": SEED,
        "replicates": N_REPLICATES,
        "shots": N_SHOTS,
        "bootstrap_samples": N_BOOTSTRAP,
        "permutation_samples": N_PERMUTATIONS,
        "measurement_model": (
            "finite-shot Pauli expectation "
            "Gaussian approximation"
        ),
        "channels": {},
    }

    for channel_name, channel in CHANNELS.items():

        print("=" * 70)
        print(
            "CHANNEL:",
            channel_name,
        )
        print("=" * 70)

        deterministic = {}
        replicates = {}

        # ----------------------------------------------------
        # Generate deterministic fingerprints
        # ----------------------------------------------------

        for name, tensor in STRUCTURES.items():

            psi = tensor_to_state(tensor)
            rho = density_matrix(psi)

            F = fingerprint(
                rho,
                channel,
            )

            deterministic[name] = (
                normalize_columns(F)
            )

            replicates[name] = (
                generate_replicates(
                    deterministic[name]
                )
            )

        channel_result = {
            "within_structure": {},
            "between_structure": {},
            "discriminability": {},
        }

        # ----------------------------------------------------
        # Within-structure variation
        # ----------------------------------------------------

        for name in STRUCTURES:

            within = pairwise_within(
                replicates[name]
            )

            ci = bootstrap_mean_ci(
                within
            )

            channel_result[
                "within_structure"
            ][name] = {
                "n_pairs": int(
                    len(within)
                ),
                "mean_distance": float(
                    np.mean(within)
                ),
                "std_distance": float(
                    np.std(
                        within,
                        ddof=1,
                    )
                ),
                "max_distance": float(
                    np.max(within)
                ),
                "bootstrap_mean_ci95": ci,
            }

            print(
                name,
                "within mean:",
                f"{np.mean(within):.12e}",
            )

        # ----------------------------------------------------
        # Between-structure separation
        # ----------------------------------------------------

        names = list(
            STRUCTURES.keys()
        )

        between_means = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a = names[i]
                b = names[j]

                between = pairwise_between(
                    replicates[a],
                    replicates[b],
                )

                ci = bootstrap_mean_ci(
                    between
                )

                effect = standardized_effect(
                    between,
                    pairwise_within(
                        replicates[a]
                    ),
                )

                key = f"{a}_vs_{b}"

                channel_result[
                    "between_structure"
                ][key] = {
                    "n_pairs": int(
                        len(between)
                    ),
                    "mean_distance": float(
                        np.mean(between)
                    ),
                    "std_distance": float(
                        np.std(
                            between,
                            ddof=1,
                        )
                    ),
                    "min_distance": float(
                        np.min(between)
                    ),
                    "max_distance": float(
                        np.max(between)
                    ),
                    "bootstrap_mean_ci95": ci,
                    "descriptive_effect_vs_A_within": effect,
                }

                between_means.append(
                    np.mean(between)
                )

                print(
                    key,
                    "between mean:",
                    f"{np.mean(between):.12e}",
                )

        # ----------------------------------------------------
        # Global discriminability
        # ----------------------------------------------------

        all_within = []

        for name in names:
            all_within.extend(
                pairwise_within(
                    replicates[name]
                )
            )

        all_between = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a = names[i]
                b = names[j]

                all_between.extend(
                    pairwise_between(
                        replicates[a],
                        replicates[b],
                    )
                )

        all_within = np.asarray(
            all_within
        )

        all_between = np.asarray(
            all_between
        )

        within_mean = float(
            np.mean(all_within)
        )

        between_mean = float(
            np.mean(all_between)
        )

        ratio = float(
            between_mean
            / max(within_mean, TOL)
        )

        channel_result[
            "discriminability"
        ] = {
            "mean_between_distance": (
                between_mean
            ),
            "mean_within_distance": (
                within_mean
            ),
            "between_within_ratio": ratio,
            "between_bootstrap_ci95": (
                bootstrap_mean_ci(
                    all_between
                )
            ),
            "within_bootstrap_ci95": (
                bootstrap_mean_ci(
                    all_within
                )
            ),
        }

        # ----------------------------------------------------
        # Permutation test
        #
        # STRUCTURED vs ASYMMETRIC
        # chosen because QDK-022 shows large deterministic
        # separation for this pair.
        # ----------------------------------------------------

        permutation = permutation_test(
            replicates["STRUCTURED"],
            replicates["ASYMMETRIC"],
        )

        channel_result[
            "permutation_test"
        ] = permutation

        print(
            "global between/within ratio:",
            f"{ratio:.6f}",
        )

        print(
            "STRUCTURED vs ASYMMETRIC permutation p:",
            f"{permutation['permutation_p_value']:.6f}",
        )

        results[
            "channels"
        ][channel_name] = channel_result

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {}

    for channel_name in CHANNELS:

        D = results[
            "channels"
        ][channel_name][
            "discriminability"
        ]

        summary[channel_name] = {
            "between_within_ratio": float(
                D["between_within_ratio"]
            ),
            "between_mean": float(
                D["mean_between_distance"]
            ),
            "within_mean": float(
                D["mean_within_distance"]
            ),
            "permutation_p_value": float(
                results[
                    "channels"
                ][channel_name][
                    "permutation_test"
                ][
                    "permutation_p_value"
                ]
            ),
        }

    results["summary"] = summary

    output = Path(
        "results/qdk_023/"
        "QDK-023-STATISTICAL-DISCRIMINABILITY.json"
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
    print("QDK-023 COMPLETE")
    print("=" * 70)

    for channel_name, data in summary.items():
        print(
            channel_name,
            "between/within ratio:",
            f"{data['between_within_ratio']:.6f}",
        )

        print(
            channel_name,
            "permutation p:",
            f"{data['permutation_p_value']:.6f}",
        )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
