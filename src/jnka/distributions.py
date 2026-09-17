import numpy as np

EPS = 1e-12

def normalize(p):
    p = np.asarray(p, dtype=float)
    total = np.sum(p)
    if total <= 0:
        raise ValueError("Probability distribution has zero mass.")
    return p / total

def entropy(p):
    p = normalize(p)
    q = np.clip(p, EPS, None)
    return float(-np.sum(p * np.log2(q)))

def kl_divergence(p, q):
    p = normalize(p)
    q = normalize(q)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p, q):
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)
    return float(
        0.5 * kl_divergence(p, m)
        + 0.5 * kl_divergence(q, m)
    )

def total_variation(p, q):
    p = normalize(p)
    q = normalize(q)
    return float(0.5 * np.sum(np.abs(p - q)))

def l1_distance(p, q):
    p = normalize(p)
    q = normalize(q)
    return float(np.sum(np.abs(p - q)))

def counts_to_probability(counts, n_qubits):
    dimension = 2 ** n_qubits
    p = np.zeros(dimension, dtype=float)
    total = sum(counts.values())

    if total <= 0:
        raise ValueError("Empty counts.")

    for bitstring, count in counts.items():
        p[int(bitstring, 2)] = count / total

    return p
