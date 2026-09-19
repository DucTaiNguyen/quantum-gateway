import numpy as np


def normalize_state(state):
    state = np.asarray(state, dtype=complex).flatten()
    norm = np.linalg.norm(state)

    if norm == 0:
        raise ValueError("State cannot be zero.")

    return state / norm


def depolarizing_mixture(rho, p):
    """
    Reference depolarizing-like channel.

    rho' = (1-p) rho + p I/d

    p = 0 -> no noise
    p = 1 -> maximally mixed state
    """

    rho = np.asarray(rho, dtype=complex)

    d = rho.shape[0]

    if rho.shape != (d, d):
        raise ValueError("rho must be square.")

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")

    identity = np.eye(d, dtype=complex) / d

    return (1.0 - p) * rho + p * identity


def mixed_state_entropy(rho):
    rho = np.asarray(rho, dtype=complex)

    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)

    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    if len(eigenvalues) == 0:
        return 0.0

    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def purity(rho):
    rho = np.asarray(rho, dtype=complex)

    return float(np.real(np.trace(rho @ rho)))


def fidelity_pure_state(state, rho):
    """
    Fidelity between a pure state vector and a density matrix.

    For compatibility, accepts the historical call ordering
    fidelity_pure_state(rho, state).
    """
    state = np.asarray(state, dtype=complex)
    rho = np.asarray(rho, dtype=complex)

    if state.ndim == 2 and rho.ndim == 1:
        state, rho = rho, state

    psi = normalize_state(state)

    if rho.shape != (psi.size, psi.size):
        raise ValueError(
            f"rho shape {rho.shape} is incompatible with state dimension {psi.size}"
        )

    value = np.vdot(psi, rho @ psi)

    return float(np.real(value))


def noise_metrics(state_or_rho, p_or_rho, amplitudes=None):
    """
    Compute standard noise metrics.

    Preferred API:
        noise_metrics(state, p)

    Backward-compatible API:
        noise_metrics(rho_reference, rho_noisy, amplitudes)
    """
    if amplitudes is not None:
        reference_rho = np.asarray(state_or_rho, dtype=complex)
        noisy = np.asarray(p_or_rho, dtype=complex)
        psi = normalize_state(amplitudes)

        return {
            "purity": purity(noisy),
            "fidelity": fidelity_pure_state(psi, noisy),
            "trace": float(np.real(np.trace(noisy))),
        }

    psi = normalize_state(state_or_rho)
    p = float(p_or_rho)

    rho = np.outer(psi, np.conjugate(psi))
    noisy = depolarizing_mixture(rho, p)

    return {
        "noise_probability": p,
        "entropy": mixed_state_entropy(noisy),
        "purity": purity(noisy),
        "fidelity": fidelity_pure_state(psi, noisy),
        "trace": float(np.real(np.trace(noisy))),
    }
