import numpy as np

from quantum_gateway import (
    InformationTensor,
    QuantumState,
    density_matrix,
    depolarizing_mixture,
    entanglement_entropy,
    entropy,
    fidelity_pure_state,
    noise_metrics,
    probability_distribution,
    purity,
    tensor_to_state,
    zz_correlation,
)


def test_information_tensor_normalization():
    tensor = InformationTensor([[1.0, 0.8], [0.8, 1.0]])
    normalized = tensor.normalize()

    assert normalized.shape == (2, 2)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_probability_distribution():
    values = np.array([1.0, 1.0, 2.0])
    probabilities = probability_distribution(values)

    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)


def test_entropy():
    probabilities = np.array([0.5, 0.5])

    value = entropy(probabilities)

    assert np.isclose(value, 1.0)


def test_zz_correlation():
    probabilities = np.array([0.5, 0.0, 0.0, 0.5])

    value = zz_correlation(probabilities)

    assert np.isclose(value, 1.0)


def test_quantum_state_normalization():
    state = QuantumState([1.0, 1.0])

    assert np.isclose(np.linalg.norm(state.amplitudes), 1.0)
    assert state.dimension == 2
    assert np.isclose(state.probabilities().sum(), 1.0)


def test_tensor_to_state():
    state = tensor_to_state(np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert state.shape == (4,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_density_matrix():
    amplitudes = np.array([1.0, 0.0, 0.0, 0.0])

    rho = density_matrix(amplitudes)

    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho), 1.0)


def test_depolarizing_mixture():
    amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
    rho = density_matrix(amplitudes)

    noisy = depolarizing_mixture(rho, 0.1)

    assert np.isclose(np.trace(noisy), 1.0)
    assert np.allclose(noisy, noisy.conj().T)


def test_purity_and_fidelity():
    amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
    rho = density_matrix(amplitudes)

    assert np.isclose(purity(rho), 1.0)
    assert np.isclose(fidelity_pure_state(rho, amplitudes), 1.0)


def test_noise_metrics():
    amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
    rho = density_matrix(amplitudes)

    metrics = noise_metrics(rho, rho, amplitudes)

    assert "purity" in metrics
    assert "fidelity" in metrics
    assert np.isclose(metrics["purity"], 1.0)
    assert np.isclose(metrics["fidelity"], 1.0)


def test_product_state_has_zero_entanglement():
    amplitudes = np.array([1.0, 0.0, 0.0, 0.0])

    value = entanglement_entropy(amplitudes)

    assert np.isclose(value, 0.0)
