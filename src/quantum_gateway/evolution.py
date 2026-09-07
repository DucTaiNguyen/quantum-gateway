import numpy as np

I = np.eye(2, dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    return np.kron(a, b)


def ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)

    return np.array([[c, -s], [s, c]], dtype=complex)


def cnot():
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )


def normalize_state(state):
    state = np.asarray(state, dtype=complex).flatten()
    norm = np.linalg.norm(state)

    if norm == 0:
        raise ValueError("State vector cannot be zero.")

    return state / norm


def evolve_two_qubit_state(amplitudes, theta=0.0):
    """
    Reference two-qubit evolution:

        U = CNOT · (RY(theta) ⊗ RY(theta))

    The input may already contain correlations or entanglement.
    """

    psi0 = normalize_state(amplitudes)

    local = kron(ry(theta), ry(theta))
    U = cnot() @ local

    psi1 = U @ psi0
    psi1 = normalize_state(psi1)

    return psi0, psi1


def density_matrix(state):
    psi = normalize_state(state)
    return np.outer(psi, np.conjugate(psi))


def reduced_density_matrix_A(state):
    rho = density_matrix(state)

    rho4 = rho.reshape(2, 2, 2, 2)

    return np.trace(rho4, axis1=1, axis2=3)


def von_neumann_entropy(rho):
    rho = np.asarray(rho, dtype=complex)

    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.real(eigenvalues)

    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)

    nonzero = eigenvalues[eigenvalues > 1e-12]

    if len(nonzero) == 0:
        return 0.0

    return float(-np.sum(nonzero * np.log2(nonzero)))


def entanglement_entropy(state):
    rho_A = reduced_density_matrix_A(state)
    return von_neumann_entropy(rho_A)
