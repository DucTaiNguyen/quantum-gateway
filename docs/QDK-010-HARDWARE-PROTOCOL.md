# QDK-010 — IBM Quantum Hardware Validation Protocol

## Objective

Test whether the structure-dependent noise-response signal
observed in QDK-008 and QDK-009 persists under a real
quantum-device execution environment.

## Structures

STRUCTURED:
[[1.0, 0.8],
 [0.8, 1.0]]

WEAK:
[[1.0, 0.1],
 [0.1, 1.0]]

ASYMMETRIC:
[[1.0, 0.2],
 [0.7, 1.0]]

## Protocol

1. Encode each information tensor into a two-qubit state.
2. Apply the same parameterized unitary to every structure.
3. Execute the circuits on an IBM Quantum backend.
4. Repeat measurements with identical shot budgets.
5. Extract observable distributions.
6. Estimate correlation and entropy metrics.
7. Compare structure-dependent trajectories.
8. Quantify statistical uncertainty.
9. Compare hardware results against the NumPy reference model.

## Primary hypothesis

H1:

Noise and device-response trajectories contain measurable
information about the encoded structure.

## Null hypothesis

H0:

After accounting for statistical uncertainty and device noise,
the response trajectories are indistinguishable across structures.

## Scientific requirement

Hardware results must be reported separately from simulation results.

No claim of physical validation is made until real-device
measurements are obtained.

## Reproducibility

Record:

- backend name
- backend configuration
- circuit depth
- number of qubits
- shots
- transpilation settings
- execution timestamp
- random seeds where applicable
- raw measurement counts

## Success criterion

Structure-dependent separation observed on hardware must exceed
the estimated within-structure statistical variation under the
same protocol.

## Status

Reference experimental protocol.
