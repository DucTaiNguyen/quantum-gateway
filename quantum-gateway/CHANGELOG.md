# Changelog — Quantum Gateway

All notable changes are documented here.
Format: [Semantic Versioning](https://semver.org/)

© 2026 Tai D. Nguyen. All rights reserved.

---

## [Unreleased]

### Planned
- Variational gradient optimizer (parameter shift rule)
- ILPS graph embedding interface
- Multi-node entanglement routing
- IBM Quantum real hardware execution
- JARVIS-Q integration layer

---

## [0.1.0-alpha] — 2026-05-30

### Added
- Core `QuantumGateway` class implementing `Q = M ∘ U(θ) ∘ ε`
- `GatewayPipeline`: full pipeline orchestration with timing and history
- **Encoding Layer**
  - `AngleEncoder`: classical data → Bloch sphere rotations (RY gates)
  - `AmplitudeEncoder`: classical data → quantum amplitudes |ψ(x)⟩
- **Evolution Layer**
  - `ParameterizedEvolution`: variational ansatz U(θ) with CX entanglement
  - Multi-layer support with configurable depth
- **Measurement Layer**
  - `StatevectorMeasurer`: exact Born rule probability extraction
  - `ProbabilisticMeasurer`: shot-based QASM sampling (hardware-realistic)
- **Noise Model**
  - `KrausNoiseModel`: depolarizing + amplitude damping channels
  - Structural noise philosophy: noise as exploration, not error
- **Utilities**
  - `metrics.py`: fidelity, von Neumann entropy, KL divergence, trace distance
- Loss function `qg.loss()` for variational optimization
- Commercial Source License (© 2026 Tai D. Nguyen)
- Full documentation in README.md
- GitHub-ready repository structure

### Architecture
- Formal definition: `Q: I → |ψ(I)⟩ → I'`
- ILPS integration model documented
- JARVIS-Q stack compatibility layer described

---

*Quantum Gateway is a research project by Tai D. Nguyen.*
*Commercial use requires a license. See LICENSE for details.*
