# ⚛ Quantum Gateway (QG)

> **A Hybrid Classical–Quantum Information Representation and Transformation Layer**

[![License: Commercial](https://img.shields.io/badge/License-Commercial%20Source-red.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-purple.svg)](https://qiskit.org)
[![Version](https://img.shields.io/badge/Version-0.1.0--alpha-orange.svg)](./CHANGELOG.md)
[![Author](https://img.shields.io/badge/Author-Tai%20D.%20Nguyen-green.svg)](https://github.com/taidnguyen)

---

## What is Quantum Gateway?

**Quantum Gateway (QG)** is a novel hybrid computational architecture that provides a structured interface between classical information systems and quantum state representations.

Unlike traditional quantum computing frameworks focused on algorithmic speedup, QG is designed for:

- 🔍 **Representation Discovery** — encode classical data into quantum state spaces to reveal hidden structure
- 🔀 **Information Routing** — transform and route information through quantum dynamics
- 🌊 **State-Space Transformation** — leverage unitary evolution and measurement collapse as computational primitives
- 🔊 **Noise-Driven Exploration** — treat quantum noise as a structural mechanism, not an error

```
Classical Input I  →  Encoding ε  →  Quantum State |ψ(I)⟩
                                              ↓
Classical Output I'  ←  Measurement M  ←  Unitary U(θ)
```

Formally:

```
Q = M ∘ U(θ) ∘ ε
```

---

## Core Architecture

| Layer | Symbol | Role |
|-------|--------|------|
| **Encoding Layer** | `ε` | Maps classical data `I` → Hilbert space `\|ψ(I)⟩` |
| **Evolution Layer** | `U(θ)` | Parameterized unitary circuit transforms quantum state |
| **Measurement Layer** | `M` | Collapses quantum state to classical probability distribution |
| **Noise Model** | `ρ' = ΣₖEₖρEₖ†` | Kraus operator noise for structural diversification |

---

## Installation

### Requirements

- Python 3.9+
- Qiskit 1.x
- NumPy, SciPy
- (Optional) IBM Quantum account for real hardware execution

```bash
# Clone the repository
git clone https://github.com/taidnguyen/quantum-gateway.git
cd quantum-gateway

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

---

## Quick Start

```python
from quantum_gateway import QuantumGateway
from quantum_gateway.encoding import AmplitudeEncoder
from quantum_gateway.evolution import ParameterizedCircuit
from quantum_gateway.measurement import ProbabilisticMeasurer

# Create a 2-qubit Quantum Gateway
qg = QuantumGateway(n_qubits=2)

# Encode classical data
data = [0.5, 0.8, 0.1, 0.3]
encoded_state = qg.encode(data)

# Apply parameterized evolution
import numpy as np
theta = np.pi / 4
evolved_state = qg.evolve(encoded_state, theta=theta)

# Measure output
output = qg.measure(evolved_state, shots=1024)
print("Output distribution:", output)

# Full pipeline (shorthand)
result = qg.transform(data, theta=np.pi/4, shots=1024)
print("Transformed:", result)
```

---

## Architecture Overview

```
quantum_gateway/
├── core/
│   ├── gateway.py          # QuantumGateway main class
│   ├── pipeline.py         # Pipeline orchestration
│   └── registry.py         # Circuit / encoder registry
├── encoding/
│   ├── amplitude.py        # Amplitude encoding
│   ├── angle.py            # Angle encoding (RY/RZ)
│   ├── basis.py            # Computational basis encoding
│   └── dense_angle.py      # Dense angle encoding
├── evolution/
│   ├── parameterized.py    # Variational quantum circuits
│   ├── hardware_efficient.py
│   └── ansatz.py           # Custom ansatz builder
├── measurement/
│   ├── probabilistic.py    # Shot-based sampling
│   ├── statevector.py      # Exact statevector measurement
│   └── expectation.py      # Observable expectation values
├── noise/
│   ├── kraus.py            # Kraus operator noise model
│   ├── depolarizing.py     # Depolarizing channel
│   └── structural.py       # Noise-as-structure interface
└── utils/
    ├── visualization.py    # State & circuit visualization
    ├── metrics.py          # Fidelity, entropy, distance metrics
    └── ibm_backend.py      # IBM Quantum hardware integration
```

---

## ILPS Integration

Quantum Gateway extends the **Information-Logical Processing System (ILPS)** framework:

```
I_{t+1} = O(ε(I_t))
```

Where operator `O` now embeds quantum dynamics — enabling **cognitive graph-to-quantum transformations** inside JARVIS-Q agent stacks.

```
JARVIS-Q Stack:

  [Classical Agents]
         ↓
  [ILPS Graph Memory]
         ↓
  [Quantum Gateway Layer]   ← THIS MODULE
         ↓
  [Quantum Processing / IBM Q]
         ↓
  [Classical Output Decision]
```

---

## Learning Objective

QG supports variational optimization:

```
min_θ  L( M(U(θ) ε(I)),  I_target )
```

This turns Quantum Gateway into a **trainable representation engine** — learning quantum encodings that minimize task loss.

---

## Roadmap

- [x] Core pipeline: encode → evolve → measure
- [x] Noise model (Kraus / depolarizing)
- [ ] `v0.2` — Variational optimizer integration (gradient descent on θ)
- [ ] `v0.3` — ILPS graph embedding interface
- [ ] `v0.4` — Multi-node entanglement routing
- [ ] `v0.5` — IBM Quantum real hardware execution
- [ ] `v1.0` — JARVIS-Q full integration + commercial SDK

---

## Research Paper

This repository is the official implementation of:

> **Nguyen, T. D. (2026).** *Quantum Gateway: A Hybrid Classical–Quantum Information Representation and Transformation Layer.* [Preprint]

---

## License & Commercial Use

This project is released under a **Commercial Source License**.

- ✅ Free for personal study, academic research (with attribution)
- ✅ Open to community contributions (see [CONTRIBUTING.md](./CONTRIBUTING.md))
- ❌ Commercial use requires a paid license

**For commercial licensing, enterprise deployment, or research partnerships:**

📧 [contact@quantumgateway.dev](mailto:contact@quantumgateway.dev)

---

## Author

**Tai D. Nguyen**
*Architect, Quantum Gateway & JARVIS-Q Systems*

> © 2026 Tai D. Nguyen. All rights reserved.
