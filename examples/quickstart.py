# examples/quickstart.py
#
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — Quickstart Example

"""
Quantum Gateway — Quickstart Example
=====================================

Demonstrates the full Q = M ∘ U(θ) ∘ ε pipeline:

1. Encode classical data with AngleEncoder
2. Apply parameterized evolution U(θ)
3. Measure output distribution
4. Compute loss against a target distribution
5. (Optional) Noisy simulation with KrausNoiseModel

Run:
    python examples/quickstart.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# ─── Imports ──────────────────────────────────────────────────────────────────

from core.gateway import QuantumGateway
from noise.kraus import KrausNoiseModel

# ─── 1. System Info ───────────────────────────────────────────────────────────

print("=" * 60)
print("  QUANTUM GATEWAY — Quickstart Demo")
print("=" * 60)

qg = QuantumGateway(n_qubits=2, encoding="angle", backend="statevector")
info = qg.info()

for k, v in info.items():
    print(f"  {k:15s}: {v}")
print()

# ─── 2. Encode classical data ─────────────────────────────────────────────────

print("─── Step 1: Encoding ────────────────────────────────────────")
data = [0.5, 0.8]  # classical input I ∈ [0,1]^2
circuit = qg.encode(data)
print(f"  Input data : {data}")
print(f"  Circuit    : {circuit.num_qubits} qubits, depth={circuit.depth()}")
print(circuit.draw(output="text"))

# ─── 3. Evolve with U(θ) ─────────────────────────────────────────────────────

print("─── Step 2: Evolution U(θ) ──────────────────────────────────")
theta = np.pi / 4
evolved = qg.evolve(circuit, theta=theta)
print(f"  θ = {theta:.4f} rad ({np.degrees(theta):.1f}°)")
print(f"  Evolved depth: {evolved.depth()}")

# ─── 4. Measure output ────────────────────────────────────────────────────────

print("─── Step 3: Measurement M ───────────────────────────────────")
result = qg.measure(evolved)
print(f"  Entropy: {result['entropy']:.4f} bits")
print("  Probabilities:")
for bs, p in sorted(result["probabilities"].items()):
    bar = "█" * int(p * 40)
    print(f"    |{bs}⟩  {p:.4f}  {bar}")

# ─── 5. Full pipeline (shorthand) ─────────────────────────────────────────────

print("─── Full Pipeline Q = M ∘ U(θ) ∘ ε ────────────────────────")
full_result = qg.transform(data, theta=theta, shots=1024)
print(f"  Elapsed: {full_result['elapsed_ms']:.2f} ms")
print(f"  Entropy: {full_result['entropy']:.4f} bits")
print(f"  Metadata: {full_result['metadata']}")

# ─── 6. Loss computation ──────────────────────────────────────────────────────

print("─── Variational Loss ─────────────────────────────────────────")
# Target: uniform distribution
target = [0.25, 0.25, 0.25, 0.25]
loss = qg.loss(data, target=target, theta=theta)
print(f"  Loss (KL divergence from uniform): {loss:.6f}")

# Sweep θ to show loss landscape
print("\n  Loss landscape over θ:")
thetas = np.linspace(0, 2 * np.pi, 9)
for t in thetas:
    l = qg.loss(data, target=target, theta=t)
    bar = "▓" * int(l * 10)
    print(f"    θ={t:.2f}  loss={l:.4f}  {bar}")

# ─── 7. Noisy simulation ──────────────────────────────────────────────────────

print("─── Noisy Simulation (Kraus Model) ──────────────────────────")
noise = KrausNoiseModel(depolarizing_rate=0.02, amplitude_damping_rate=0.01)
qg_noisy = QuantumGateway(
    n_qubits=2, encoding="angle", backend="statevector", noise_model=noise
)
noisy_result = qg_noisy.transform(data, theta=theta, shots=1024)
print(f"  Noise summary: {noise.kraus_summary()}")
print(f"  Noisy entropy: {noisy_result['entropy']:.4f} bits")

# ─── 8. Batch encoding ────────────────────────────────────────────────────────

print("─── Batch Encoding ──────────────────────────────────────────")
batch = [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
circuits = qg._encoder.encode_batch(batch)
print(f"  Encoded {len(circuits)} samples → {len(circuits)} circuits")

print()
print("=" * 60)
print("  Demo complete.")
print("  © 2026 Tai D. Nguyen — Quantum Gateway")
print("=" * 60)
