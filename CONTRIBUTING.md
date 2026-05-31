# Contributing to Quantum Gateway

Thank you for your interest in contributing to Quantum Gateway.

© 2026 Tai D. Nguyen. All rights reserved.

---

## Contributor License Agreement

By submitting a pull request or any contribution to this repository, you agree to the following:

1. **Copyright Assignment**: You assign to Tai D. Nguyen joint copyright of all contributions.
2. **Unrestricted Rights**: You grant the Author the right to use, modify, license, and commercially distribute your contribution under any terms.
3. **Originality**: You certify that the contribution is your original work and you have the legal right to make this assignment.

**If you do not agree to these terms, do not submit contributions.**

---

## How to Contribute

### Reporting Bugs
Open an issue with:
- Python and Qiskit version
- Minimal reproducible example
- Expected vs actual behavior

### Proposing Features
Open an issue tagged `enhancement` with:
- Problem statement and motivation
- Proposed API design
- Relation to core QG architecture (encoding / evolution / measurement / noise)

### Submitting Code
1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Write code with docstrings following the existing style
4. Add tests in `tests/`
5. Run: `pytest tests/ -v`
6. Submit a pull request with a clear description

### Code Style
- Follow PEP 8
- Use type hints throughout
- All public methods must have docstrings
- Every file must include the copyright header:

```python
# © 2026 Tai D. Nguyen. All rights reserved.
# Quantum Gateway — [Module Name]
```

---

## Areas Seeking Contribution
- Hardware-efficient amplitude encoding (v0.3)
- Parameter shift rule gradient implementation (v0.2)
- IBM Quantum hardware runner (v0.5)
- Visualization tools for quantum state evolution
- Additional noise channel models

---

*Quantum Gateway is the intellectual property of Tai D. Nguyen.*
*All contributions become part of a commercially licensed project.*
