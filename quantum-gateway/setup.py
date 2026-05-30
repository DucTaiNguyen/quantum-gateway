"""
Quantum Gateway — setup.py
© 2026 Tai D. Nguyen. All rights reserved.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantum-gateway",
    version="0.1.0a1",
    author="Tai D. Nguyen",
    author_email="contact@quantumgateway.dev",
    description="A Hybrid Classical–Quantum Information Representation and Transformation Layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taidnguyen/quantum-gateway",
    project_urls={
        "Documentation": "https://docs.quantumgateway.dev",
        "Research Paper": "https://quantumgateway.dev/paper",
        "Bug Tracker": "https://github.com/taidnguyen/quantum-gateway/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "ibm": ["qiskit-ibm-runtime>=0.20.0"],
        "viz": ["pylatexenc>=2.10", "ipython>=8.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: Other/Proprietary License",
    ],
    keywords=[
        "quantum computing", "quantum machine learning",
        "hybrid quantum-classical", "qiskit", "representation learning",
        "quantum gateway", "ILPS", "JARVIS-Q"
    ],
)
