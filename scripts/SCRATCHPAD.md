# Qrack Codebase Review - Scripts Directory SCRATCHPAD

## Python Scripts

The `scripts/` directory contains Python utilities for research, validation, and specialized quantum computing applications.

---

## Research & Validation Scripts

### [`fc_tn_qiskit_validation.py`](fc_tn_qiskit_validation.py) - Full Circuit Tensor Network Qiskit Validation
Validates Qrack's tensor network simulation against Qiskit.

**Purpose:**
- Cross-validation of quantum circuit simulation results
- Tensor network vs. state vector comparison
- Verification of Qrack's TN implementation

---

### [`rcs_nn_tn_qiskit_validation.py`](rcs_nn_tn_qiskit_validation.py) - Random Circuit Sampling Neural Network Tensor Network Qiskit Validation
RCS benchmark validation with neural network analysis.

**Purpose:**
- Random circuit sampling (RCS) benchmark
- Cross-entropy benchmarking (XEB) fidelity
- Neural network analysis of output distribution

---

## Transverse Field Ising Model (TFIM)

### [`generate_tfim_samples.py`](generate_tfim_samples.py) - TFIM Sample Generation (~225 lines)
Generates measurement samples for TFIM simulations.

**Key Features:**
- Ising model Trotterization
- Closeness-of-like-bits metric for toroidal grids
- Expected closeness weight calculation
- Supports various Ising regimes:
  - Pure ferromagnetic (J=-1, h=0)
  - Pure transverse field (J=0, h=2)
  - Critical point (J=-1, h=1)

**Usage:**
```python
# Generate samples for n-qubit system
python generate_tfim_samples.py 16  # 16 qubits
```

---

### [`ising_depth_series.py`](ising_depth_series.py) - Ising Depth Series
Runs TFIM simulations at varying circuit depths.

**Purpose:**
- Depth scaling analysis
- Phase transition studies
- Critical point detection

---

### [`tfim_model_fit.py`](tfim_model_fit.py) - TFIM Model Fitting
Fits analytical models to TFIM simulation data.

---

### [`tfim_solver_functions.py`](tfim_solver_functions.py) - TFIM Solver Functions
Core TFIM solving utilities.

---

## MAXCUT/QUBO Optimization

### [`maxcut_approx.py`](maxcut_approx.py) - MAXCUT Approximate Solver (~330 lines)
Physics-inspired approximate solver for MAXCUT problem.

**Key Features:**
- (n+1)-dimensional approximation model
- No Trotter error for uniform parameters
- Geometric series weighting for Hamming weights
- Numba JIT compilation for performance
- Supports arbitrary graph structures

**Algorithm:**
1. Reduce TFIM to (n+1)-dimensional problem
2. Use geometric series for Hamming weight distribution
3. Apply oscillation component with frequency proportional to J
4. Extract optimal cut from probability distribution

**Usage:**
```python
# Solve MAXCUT on graph
python maxcut_approx.py graph_file
```

---

### [`maxcut_exact.py`](maxcut_exact.py) - MAXCUT Exact Solver
Exact MAXCUT solution for comparison.

---

## Quantum Random Number Generation

### [`qrng.py`](qrng.py:1) - Quantum Random Number Generator (~25 lines)
Generates quantum random bit strings for Qrack.

**Purpose:**
- Generate true random numbers from quantum sources
- Feed Qrack's RNG file mode (`ENABLE_RNDFILE`)
- Uses ANU Quantum Numbers service

**Usage:**
```python
# Generate 1 page of random data
python qrng.py 1
# Saves to ~/.qrack/rng/qrng.bin
```

**Integration:**
```cmake
cmake -DENABLE_RNDFILE=ON -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=OFF ..
```

---

## Supply Chain Analysis

### [`supply_chain.py`](supply_chain.py) - Supply Chain Analysis
Analyzes quantum computing supply chain implications.

---

## Quantum Chemistry

### [`quantum_chemistry/`](quantum_chemistry/) - Quantum Chemistry Scripts

#### [`README.md`](quantum_chemistry/README.md)
Documents quantum chemistry utilities.

**Key Insight:**
> VQE returns no improvement over correct choice of Hartree-Fock priors for multiplicity and charge.

---

### Control Scripts

#### [`clifford_vqe_entangled.py`](quantum_chemistry/control/clifford_vqe_entangled.py)
Clifford VQE with entangled states.

#### [`clifford_vqe_min_xyz.py`](quantum_chemistry/control/clifford_vqe_min_xyz.py)
Minimal VQE with XYZ coupling.

#### [`clifford_vqe_positive_control.py`](quantum_chemistry/control/clifford_vqe_positive_control.py)
Positive control for VQE experiments.

#### [`clifford_vqe_xyz_cnot.py`](quantum_chemistry/control/clifford_vqe_xyz_cnot.py)
VQE with XYZ and CNOT gates.

---

### Experiment Scripts

#### [`clifford_vqe_min.py`](quantum_chemistry/experiment/clifford_vqe_min.py)
Minimal VQE experiment.

#### [`clifford_vqe_streaming.py`](quantum_chemistry/experiment/clifford_vqe_streaming.py)
Streaming VQE for large systems.

---

## Script Categories

| Category | Scripts |
|----------|---------|
| Validation | fc_tn_qiskit_validation.py, rcs_nn_tn_qiskit_validation.py |
| TFIM | generate_tfim_samples.py, ising_depth_series.py, tfim_model_fit.py, tfim_solver_functions.py |
| MAXCUT | maxcut_approx.py, maxcut_exact.py |
| RNG | qrng.py |
| Supply Chain | supply_chain.py |
| Quantum Chemistry | clifford_vqe_*.py (control & experiment) |

---

## Common Dependencies

```python
import numpy as np
import networkx as nx
from numba import njit
import itertools
import math
import multiprocessing
```

---

## Research Applications

1. **Quantum Supremacy Benchmarks**
   - Random circuit sampling (RCS)
   - Cross-entropy benchmarking (XEB)

2. **Optimization Problems**
   - MAXCUT on arbitrary graphs
   - QUBO formulation

3. **Many-Body Physics**
   - Transverse Field Ising Model
   - Phase transitions
   - Critical point detection

4. **Quantum Chemistry**
   - Variational Quantum Eigensolver (VQE)
   - Hartree-Fock comparison

---

## Next Steps
Synthesize findings into final summary.
