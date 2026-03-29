# Qrack Codebase Review - Root Directory SCRATCHPAD

## Project Overview

**Qrack** is a high-performance quantum computing simulator library written in C++11. It provides simulation capabilities for ideal, virtually error-free quantum computers across diverse hardware platforms.

### Key Features
- **Multi-engine architecture**: CPU, GPU (OpenCL/CUDA), and hybrid engines
- **Layered optimization**: QUnit, QPager, QTensorNetwork layers on top of base engines
- **Novel algorithms**: Schmidt decomposition-based optimizations, controlled gate buffer caching
- **Cross-platform**: Linux, Windows, macOS, WebAssembly, Android, iOS
- **Approximate simulation**: Sparse truncation, classical shadow (ACE), near-Clifford simulation

### Core Architecture

```
┌─────────────────────────────────────────┐
│         QInterface (Abstract API)       │
├─────────────────────────────────────────┤
│   QUnit / QUnitClifford / QPager        │  ← Optimization layers
├─────────────────────────────────────────┤
│   QEngineCPU / QEngineOCL / QEngineCUDA │  ← Hardware engines
└─────────────────────────────────────────┘
```

### Build System
- **CMake-based** with modular configuration files in `cmake/`
- **Version**: 10.5.3
- **Dependencies**: Minimal at base (OpenCL for GPU), optional Boost
- **Installable**: PPA for Ubuntu, pip package (pyqrack), source build

### Key Environment Variables
- `QRACK_PSTRIDEPOW`: CPU work item stride power (performance tuning)
- `QRACK_MAX_PAGING_QB`: Maximum qubits per QPager instance
- `QRACK_OCL_DEFAULT_DEVICE`: Default OpenCL device index
- `QRACK_SPARSE_TRUNCATION_THRESHOLD`: Sparse simulation threshold
- `QRACK_DISABLE_QUNIT_FIDELITY_GUARD`: Disable fidelity limiter for ACE

### Notable Algorithms Implemented
- Grover's search
- Shor's factoring
- Quantum Fourier Transform (QFT)
- Quantum teleportation
- Quantum perceptron/associative memory
- Random circuit sampling (RCS)
- Transverse Field Ising Model (TFIM)
- MAXCUT/QUBO optimization

### Testing & Benchmarks
- Catch2 framework for unit tests
- Performance benchmarks for various algorithms
- Code coverage support via gcov/lcov

---

## Directory Structure Summary

| Directory | Purpose |
|-----------|---------|
| `include/` | Header files defining interfaces and classes |
| `src/` | Implementation files |
| `cmake/` | CMake configuration modules |
| `examples/` | Usage examples and demonstrations |
| `test/` | Unit tests and benchmarks |
| `scripts/` | Python scripts for research/validation |
| `debian/` | Debian packaging files |

---

## Next Steps
Continue reviewing subdirectories to document detailed functionality, class hierarchies, and implementation details.
