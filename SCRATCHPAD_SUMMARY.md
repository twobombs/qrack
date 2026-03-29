# Qrack Codebase - Comprehensive Review Summary

## Executive Overview

**Qrack** is a high-performance quantum computing simulator library written in C++11, designed to simulate ideal (error-free) quantum computers across diverse hardware platforms. Version 10.5.3, it represents a sophisticated framework combining novel optimization algorithms with multi-engine hardware acceleration.

---

## 1. Core Functionality

### 1.1 Quantum Simulation Engine

Qrack implements **Schroedinger-style state vector simulation** with multiple optimization layers:

```
┌─────────────────────────────────────────────────────────┐
│              QInterface (Abstract API)                  │
│  - Gate operations, measurement, state manipulation     │
│  - Register-like operations on contiguous qubits        │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  QUnit          │ │  QPager         │ │  QTensorNetwork │
│  (Schmidt       │ │  (Paged State   │ │  (Tensor Network│
│   Decomposition)│ │   Vector)       │ │   Optimization) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Hardware Engines                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
│  │ QEngineCPU  │ │ QEngineOCL  │ │ QEngineCUDA     │   │
│  │ (CPU)       │ │ (OpenCL GPU)│ │ (NVIDIA GPU)    │   │
│  └─────────────┘ └─────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Specialized Simulators

| Simulator | Purpose | Key Feature |
|-----------|---------|-------------|
| **QStabilizer** | Clifford-only circuits | Stabilizer formalism (efficient) |
| **QStabilizerHybrid** | Near-Clifford circuits | Reverse gadget for T gates |
| **QBdt** | Compressed representation | Quantum Binary Decision Tree |
| **QUnitClifford** | Clifford-optimized QUnit | Faster than general QUnit |
| **QInterfaceNoisy** | Noisy simulation | Depolarizing noise channels |

---

## 2. Architecture & Design Patterns

### 2.1 Layered Architecture

**QUnit Layer** (Primary Optimization):
- Maintains explicit separability of qubits
- Uses `QEngineShard` for each qubit
- Implements **phase shard buffering** for commuting gates
- Tracks unitary fidelity for **Automatic Circuit Elision (ACE)**
- Based on Schmidt decomposition (arXiv:1710.05867)

**Key QUnit Optimizations:**
1. **Lazy Evaluation**: Delays state vector materialization
2. **Phase Sharding**: Buffles commuting controlled phase gates
3. **Basis Caching**: Tracks X/Z basis for each qubit
4. **Fidelity Tracking**: `logFidelity` for ACE decisions
5. **Parallel Execution**: `ParallelUnitApply()` across shards

### 2.2 Factory Pattern

```cpp
// Create quantum interface with optimal engine selection
QInterfacePtr qReg = CreateQuantumInterface(
    QINTERFACE_OPTIMAL, numQubits, ZERO_BCI);

// Explicit engine selection
QInterfacePtr qReg = CreateQuantumInterface(
    QINTERFACE_QUNIT, numQubits, ZERO_BCI);
QInterfacePtr qReg = CreateQuantumInterface(
    QINTERFACE_QPAGER, numQubits, ZERO_BCI);
```

### 2.3 Memory Management

**QPager** implements paged state vector:
- Splits state into equal-length "pages"
- Dynamic page allocation based on qubit count
- Supports multi-device distribution
- Sparse state vector option for polarized states

```cpp
baseQubitsPerPage = (qubitCount < thresholdQubitsPerPage) 
                    ? qubitCount : thresholdQubitsPerPage;
basePageCount = pow2Ocl(qubitCount - baseQubitsPerPage);
```

---

## 3. Key Algorithms & Implementations

### 3.1 Gate Operations

**Single-Qubit Gates:**
- Pauli gates: `X()`, `Y()`, `Z()`
- Hadamard: `H()`
- Phase gates: `S()`, `SDG()`, `T()`, `TDG()`
- Rotation gates: `RX()`, `RY()`, `RZ()`, `Phase()`

**Multi-Qubit Gates:**
- Controlled gates: `CNOT()`, `CCNOT()`, `CZ()`
- Swap gates: `SWAP()`, `CSWAP()`, `SQRTSWAP()`
- Uniformly controlled: `UCMtrx()`, `UniformlyControlledSingleBit()`

**Arithmetic Operations (QAlu):**
- Increment/Decrement: `INC()`, `DEC()`
- Multiplication/Division: `MUL()`, `DIV()`
- Bitwise: `AND()`, `OR()`, `XOR()`
- Comparison: `CMP()`
- Modular: `POWModNOut()` (Shor's algorithm)

### 3.2 Quantum Fourier Transform

```cpp
void QInterface::QFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    for (bitLenInt i = 0U; i < length; ++i) {
        const bitLenInt hBit = end - i;
        for (bitLenInt j = 0U; j < i; ++j) {
            CPhaseRootN(j + 2U, c, t);
            if (trySeparate) TrySeparate(c, t);
        }
        H(hBit);
    }
}
```

### 3.3 Measurement & Sampling

```cpp
bool result = qReg->M(qubit);           // Single qubit measurement
bitCapInt result = qReg->MAll();        // All qubits measurement
real1 prob = qReg->Prob(qubit);         // Probability of |1>
real1 prob = qReg->ProbAll(state);      // Probability of specific state
bitCapInt sample = qReg->Sample();      // Sample without measurement
```

---

## 4. Hardware Acceleration

### 4.1 CPU Engine (QEngineCPU)

- Pure C++11 implementation
- No external dependencies
- SIMD vectorization (SSE2/AVX)
- pthread parallelism (PSTRIDEPOW batching)
- Sparse state vector support

### 4.2 OpenCL Engine (QEngineOCL)

- GPU acceleration via OpenCL
- Precompiled kernel support
- Out-of-order queue execution (v2.0)
- Multi-device distribution
- Host/device memory options

### 4.3 CUDA Engine (QEngineCUDA)

- NVIDIA GPU acceleration
- GVirtus distributed CUDA support
- Automatic architecture detection

### 4.4 Hybrid Engine (QHybrid)

- Automatic CPU/GPU workload distribution
- Best of both worlds for heterogeneous systems

---

## 5. Build Configuration

### 5.1 CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_OPENCL` | ON | OpenCL GPU support |
| `ENABLE_CUDA` | OFF | CUDA GPU support |
| `ENABLE_PTHREAD` | ON | pthread parallelism |
| `ENABLE_QBDT` | OFF | QBdt support |
| `ENABLE_ALU` | ON | ALU API |
| `FPPOW` | 5 | Floating-point precision (2^5=32-bit) |
| `UINTPOW` | 6 | Qubit addressing width (64-bit) |
| `QBCAPPOW` | 6 | Qubit capacity power (64 qubits) |

### 5.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `QRACK_PSTRIDEPOW` | CPU work item stride power |
| `QRACK_MAX_PAGING_QB` | Max qubits per QPager |
| `QRACK_OCL_DEFAULT_DEVICE` | Default OpenCL device |
| `QRACK_SPARSE_TRUNCATION_THRESHOLD` | Sparse simulation threshold |
| `QRACK_DISABLE_QUNIT_FIDELITY_GUARD` | Disable ACE fidelity limiter |
| `QRACK_QPAGER_DEVICES` | QPager device list |

---

## 6. Approximation & Noise Features

### 6.1 Automatic Circuit Elision (ACE)

When memory limits are approached, QUnit replaces gates with "classical shadows":
- Tracks fidelity estimate (`logFidelity`)
- Throws exception if fidelity drops below threshold
- Can be disabled with `QRACK_DISABLE_QUNIT_FIDELITY_GUARD`

### 6.2 Sparse Simulation

CPU-only sparse state vector:
- Only stores non-zero amplitudes
- Configurable threshold: `QRACK_SPARSE_TRUNCATION_THRESHOLD`
- Memory limit: `QRACK_SPARSE_MAX_ALLOC_MB`

### 6.3 Near-Clifford Simulation

Efficient simulation for Clifford + RZ circuits:
- "Reverse gadget" for T gate injection
- Stochastic approximation option: `QRACK_USE_APPROX_NEAR_CLIFFORD`
- Non-Clifford rounding: `QRACK_NONCLIFFORD_ROUNDING_THRESHOLD`

### 6.4 Noise Channels

Single-qubit depolarizing noise:
- `QRACK_GATE_DEPOLARIZATION` environment variable
- `QInterfaceNoisy` wrapper class

---

## 7. Example Applications

### 7.1 Quantum Algorithms

| Algorithm | Example | Description |
|-----------|---------|-------------|
| Teleportation | `teleport.cpp` | Quantum teleportation protocol |
| Grover's Search | `grovers.cpp` | Unstructured search (quadratic speedup) |
| Shor's Factoring | `shors_factoring.cpp` | Integer factoring via period finding |
| QFT | Various | Quantum Fourier Transform |

### 7.2 Quantum Machine Learning

- `quantum_perceptron.cpp` - Quantum neuron/perceptron
- `quantum_associative_memory.cpp` - Associative memory

### 7.3 Optimization

- `maxcut_approx.py` - Physics-inspired MAXCUT solver
- `ising_depth_series.py` - TFIM phase transition studies

### 7.4 Benchmarks

- `quantum_volume.cpp` - Quantum volume benchmark
- `rcs_nn_tn_qiskit_validation.py` - RCS cross-entropy benchmarking

---

## 8. Testing & Validation

### 8.1 Unit Tests (tests.cpp)

- 7200+ lines of comprehensive tests
- All basic quantum gates
- Arithmetic operations
- Measurement & probability
- State vector manipulation
- Engine-specific features
- Edge cases

### 8.2 Benchmarks (benchmarks.cpp)

- Algorithm performance testing
- Engine scaling tests
- Qubit count scaling
- Circuit depth scaling

### 8.3 Cross-Validation

- Qiskit validation scripts
- Tensor network vs. state vector comparison
- Random circuit sampling validation

---

## 9. Requirements & Dependencies

### 9.1 Build Requirements

- **C++ Compiler**: C++11 or later (default C++14)
- **CMake**: 3.10+
- **Optional**: OpenCL SDK, CUDA Toolkit, Boost (for large qubit counts)

### 9.2 Runtime Requirements

- **CPU**: x86_64 (SSE2/AVX optional)
- **GPU**: OpenCL-compatible or NVIDIA CUDA
- **Memory**: 2^n × 16 bytes per n qubits (state vector)

### 9.3 Platform Support

- Linux (Ubuntu 18.04/20.04/22.04/24.04)
- Windows (10+)
- macOS
- WebAssembly (Emscripten)
- Android
- iOS

---

## 10. Performance Characteristics

### 10.1 Scaling

| Qubits | Memory (State Vector) | Notes |
|--------|----------------------|-------|
| 30 | ~256 MB | Desktop feasible |
| 35 | ~8 GB | High-end desktop |
| 40 | ~256 GB | Server required |
| 50 | ~256 TB | GPU cluster needed |

### 10.2 Optimization Impact

- **QUnit**: Can reduce effective qubit count via separability
- **QPager**: Enables larger simulations via paging
- **Sparse**: Significant savings for polarized states
- **QBdt**: Compression for structured states

### 10.3 Hardware Performance

- **CPU**: Baseline performance, highly tunable via PSTRIDEPOW
- **OpenCL GPU**: 10-100x speedup for large circuits
- **CUDA GPU**: Comparable to OpenCL, platform-dependent

---

## 11. Code Quality & Maintenance

### 11.1 Code Organization

- **Header files**: `include/` (30+ files)
- **Source files**: `src/` (organized by component)
- **Common utilities**: `include/common/`, `src/common/`
- **Tests**: `test/` (Catch2 framework)
- **Examples**: `examples/` (12+ demonstration programs)
- **Scripts**: `scripts/` (Python research utilities)

### 11.2 Documentation

- Doxygen configuration (`doxygen.config`)
- README.md with comprehensive usage guide
- API documentation at readthedocs.io
- Inline code comments

### 11.3 Licensing

- **License**: LGPL v3 (GNU Lesser General Public License)
- **Copyright**: Daniel Strano and Qrack contributors 2017-2026
- **Unitary Fund**: Supported by Unitary Foundation

---

## 12. Key Innovations

1. **Schmidt Decomposition Optimization**: Novel approach to minimize entanglement tracking
2. **Phase Shard Buffering**: Gate fusion for commuting operations
3. **Automatic Circuit Elision**: Memory-limited simulation via classical shadows
4. **Near-Clifford Simulation**: Efficient simulation for Clifford + RZ circuits
5. **Multi-Engine Architecture**: Seamless CPU/GPU/hybrid operation
6. **QBdt**: Compressed state representation via decision trees

---

## 13. Usage Pattern

```cpp
#include "qfactory.hpp"

int main() {
    // Create quantum register with optimal engine
    QInterfacePtr qReg = CreateQuantumInterface(
        QINTERFACE_OPTIMAL, numQubits, ZERO_BCI);
    
    // Prepare state
    qReg->H(0);
    qReg->CNOT(0, 1);
    
    // Apply algorithm
    // ... quantum operations ...
    
    // Measure
    bitCapInt result = qReg->MAll();
    
    return 0;
}
```

---

## 14. Summary

Qrack represents a sophisticated quantum computing simulator framework that combines:

1. **Novel algorithms** (Schmidt decomposition, phase sharding)
2. **Multi-engine architecture** (CPU, OpenCL, CUDA, hybrid)
3. **Approximation techniques** (ACE, sparse, near-Clifford)
4. **Comprehensive testing** (7200+ lines of unit tests)
5. **Research utilities** (TFIM, MAXCUT, quantum chemistry)

The codebase is well-organized, documented, and designed for extensibility. It serves both as a research tool for quantum algorithm development and as a benchmark platform for quantum computing claims.

---

*This summary was generated through systematic review of the Qrack codebase, examining header files, source implementations, examples, tests, and research scripts.*
