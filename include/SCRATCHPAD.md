# Qrack Codebase Review - Include Directory SCRATCHPAD

## Header File Organization

The `include/` directory contains all public API headers, organized into:
- **Core interfaces** (top-level headers)
- **Common utilities** (`include/common/` subdirectory)

---

## Core Interface Hierarchy

### 1. Abstract Base Classes

#### [`QInterface`](qinterface.hpp:141) - Abstract Quantum Interface
The foundational abstract class defining the quantum computing API.

**Key Responsibilities:**
- Defines the universal quantum register interface
- Exposes gate operations, measurement, and state manipulation
- Implements probability and phase measurement
- Supports register-like operations on contiguous qubit sets

**Key Methods:**
- `X()`, `Y()`, `Z()` - Pauli gates
- `H()` - Hadamard gate
- `CNOT()`, `CCNOT()` - Controlled gates
- `M()`, `MAll()` - Measurement operations
- `Phase()`, `T()`, `S()`, `SDG()` - Phase gates
- `INC()`, `DEC()` - Arithmetic operations
- `GetProb()`, `Sample()`, `MAll()` - Sampling methods

**Engine Types Enum:**
```cpp
enum QInterfaceEngine {
    QINTERFACE_CPU = 0,           // QEngineCPU
    QINTERFACE_OPENCL,            // QEngineOCL
    QINTERFACE_CUDA,              // QEngineCUDA
    QINTERFACE_HYBRID,            // QHybrid (CPU+GPU)
    QINTERFACE_BDT,               // QBdt (decision tree)
    QINTERFACE_BDT_HYBRID,        // QBdtHybrid
    QINTERFACE_STABILIZER,        // QStabilizer (Clifford only)
    QINTERFACE_STABILIZER_HYBRID, // QStabilizerHybrid
    QINTERFACE_QPAGER,            // QPager (paged state vector)
    QINTERFACE_QUNIT,             // QUnit (Schmidt decomposition)
    QINTERFACE_QUNIT_MULTI,       // QUnitMulti (multi-device)
    QINTERFACE_QUNIT_CLIFFORD,    // QUnitClifford
    QINTERFACE_TENSOR_NETWORK,    // QTensorNetwork
    QINTERFACE_NOISY,             // QInterfaceNoisy
    QINTERFACE_OPTIMAL = QINTERFACE_QUNIT
}
```

#### [`QEngine`](qengine.hpp:31) - Base Engine Implementation
Abstract implementation for "Schroedinger method" engines (state vector simulation).

**Key Responsibilities:**
- Manages state vector storage and manipulation
- Implements gate application at the engine level
- Handles normalization and fidelity tracking
- Provides GPU/OpenCL/CUDA-specific operations

**Key Methods:**
- `ZeroAmplitudes()`, `CopyStateVec()` - State management
- `GetAmplitudePage()`, `SetAmplitudePage()` - Page operations
- `Mtrx()`, `MCMtrx()`, `MACMtrx()` - Matrix operations
- `ForceM()`, `ForceMReg()` - Forced measurement
- `ApplyM()` - Apply measurement result

---

### 2. Optimization Layers

#### [`QUnit`](qunit.hpp:28) - Schmidt Decomposition Optimizer
Implements the novel optimization based on Schmidt decomposition (arXiv:1710.05867).

**Key Features:**
- Maintains explicit separability of qubits as optimization
- Uses `QEngineShard` objects for each qubit
- Implements controlled gate buffer caching
- Tracks unitary fidelity estimate
- Supports "Automatic Circuit Elision" (ACE) for memory-limited cases

**Key Members:**
- `shards` - Map of qubit to `QEngineShard`
- `logFidelity` - Fidelity tracking for ACE
- `separabilityThreshold` - Threshold for separability detection
- `aceMb` - ACE memory limit in MB
- `aceQubits` - ACE qubit threshold

**Key Methods:**
- `ParallelUnitApply()` - Apply operation across shards
- `CheckFidelity()` - Check if fidelity is acceptable
- `ElideCz()` - Classical shadow replacement for CZ gates
- `Dump()`, `ForceDump()` - Force state vector materialization

#### [`QPager`](qpager.hpp:31) - Paged State Vector
Splits state vector into equal-length "pages" for memory efficiency.

**Key Features:**
- Dynamic page allocation based on qubit count
- Supports multi-device distribution
- Sparse state vector option
- Automatic page combining/separating

**Key Members:**
- `qPages` - Vector of QEngine pages
- `baseQubitsPerPage` - Qubits per page
- `basePageCount` - Number of pages
- `thresholdQubitsPerPage` - Threshold for page operations
- `isSparse` - Sparse simulation flag

**Key Methods:**
- `CombineEngines()`, `SeparateEngines()` - Page management
- `GetPageDevice()`, `GetPageHostPointer()` - Device selection
- `MakeEngine()` - Create new page engine

#### [`QTensorNetwork`](qtensornetwork.hpp:29) - Tensor Network Optimizer
Gate-based QInterface wrapping cuQuantum for tensor network simulation.

**Key Features:**
- Past light cone optimization for measurement/probability
- Circuit simplification layer
- Supports sparse truncation
- Classical shadow (ACE) support

**Key Methods:**
- `RunAsAmplitudes()` - Execute circuit for amplitude calculation
- `RemovePastLightCone()` - Optimize measurement scope
- `GetUnitaryFidelity()` - Get fidelity estimate

---

### 3. Specialized Engines

#### [`QStabilizerHybrid`](qstabilizerhybrid.hpp:43) - Near-Clifford Simulator
Switches between QStabilizer (Clifford) and QEngine (non-Clifford) as efficient.

**Key Features:**
- "Reverse gadget" for non-Clifford phase injection
- Efficient for near-Clifford circuits (Clifford + RZ)
- MpsShard tracking for buffered operations
- Supports stochastic approximation

**Key Members:**
- `stabilizer` - QUnitClifford for Clifford operations
- `engine` - QEngine for non-Clifford operations
- `shards` - MpsShard vector for buffering
- `isNearCliffordExact` - Exact vs. stochastic mode

**Key Methods:**
- `FlushH()`, `InvertBuffer()` - Buffer management
- `CollapseSeparableShard()` - Shard collapse
- `MakeStabilizer()`, `MakeEngine()` - Engine creation

#### [`QBdt`](qbdt.hpp:37) - Quantum Binary Decision Tree
Alternative quantum state representation using decision trees.

**Key Features:**
- Compressed state representation
- Direct operation on compressed form
- Inspiration from JKQ DDSIM
- Hybrid mode with state vector

**Key Members:**
- `root` - QBdtNodeInterfacePtr (tree root)
- `bdtStride` - Tree traversal stride
- `bdtMaxQPower` - Maximum qubit power

**Key Methods:**
- `GetTraversal()`, `SetTraversal()` - State vector I/O
- `ExecuteAsStateVector()` - Hybrid mode execution
- `ParForQbdt()` - Parallel tree operations

#### [`QUnitClifford`](qunitclifford.hpp) - Clifford-Only QUnit
QUnit specialized for Clifford gates only.

**Key Features:**
- Highly optimized for Clifford circuits
- Uses stabilizer formalism
- Faster than general QUnit for Clifford-only circuits

---

### 4. Multi-Device & Hybrid

#### [`QHybrid`](qhybrid.hpp) - CPU/GPU Hybrid Engine
Switches between QEngineCPU and QEngineOCL as efficient.

**Key Features:**
- Automatic CPU/GPU workload distribution
- Best of both worlds for heterogeneous systems

#### [`QUnitMulti`](qunitmulti.hpp) - Multi-Device QUnit
Distributes QUnit "shards" across available OpenCL devices.

**Key Features:**
- Multi-GPU support
- Load balancing across devices
- Device list configuration via environment variables

---

### 5. Additional Interfaces

#### [`QInterfaceNoisy`](qinterface_noisy.hpp) - Noisy Simulation Wrapper
Adds noise channels to quantum simulation.

**Key Features:**
- Single-qubit depolarizing noise
- Environment variable configuration
- Wrapper pattern for existing interfaces

#### [`QAlu`](qalu.hpp) - Arithmetic Logic Unit API
Arithmetic operations on quantum registers.

**Key Methods:**
- `INC()`, `DEC()` - Increment/decrement
- `MUL()`, `DIV()` - Multiplication/division
- `AND()`, `OR()`, `XOR()` - Bitwise operations
- `CMP()` - Comparison

#### [`QParity`](qparity.hpp) - Parity Operations
Parity-based operations for optimization.

**Key Methods:**
- `PhaseParity()` - Parity phase
- `XParity()`, `ZParity()` - Parity measurements

#### [`QCircuit`](qcircuit.hpp) - Circuit Representation
Circuit abstraction for tensor network simulation.

---

## Common Utilities (`include/common/`)

### Type Definitions

#### [`qrack_types.hpp`](common/qrack_types.hpp:13) - Core Type Definitions
Defines all fundamental types used throughout Qrack.

**Key Type Aliases:**
```cpp
// Qubit indexing (configurable via QBCAPPOW)
typedef uint64_t bitLenInt;      // Qubit index
typedef uint64_t bitCapInt;      // Qubit power mask
typedef uint64_t bitCapIntOcl;   // OpenCL qubit power

// Floating point (configurable via FPPOW)
typedef float real1;             // Primary floating point
typedef float real1_f;           // Float for comparisons
typedef float real1_s;           // Float for sums
typedef std::complex<real1> complex;  // Complex numbers
```

**Configuration Macros:**
- `IS_AMP_0(c)` - Amplitude is zero
- `IS_NORM_0(c)` - Norm is zero
- `IS_SAME(c1, c2)` - Complex numbers are same
- `IS_OPPOSITE(c1, c2)` - Complex numbers are opposite

#### [`qrack_functions.hpp`](common/qrack_functions.hpp) - Utility Functions
Common mathematical and quantum computing functions.

**Key Functions:**
- `pow2()`, `pow2Ocl()` - Power of 2
- `log2Ocl()` - Logarithm base 2
- `norm()` - Complex norm
- `Rand()`, `RandRange()` - Random number generation

### Parallel Processing

#### [`parallel_for.hpp`](common/parallel_for.hpp) - Parallel For Loop
Parallel execution framework for quantum operations.

**Key Features:**
- Thread pool management
- Work item batching (PSTRIDEPOW)
- OpenMP and pthread support

#### [`dispatchqueue.hpp`](common/dispatchqueue.hpp) - Task Dispatch Queue
Asynchronous task dispatching for QUnit parallelism.

### GPU Support

#### [`oclengine.hpp`](common/oclengine.hpp) - OpenCL Engine
OpenCL device management and kernel execution.

**Key Features:**
- Device enumeration
- Kernel compilation
- Memory management
- Out-of-order queue support

#### [`cudaengine.cuh`](common/cudaengine.cuh) - CUDA Engine
NVIDIA CUDA device management.

**Key Features:**
- CUDA device enumeration
- Memory allocation
- Kernel execution

#### [`cuda_kernels.cuh`](common/cuda_kernels.cuh) - CUDA Kernels
CUDA kernel definitions for GPU operations.

### Random Number Generation

#### [`rdrandwrapper.hpp`](common/rdrandwrapper.hpp) - Hardware RNG Wrapper
Hardware random number generation via RDRAND instruction.

**Key Features:**
- x86_64 RDRAND support
- Fallback to software RNG
- Thread-safe generation

### Specialized Types

#### [`big_integer.hpp`](common/big_integer.hpp) - Big Integer Support
Arbitrary-precision integer arithmetic for large qubit counts.

#### [`half.hpp`](common/half.hpp) - Half Precision Float
IEEE 754 half-precision floating point support.

#### [`complex8x2simd.hpp`](common/complex8x2simd.hpp) - SIMD Complex
SSE2-accelerated complex number operations (32-bit).

#### [`complex16x2simd.hpp`](common/complex16x2simd.hpp) - SIMD Complex
SSE2-accelerated complex number operations (64-bit).

#### [`pauli.hpp`](common/pauli.hpp) - Pauli Operations
Pauli gate definitions and operations.

#### [`qneuron_activation_function.hpp`](common/qneuron_activation_function.hpp) - Neural Network
Activation functions for quantum neural networks.

---

## State Vector Types

#### [`statevector.hpp`](statevector.hpp) - State Vector Classes
Defines various state vector representations.

**Key Classes:**
- `StateVector` - Dense state vector
- `StateVectorArray` - Array-based state vector
- `StateVectorSparse` - Sparse state vector

---

## Factory Pattern

#### [`qfactory.hpp`](qfactory.hpp:45) - Quantum Interface Factory
Template-based factory for creating quantum interfaces.

**Key Function:**
```cpp
template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, Ts... args)
```

**Usage:**
```cpp
auto qreg = CreateQuantumInterface(QINTERFACE_QUNIT, numQubits);
auto qreg = CreateQuantumInterface(QINTERFACE_QPAGER, numQubits);
auto qreg = CreateQuantumInterface(QINTERFACE_OPENCL, numQubits);
```

---

## Build Configuration

#### [`config.h.in`](common/config.h.in) - Generated Configuration
Generated during CMake build with compile-time options.

**Key Macros:**
- `ENABLE_OPENCL` - OpenCL support
- `ENABLE_CUDA` - CUDA support
- `ENABLE_PTHREAD` - pthread support
- `ENABLE_QBDT` - QBdt support
- `ENABLE_ALU` - ALU API
- `FPPOW` - Floating point precision
- `UINTPOW` - Qubit addressing width
- `QBCAPPOW` - Qubit capacity power

---

## Next Steps
Continue reviewing src/ directory for implementation details.
