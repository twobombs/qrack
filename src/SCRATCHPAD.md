# Qrack Codebase Review - Source Directory SCRATCHPAD

## Source File Organization

The `src/` directory contains implementation files organized by component:

```
src/
├── common/           # Shared utilities and platform-specific code
├── qengine/          # Base engine implementations
├── qinterface/       # Interface method implementations
├── qbdt/             # Quantum Binary Decision Tree implementation
├── *.cpp             # Top-level class implementations
└── *.cu              # CUDA GPU kernels
```

---

## Core Implementation Files

### [`qunit.cpp`](qunit.cpp:1) - QUnit Implementation (~4300 lines)
The heart of Qrack's optimization layer.

**Key Implementation Details:**

**Phase Shard Optimization:**
- Maintains `PhaseShard` objects for each controlled phase relationship
- Buffles commuting controlled phase gates to avoid unnecessary entanglement
- Uses `controlsShards`, `targetOfShards`, `antiControlsShards`, `antiTargetOfShards` maps
- Implements "gate fusion" by composing phase gates before application

**Cache States:**
```cpp
#define DIRTY(shard) (shard.isPhaseDirty || shard.isProbDirty)
#define CACHED_X(shard) ((shard.pauliBasis == PauliX) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_Z(shard) ((shard.pauliBasis == PauliZ) && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_ZERO(q) (CACHED_Z(shards[q]) && ProbBase(q) <= FP_NORM_EPSILON)
#define CACHED_ONE(q) (CACHED_Z(shards[q]) && (ONE_R1_F - ProbBase(q)) <= FP_NORM_EPSILON)
```

**Key Methods:**
- `Dump()` - Force materialization of all shards into state vector
- `ForceDump()` - Force dump with fidelity tracking
- `ParallelUnitApply()` - Apply operation across all shards in parallel
- `CheckFidelity()` - Check if accumulated infidelity exceeds threshold
- `ElideCz()` - Classical shadow replacement for CZ gates (ACE)

**Gate Application Flow:**
1. Check if gate can be applied to cached shard state
2. If not, call `Dump()` to materialize state vector
3. Apply gate to underlying engine
4. Update shard cache state

---

### [`qpager.cpp`](qpager.cpp:1) - QPager Implementation (~530 lines)
Paged state vector management.

**Key Implementation Details:**

**Page Management:**
```cpp
baseQubitsPerPage = (qubitCount < thresholdQubitsPerPage) ? qubitCount : thresholdQubitsPerPage;
basePageCount = pow2Ocl(qubitCount - baseQubitsPerPage);
```

**Key Methods:**
- `CombineEngines()` - Merge pages when qubit count is low
- `SeparateEngines()` - Split pages when qubit count is high
- `MakeEngine()` - Create new page with specified engine type
- `GetPageDevice()` - Select device for specific page

**Sparse Mode:**
- When `isSparse` is true, uses sparse state vector representation
- Only stores non-zero amplitudes
- Memory efficient for highly polarized states

---

### [`qengineshard.cpp`](qengineshard.cpp:1) - QEngineShard Implementation (~430 lines)
Per-qubit state tracking for QUnit.

**Key Implementation Details:**

**Shard Structure:**
```cpp
struct QEngineShard {
    QInterfacePtr unit;           // Underlying engine for entangled group
    complex amp0, amp1;           // Amplitudes in Z basis
    PauliBasisEnum pauliBasis;    // Current basis (X or Z)
    bool isProbDirty, isPhaseDirty; // Cache dirty flags
    ShardToPhaseMap controlsShards;   // Phase relationships with controls
    ShardToPhaseMap targetOfShards;   // Phase relationships as target
    // ... more maps for anti-controlled
};
```

**Key Methods:**
- `ClampAmps()` - Clamp amplitudes to basis states if near-zero
- `DumpMultiBit()` - Remove all phase shard buffers
- `AddBuffer()` - Add phase buffer between two shards
- `AddAngles()` - Compose phase angles
- `OptimizeBuffer()` - Optimize phase buffer application

---

### [`qinterface.cpp`](qinterface.cpp:1) - QInterface Implementation (~1400 lines)
Base interface method implementations.

**Key Implementations:**

**Quantum Fourier Transform:**
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

**Key Methods:**
- `SetPermutation()` - Initialize to specific basis state
- `QFT()` / `IQFT()` - Quantum Fourier transform
- `XMask()`, `YMask()`, `ZMask()` - Mask-based bit flips
- `UniformlyControlledSingleBit()` - UCRX/UCRY/UCRZ gates
- `M()` - Measurement with probability sampling
- `Sample()` - Sample without measurement

---

### [`qengine.cpp`](qengine/qengine.cpp:1) - QEngine Implementation (~610 lines)
Base engine gate implementations.

**Key Implementations:**

**Gate Application:**
```cpp
void QEngine::Mtrx(const complex* mtrx, bitLenInt qubit)
{
    if (IsIdentity(mtrx, false)) return;
    const bitCapInt qPowers[1U]{ pow2(qubit) };
    Apply2x2(0U, qPowers[0U], mtrx, 1U, qPowers, doNormalize);
}
```

**Key Methods:**
- `EitherMtrx()` - Apply controlled or anti-controlled gate
- `UCMtrx()` - Uniformly controlled matrix
- `ForceM()` - Forced measurement
- `ForceMReg()` - Forced register measurement
- `ApplyM()` - Apply measurement result to state

---

### [`qinterface/gates.cpp`](qinterface/gates.cpp:1) - Gate Implementations (~500 lines)
Standard quantum gate implementations.

**Key Gates:**
- `UCMtrx()` - Uniformly controlled matrix
- `UniformlyControlledSingleBit()` - UCRX/UCRY/UCRZ
- `ZeroPhaseFlip()` - Phase flip on |0> state
- `XMask()`, `YMask()`, `ZMask()` - Mask operations

---

### [`qinterface/logic.cpp`](qinterface/logic.cpp) - Logic Gate Implementations
Logical operations (AND, OR, XOR, CMP).

---

### [`qinterface/arithmetic.cpp`](qinterface/arithmetic.cpp) - Arithmetic Implementations
Arithmetic operations (INC, DEC, MUL, DIV).

---

### [`qinterface/rotational.cpp`](qinterface/rotational.cpp) - Rotation Implementations
Rotation gates (RX, RY, RZ, phase rotations).

---

### [`qstabilizer.cpp`](qstabilizer.cpp) - QStabilizer Implementation
Clifford-only simulation using stabilizer formalism.

---

### [`qstabilizerhybrid.cpp`](qstabilizerhybrid.cpp) - QStabilizerHybrid Implementation
Switches between stabilizer and state vector for near-Clifford circuits.

**Key Features:**
- "Reverse gadget" for T gate injection
- MpsShard tracking for buffered operations
- Supports stochastic approximation

---

### [`qhybrid.cpp`](qhybrid.cpp) - QHybrid Implementation
Switches between CPU and GPU engines.

---

### [`qunitmulti.cpp`](qunitmulti.cpp) - QUnitMulti Implementation
Multi-device distribution of QUnit shards.

---

### [`qunitclifford.cpp`](qunitclifford.cpp) - QUnitClifford Implementation
Clifford-optimized QUnit.

---

### [`qtensornetwork.cpp`](qtensornetwork.cpp) - QTensorNetwork Implementation
Tensor network simulation using cuQuantum.

**Key Features:**
- Past light cone optimization
- Circuit simplification
- Sparse truncation support

---

### [`qbdt/`](qbdt/) - QBdt Implementation
Quantum Binary Decision Tree.

**Files:**
- [`node.cpp`](qbdt/node.cpp) - QBdtNode implementation
- [`node_interface.cpp`](qbdt/node_interface.cpp) - QBdtNodeInterface
- [`tree.cpp`](qbdt/tree.cpp) - Tree traversal

---

### [`qbdthybrid.cpp`](qbdthybrid.cpp) - QBdtHybrid Implementation
Hybrid QBdt/state vector simulation.

---

### [`qcircuit.cpp`](qcircuit.cpp) - QCircuit Implementation
Circuit representation for tensor network.

---

### [`qalu.cpp`](qalu.cpp) - QAlu Implementation
Arithmetic Logic Unit operations.

---

### [`qinterface_noisy.cpp`](qinterface_noisy.cpp) - QInterfaceNoisy Implementation
Noisy simulation wrapper.

---

## Common Utilities

### [`common/functions.cpp`](common/functions.cpp:1) - Core Functions (~330 lines)

**Key Functions:**
- `cl_alloc()`, `cl_free()` - Aligned memory allocation
- `intPow()`, `intPowOcl()` - Integer power
- `mul2x2()` - 2x2 complex matrix multiplication (with SIMD)
- `_expLog2x2()` - Matrix exponentiation/logarithm
- `Exp2x2()`, `Log2x2()` - Exponential/logarithm wrappers
- `GetAmplitudes()` - Extract amplitude subset
- `SetAmplitudes()` - Set amplitude subset
- `Compose()` - Compose two unitary matrices
- `Rand()` - Random number generation

**SIMD Support:**
```cpp
#if ENABLE_COMPLEX_X2
void mul2x2(const complex* left, const complex* right, complex* out)
{
    // Uses SSE2/AVX vectorized operations
}
#endif
```

---

### [`common/parallel_for.cpp`](common/parallel_for.cpp) - Parallel For Implementation
Parallel execution framework.

**Key Features:**
- Thread pool management
- Work item batching (PSTRIDEPOW)
- OpenMP and pthread support

---

### [`common/dispatchqueue.cpp`](common/dispatchqueue.cpp) - Dispatch Queue
Asynchronous task dispatching.

---

### [`common/rdrandwrapper.cpp`](common/rdrandwrapper.cpp) - RDRAND Wrapper
Hardware random number generation.

---

## GPU Implementations

### [`qengine/state.cpp`](qengine/state.cpp) - State Vector Operations
State vector manipulation (CPU).

---

### [`qengine/utility.cpp`](qengine/utility.cpp) - Utility Functions
GPU-specific utility functions.

---

### [`qengine/opencl.cpp`](qengine/opencl.cpp) - OpenCL Implementation
OpenCL kernel execution.

---

### [`qengine/cuda.cu`](qengine/cuda.cu) - CUDA Implementation
CUDA kernel execution.

---

### [`common/oclengine.cpp`](common/oclengine.cpp) - OpenCL Engine
OpenCL device management.

---

### [`common/cudaengine.cu`](common/cudaengine.cu) - CUDA Engine
CUDA device management.

---

### [`common/qengine.cl`](common/qengine.cl) - OpenCL Kernels
OpenCL kernel source code (compiled to hex at build time).

**Kernel Types:**
- Single qubit gates
- Two qubit gates
- Controlled gates
- Measurement
- State vector operations

---

## API Bindings

### [`pinvoke_api.cpp`](pinvoke_api.cpp) - P/Invoke API
Native bindings for .NET/Python.

---

### [`wasm_api.cpp`](wasm_api.cpp) - WebAssembly API
Browser-based simulation bindings.

---

### [`qrack_cl_precompile.cpp`](qrack_cl_precompile.cpp) - Kernel Precompiler
Tool to precompile OpenCL kernels.

---

## Implementation Patterns

### 1. **Lazy Evaluation**
- QUnit delays state vector materialization until necessary
- Phase shards buffer commuting operations
- Fidelity tracking enables early exit

### 2. **Cache Coherence**
- `isProbDirty` / `isPhaseDirty` flags track cache state
- `Dump()` operations invalidate caches
- `TrySeparate()` checks cache before splitting

### 3. **Parallel Execution**
- `_par_for()` for parallel loops
- `ParallelUnitApply()` for shard-level parallelism
- PSTRIDEPOW batching for reduced synchronization

### 4. **Memory Management**
- Aligned allocation for GPU compatibility
- Page-based allocation for large state vectors
- Sparse representation for polarized states

### 5. **Error Handling**
- Fidelity tracking with `logFidelity`
- ACE (Automatic Circuit Elision) for memory limits
- Exception throwing on invalid operations

---

## Next Steps
Continue reviewing examples/ and test/ directories.
