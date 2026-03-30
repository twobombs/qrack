# Memory Leak Assertion Verification

This document analyzes the claims presented in `MEMORYLEAK.md` and provides a codebase verification of each major assertion, particularly regarding memory leaks in the Qrack library.

## General Findings
Across the codebase, memory management heavily relies on RAII via C++ smart pointers (`std::shared_ptr`, `std::unique_ptr`). Most of the memory leak assertions made in the report seem to be based on an assumption of raw pointer usage which is not strictly accurate in the current state of the repository.

### 1. Phase Shard Memory Leaks (`include/qengineshard.hpp`, `src/qengineshard.cpp`, `src/qunit.cpp`)
- **Assertion:** The `QEngineShard` class maintains `std::vector<PhaseShard*> phaseShards` which stores pointers to phase correlation data, causing memory leaks because they are allocated with `new` and never freed.
- **Verification:** **FALSE**.
  - The type definition inside `qengineshard.hpp` defines `typedef std::shared_ptr<PhaseShard> PhaseShardPtr;`.
  - The maps holding the phase shards, such as `ShardToPhaseMap`, are typed as `std::map<QEngineShardPtr, PhaseShardPtr>`.
  - They are safely allocated using `std::make_shared<PhaseShard>()` in functions like `QEngineShard::AddBuffer()`.
  - Memory is safely released automatically by C++ when these structures go out of scope or elements are erased from the map.

### 2. CUDA Memory Leaks (`src/common/cudaengine.cu`, `include/qengine_cuda.hpp`)
- **Assertion:** CUDA memory allocations using `cudaMalloc` may not be properly freed, especially in error paths.
- **Verification:** **LARGELY FALSE** but with caveats.
  - The codebase utilizes `std::shared_ptr<void>` combined with a custom deleter (`[](void* c) { cudaFree(c); }`) to manage CUDA memory:
    ```cpp
    BufferPtr toRet = std::shared_ptr<void>(AllocRaw(size, &error), [](void* c) { cudaFree(c); });
    ```
  - `BufferPtr` wraps around allocations to guarantee safe deallocation and `cudaFree()` triggers whenever the shared pointer's reference count drops to zero.
  - Thus, RAII is strictly enforced for standard execution. (Note: standard CUDA runtime error checking wrappers like `tryCuda` handle error contexts).

### 3. OpenCL Memory Leaks (`src/common/oclengine.cpp`)
- **Assertion:** OpenCL memory management uses `clCreateBuffer` and `clReleaseMemObject` directly, potentially leaving unreleased memory.
- **Verification:** **FALSE**.
  - OpenCL resources are predominantly managed through the `cl::Buffer` interface (via standard C++ bindings included dynamically via `oclapi.hpp` and `<CL/cl.hpp>` or `<CL/opencl.hpp>`).
  - These wrappers implicitly manage memory by utilizing destructors that release OpenCL objects on destruction. The codebase avoids manual calls to `clCreateBuffer`.

### 4. BDT Tree Node Leaks (`src/qbdt/tree.cpp`, `src/qbdt/node.cpp`)
- **Assertion:** BDT tree nodes are allocated during tree operations but may not be properly freed, causing exponential memory growth.
- **Verification:** **FALSE**.
  - Tree nodes are actively managed via `QBdtNodeInterfacePtr`, which corresponds to a `std::shared_ptr` underneath.
  - Operations allocating nodes, such as splitting or initializing tree depth, utilize `std::make_shared<QBdtNode>(...)`.
  - There are no bare `new QBdtNode(...)` expressions susceptible to memory leaks. Branches automatically clean up memory upon destruction or reassignment.

### 5. State Vector Leaks (`src/qengine/state.cpp`, `include/qengine_cpu.hpp`)
- **Assertion:** State vector allocations are large, using raw pointers that risk leaks.
- **Verification:** **FALSE**.
  - State vectors for CPU simulation are allocated using `std::make_shared<StateVectorArray>(elemCount)` or `std::make_shared<StateVectorSparse>(elemCount)`.
  - The return type is `StateVectorPtr`.
  - Memory bounds are strictly contained and automatically de-allocated when the `QEngineCPU` object or relevant instances are destroyed.

## Conclusion
The `MEMORYLEAK.md` report accurately highlights the critical memory management boundaries inherent in writing a high-performance quantum simulator (like large state vectors, CUDA/OpenCL contexts, and tree-node management).

However, its assertions regarding existing memory leaks within Qrack are outdated or inaccurate. The Qrack repository utilizes modern C++11 RAII memory management across the entire stack (`std::shared_ptr`, `std::unique_ptr`), intrinsically negating the primary leak claims described in the report.