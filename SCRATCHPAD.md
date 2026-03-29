# Memory Leak Suspect Report

## Suspect Location
**File**: `src/qengine/arithmetic.cpp`
**Function**: `QEngineCPU::INCDECBCDC` (Binary Coded Decimal Arithmetic with Carry)
**Line Range**: 926 - 977 (specifically, line 931)

## Issue Description
Within the `QEngineCPU::INCDECBCDC` function, there is a tight parallel loop implemented via `par_for_skip`.

```cpp
    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryIndex), 1U, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        ...
        int* nibbles = new int[nibbleCount]; // LINE 931
        ...
        delete[] nibbles; // LINE 976
    });
```

During each iteration of this `par_for_skip` loop, the lambda function explicitly allocates a raw integer array using `new int[nibbleCount];`. Although there is a corresponding `delete[] nibbles;` at the end of the lambda, this code pattern is highly problematic and can cause a memory leak:

1. **Exception Safety / Leak Scenario**: If any operation within this lambda throws an exception (e.g., during `nStateVec->write`), the `delete[] nibbles;` statement at the end of the lambda will be skipped. Because `nibbles` is a raw pointer and lacks RAII (Resource Acquisition Is Initialization) semantics like `std::unique_ptr` or `std::vector`, the allocated array will leak permanently.

2. **Severe Heap Contention and Fragmentation**: The `par_for_skip` loop executes over `maxQPowerOcl` iterations (which scales exponentially with the number of qubits, frequently resulting in millions of concurrent iterations). Allocating and deallocating memory inside such a fine-grained, highly parallel loop exerts extreme pressure on the heap allocator. This leads to massive heap fragmentation, severe performance degradation, and peak memory usage that resembles a large leak, eventually triggering a failure if memory exhaustion occurs.

## Recommended Fix (Implemented)

The solution is to avoid performing memory allocations within the hot path of the parallel loop. Instead, thread-local buffers should be pre-allocated once per thread before the loop begins.

**Resolution:**
1. Determine the number of executing threads using `GetConcurrencyLevel()`.
2. Allocate a `std::vector` containing one `std::unique_ptr<int[]>` for each thread outside of the `par_for_skip` loop.
3. Inside the loop, replace the `new` allocation with a simple pointer retrieval from the pre-allocated vector using the `cpu` thread ID: `int* nibbles = nibblesVec[cpu].get();`.
4. Remove the `delete[] nibbles;` from the lambda.

This mirrors the robust memory handling logic used by the neighboring function, `QEngineCPU::INCBCD`, entirely eliminating the memory leak risk and significantly improving execution performance.
