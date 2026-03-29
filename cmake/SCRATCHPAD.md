# Qrack Codebase Review - CMake Directory SCRATCHPAD

## CMake Configuration Modules

The `cmake/` directory contains modular CMake configuration files that control various build options and platform-specific configurations.

### Core Configuration Files

| File | Purpose |
|------|---------|
| [`CppStd.cmake`](CppStd.cmake) | Sets C++ language standard (11/14/17/20/23) |
| [`Pthread.cmake`](Pthread.cmake) | Controls pthread parallelism options |
| [`EnvVars.cmake`](EnvVars.cmake) | Environment variable support configuration |
| [`Coverage.cmake`](Coverage.cmake) | Code coverage (gcov/lcov) setup |
| [`Format.cmake`](Format.cmake) | Code formatting (clang-format) configuration |
| [`EmitLlvm.cmake`](EmitLlvm.cmake) | LLVM IR emission for cross-compilation |

### Hardware Acceleration

| File | Purpose |
|------|---------|
| [`OpenCL.cmake`](OpenCL.cmake) | OpenCL GPU support, kernel compilation, SnuCL distributed computing |
| [`CUDA.cmake`](CUDA.cmake) | NVIDIA CUDA support, GVirtus distributed CUDA |
| [`OclMemGuards.cmake`](OclMemGuards.cmake) | OpenCL memory guard configuration |

### Performance Tuning

| File | Purpose |
|------|---------|
| [`Pstridepow.cmake`](Pstridepow.cmake) | CPU work item stride power (PSTRIDEPOW) |
| [`Complex_x2.cmake`](Complex_x2.cmake) | SIMD vectorization for complex number operations |
| [`FpMath.cmake`](FpMath.cmake) | Floating-point precision configuration |
| [`SSE3.cmake`](SSE3.cmake) | SSE3 instruction set support |

### Data Type Configuration

| File | Purpose |
|------|---------|
| [`UIntPow.cmake`](UIntPow.cmake) | Qubit addressing width (UINTPOW) |
| [`QbCapPow.cmake`](QbCapPow.cmake) | Qubit capacity power (QBCAPPOW) |
| [`Pure32.cmake`](Pure32.cmake) | 32-bit compilation mode |

### Algorithm-Specific

| File | Purpose |
|------|---------|
| [`Qbdt.cmake`](Qbdt.cmake) | Quantum Binary Decision Tree support |
| [`Alu.cmake`](Alu.cmake) | Arithmetic Logic Unit API |
| [`Bcd.cmake`](Bcd.cmake) | Binary Coded Decimal support (for 6502 emulation) |
| [`RegGates.cmake`](RegGates.cmake) | Register-spanning gate operations |
| [`RotApi.cmake`](RotApi.cmake) | Rotation API methods |

### Special Features

| File | Purpose |
|------|---------|
| [`Boost.cmake`](Boost.cmake) | Boost library integration (for large qubit counts) |
| [`Random.cmake`](Random.cmake) | Random number generation configuration |
| [`VM6502Q.cmake`](VM6502Q.cmake) | MOS 6502 emulator integration |
| [`QSharp.cmake`](QSharp.cmake) | Microsoft Q# runtime support |

### Examples

| File | Purpose |
|------|---------|
| [`Examples.cmake`](Examples.cmake) | Example program compilation |

---

## Key Build Options

### CPU Configuration
- `ENABLE_PTHREAD`: Enable pthread parallelism (default: ON)
- `ENABLE_QUNIT_CPU_PARALLEL`: QUnit parallelism over CPU engine (default: ON)
- `ENABLE_QBDT_CPU_PARALLEL`: QBdt parallelism (default: OFF)
- `PSTRIDEPOW`: CPU work item stride power (default: varies by system)

### GPU Configuration
- `ENABLE_OPENCL`: OpenCL GPU support (default: ON)
- `ENABLE_CUDA`: CUDA GPU support (default: OFF, mutually exclusive with OpenCL)
- `ENABLE_OOO_OCL`: OpenCL v2.0 out-of-order queue (default: ON)
- `ENABLE_SNUCL`: Distributed OpenCL via SnuCL (default: OFF)
- `ENABLE_GVIRTUS`: Distributed CUDA via GVirtus (default: OFF)

### Precision & Data Types
- `FPPOW`: Floating-point precision (2^n bits, default: 5 = 32-bit float)
- `UINTPOW`: Local qubit addressing width (2^n bits, default: 6)
- `QBCAPPOW`: Global qubit capacity power (default: 6)
- `ENABLE_COMPLEX_X2`: SIMD complex multiplication (default: ON)

### API Components
- `ENABLE_ALU`: Arithmetic Logic Unit API (default: ON)
- `ENABLE_BCD`: Binary Coded Decimal (default: OFF)
- `ENABLE_REG_GATES`: Register-spanning gates (default: OFF)
- `ENABLE_ROT_API`: Rotation API (default: OFF)

### Platform-Specific
- `EMSCRIPTEN`: WebAssembly build target
- `MSVC`: Microsoft Visual C++ compiler
- `PACK_DEBIAN`: Debian packaging mode

---

## Build Process Flow

1. **Initial Setup**: CMakeLists.txt sets project version (10.5.3), includes GNUInstallDirs
2. **Core Options**: ENABLE_EXAMPLES, ENABLE_TESTS, ENABLE_INTRINSICS
3. **Library Declaration**: Static library `qrack` with source files
4. **Platform Detection**: CPU architecture, compiler type (MSVC/GCC/Clang)
5. **Include Directories**: `include/`, `include/common/`, build output directory
6. **Modular Configuration**: Each cmake/*.cmake file configures specific feature
7. **Compiler Flags**: Optimization flags based on platform and features
8. **Dependency Linking**: OpenCL, CUDA, pthread, quadmath (if needed)
9. **OpenCL/CUDA Kernels**: Precompile `.cl` files to header format using `xxd`
10. **Target Compilation**: Apply flags to qrack, qrack_pinvoke, qrack_wasm targets
11. **Installation Rules**: Install headers, libraries, pkg-config files
12. **Packaging**: CPack configuration for DEB/TGZ packages

---

## Notable Build Artifacts

- **qrack**: Main static library
- **qrack_pinvoke**: Shared library for P/Invoke bindings (Q#, Python, etc.)
- **qrack_wasm**: WebAssembly library for browser-based simulation
- **qrack_cl_precompile**: Tool to precompile OpenCL kernels
- **unittest**: Test executable (if ENABLE_TESTS)
- **benchmarks**: Benchmark executable (if ENABLE_TESTS)

---

## Next Steps
Continue reviewing include/ directory for header file structure.
