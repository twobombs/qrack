# Qrack Codebase Review - Test Directory SCRATCHPAD

## Test Infrastructure

### [`catch.hpp`](catch.hpp:1) - Catch2 Testing Framework
Qrack uses Catch v2.13.7 for unit testing (Boost Software License).

**Key Features:**
- Test case organization with `TEST_CASE()`
- Assertion macros (`REQUIRE()`, `CHECK()`)
- Floating-point comparison with epsilon
- Complex number comparison

---

### [`tests.cpp`](tests.cpp:1) - Main Test Suite (~7200 lines)
Comprehensive unit tests for all Qrack functionality.

**Test Categories:**

#### 1. **Basic Operations**
- `test_complex` - Complex number arithmetic
- `test_push_apart_bits` - Bit manipulation utilities
- `test_gate` - Single and multi-qubit gates
- `test_phase` - Phase gate operations

#### 2. **Gate Tests**
- `test_h` - Hadamard gate
- `test_x` - Pauli-X gate
- `test_y` - Pauli-Y gate
- `test_z` - Pauli-Z gate
- `test_s` - S gate
- `test_t` - T gate
- `test_cnot` - CNOT gate
- `test_ccnot` - Toffoli gate
- `test_swap` - SWAP gate
- `test_cswap` - Controlled SWAP

#### 3. **Controlled Gates**
- `test_controlled` - Various controlled gate tests
- `test_multi_control` - Multi-controlled gates
- `test_uniformly_controlled` - Uniformly controlled gates

#### 4. **Arithmetic Operations**
- `test_alu` - ALU operations
- `test_inc` - Increment
- `test_dec` - Decrement
- `test_mul` - Multiplication
- `test_div` - Division
- `test_and` - AND operation
- `test_or` - OR operation
- `test_xor` - XOR operation
- `test_cmp` - Comparison

#### 5. **Measurement & Probability**
- `test_prob` - Probability calculations
- `test_sample` - State sampling
- `test_m` - Measurement operations
- `test_m_all` - All-qubit measurement

#### 6. **State Vector Operations**
- `test_state` - State vector manipulation
- `test_compose` - State composition
- `test_dispose` - State disposal
- `test_reshape` - State reshaping

#### 7. **Advanced Features**
- `test_qft` - Quantum Fourier Transform
- `test_iqft` - Inverse QFT
- `test_entangle` - Entanglement operations
- `test_separate` - Separation operations
- `test_try_separate` - Try separation

#### 8. **Engine-Specific Tests**
- Tests for QEngineCPU
- Tests for QEngineOCL (if OpenCL enabled)
- Tests for QEngineCUDA (if CUDA enabled)
- Tests for QPager
- Tests for QUnit
- Tests for QStabilizer
- Tests for QStabilizerHybrid

#### 9. **Edge Cases**
- `test_zero_qubit` - Zero qubit edge case
- `test_large_state` - Large state vector tests
- `test_random` - Random operation tests

---

### [`benchmarks.cpp`](benchmarks.cpp:1) - Performance Benchmarks
Performance testing for various algorithms and configurations.

**Benchmark Categories:**

#### 1. **Algorithm Benchmarks**
- `test_qft` - Quantum Fourier Transform
- `test_grover` - Grover's search
- `test_shor` - Shor's factoring
- `test_teleport` - Quantum teleportation
- `test_cosmology` - Cosmology simulation

#### 2. **Engine Benchmarks**
- CPU engine performance
- OpenCL GPU performance
- CUDA GPU performance
- Hybrid engine performance

#### 3. **Scaling Tests**
- Qubit count scaling
- Circuit depth scaling
- Memory usage scaling

**Usage:**
```bash
./benchmarks [--optimal] [--max-qubits=30] [test_name]
```

---

### [`benchmarks_main.cpp`](benchmarks_main.cpp) - Benchmark Entry Point
Main function for benchmark executable.

---

## Test Configuration

### Global Test Variables
```cpp
QInterfaceEngine testEngineType;
QInterfaceEngine testSubEngineType;
QInterfaceEngine testSubSubEngineType;
QInterfaceEngine testSubSubSubEngineType;
int device_id;
bool enable_normalization;
bool disable_hardware_rng;
bool sparse;
std::vector<int64_t> devList;
bool disable_t_injection;
bool disable_reactive_separation;
```

### Test Helpers
```cpp
QInterfacePtr MakeEngine(bitLenInt qubitCount);
void print_bin(int bits, int d);
void log(QInterfacePtr p);
```

### Assertion Macros
```cpp
#define REQUIRE_FLOAT(A, B)  // Floating-point comparison with epsilon
#define REQUIRE_CMPLX(A, B)  // Complex number comparison with epsilon
```

---

## Test Patterns

### 1. **Engine Creation**
```cpp
QInterfacePtr qReg = MakeEngine(qubitCount);
```

### 2. **State Preparation**
```cpp
qReg->SetPermutation(ZERO_BCI);
qReg->H(0);
```

### 3. **Verification**
```cpp
REQUIRE_FLOAT(qReg->Prob(0), expected_prob);
REQUIRE(qReg->M(0) == expected_result);
```

### 4. **Comparison with Reference**
```cpp
QInterfacePtr reference = CreateQuantumInterface(QINTERFACE_CPU, qubitCount, ZERO_BCI);
// ... perform same operations ...
REQUIRE_FLOAT(qReg->ProbAll(state), reference->ProbAll(state));
```

---

## Test Coverage

The test suite covers:
- All basic quantum gates
- All arithmetic operations
- All measurement operations
- State vector manipulation
- Entanglement and separation
- Multi-qubit operations
- Engine-specific features
- Edge cases and error conditions

---

## Next Steps
Continue reviewing scripts/ directory.
