# Qrack Codebase Review - Examples Directory SCRATCHPAD

## Example Programs

The `examples/` directory contains demonstration programs showcasing various quantum algorithms and Qrack features.

---

### Basic Quantum Algorithms

#### [`teleport.cpp`](teleport.cpp:1) - Quantum Teleportation (~112 lines)
Demonstrates the quantum teleportation protocol.

**Key Concepts:**
- Bell pair preparation (Hadamard + CNOT)
- Alice's message preparation with arbitrary U gate
- Bell measurement (CNOT + Hadamard + measurement)
- Bob's correction based on classical message
- MWI (Many-Worlds Interpretation) unitary equivalent

**Usage:**
```cpp
QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, 3, ZERO_BCI);
qReg->H(1);           // Prepare Bell pair
qReg->CNOT(1, 2);     // Entangle
qReg->U(0, theta, phi, lambda);  // Alice's message
qReg->CNOT(0, 1);     // Alice entangles message
qReg->H(0);           // Bell measurement
bool q0 = qReg->M(0); // Measure
bool q1 = qReg->M(1);
if (q0) qReg->Z(2);   // Bob's correction
if (q1) qReg->X(2);
```

---

#### [`grovers.cpp`](grovers.cpp:1) - Grover's Search (~68 lines)
Implements Grover's search algorithm for inverting a black box function.

**Key Concepts:**
- Oracle implementation using INC/DEC/ZeroPhaseFlip
- Grover iteration (oracle + diffusion)
- Optimal iteration count (12 for 256 elements)

**Usage:**
```cpp
void Oracle(QInterfacePtr qReg) {
    qReg->DEC(TARGET_INPUT, 0, 8);
    qReg->ZeroPhaseFlip(0, 8);
    qReg->INC(TARGET_INPUT, 0, 8);
}
// Grover iteration
Oracle(qReg);
qReg->H(0, 8);
qReg->ZeroPhaseFlip(0, 8);
qReg->H(0, 8);
qReg->PhaseFlip();
```

---

#### [`shors_factoring.cpp`](shors_factoring.cpp:1) - Shor's Algorithm (~162 lines)
Integer factoring using Shor's period-finding algorithm.

**Key Concepts:**
- Period finding via quantum Fourier transform
- Modular exponentiation (POWModNOut)
- Continued fraction expansion for period extraction
- GCD-based factorization

**Usage:**
```cpp
QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, qubitCount * 2, ZERO_BCI);
QAluPtr qAlu = std::dynamic_pointer_cast<QAlu>(qReg);
qReg->H(0, qubitCount);
qAlu->POWModNOut(base, toFactor, 0, qubitCount, qubitCount);
qReg->IQFT(0, qubitCount);
bitCapInt y = qReg->MAll() & (pow2(qubitCount) - ONE_BCI);
```

---

#### [`cosmology.cpp`](cosmology.cpp) - Cosmology Simulation
Demonstrates quantum simulation for cosmological models.

---

### Quantum Machine Learning

#### [`quantum_perceptron.cpp`](quantum_perceptron.cpp:1) - Quantum Perceptron (~73 lines)
Implements a quantum neuron/perceptron for machine learning.

**Key Concepts:**
- QNeuron class for quantum neural networks
- Training via LearnPermutation
- Prediction in superposition
- Learning to recognize powers of 2

**Usage:**
```cpp
QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, ControlCount + 1, ZERO_BCI);
std::vector<bitLenInt> inputIndices(ControlCount);
QNeuronPtr qPerceptron = std::make_shared<QNeuron>(qReg, inputIndices, ControlCount);
// Training
qPerceptron->LearnPermutation(angles.get(), eta, isPowerOf2);
// Prediction
qPerceptron->Predict(angles.get());
```

---

#### [`quantum_associative_memory.cpp`](quantum_associative_memory.cpp) - Quantum Associative Memory
Associative memory using quantum states.

---

### Optimization Algorithms

#### [`grovers_lookup.cpp`](grovers_lookup.cpp) - Grover's Lookup
Grover's algorithm for database lookup.

---

#### [`ordered_list_search.cpp`](ordered_list_search.cpp) - Ordered List Search
Search algorithm for ordered lists.

---

#### [`separability.cpp`](separability.cpp) - Separability Detection
Detects separable subsystems in quantum states.

---

#### [`qbdd_separability.cpp`](qbdd_separability.cpp) - QBdt Separability
Separability detection using quantum binary decision trees.

---

### Quantum Volume & Benchmarks

#### [`quantum_volume.cpp`](quantum_volume.cpp) - Quantum Volume Benchmark
Implements the quantum volume benchmark for measuring quantum computer performance.

---

#### [`qunit_separability.cpp`](qunit_separability.cpp) - QUnit Separability
Tests QUnit's separability detection capabilities.

---

### Specialized Features

#### [`teleport.cpp`](teleport.cpp:1) - Quantum Teleportation
(See above)

---

## Common Patterns in Examples

### 1. **Factory Pattern Usage**
```cpp
QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPTIMAL, numQubits, ZERO_BCI);
```

### 2. **Engine Selection**
```cpp
#if ENABLE_OPENCL
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPENCL, numQubits, ZERO_BCI);
#else
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, numQubits, ZERO_BCI);
#endif
```

### 3. **Register Operations**
```cpp
qReg->H(0, 8);        // Hadamard on 8 qubits starting at 0
qReg->MReg(0, 8);     // Measure 8 qubits
qReg->INC(value, 0, 8); // Increment register
```

### 4. **State Preparation**
```cpp
qReg->SetPermutation(ZERO_BCI);  // Initialize to |0>
qReg->SetPermutation(initState); // Initialize to specific state
```

### 5. **Measurement & Probability**
```cpp
bool result = qReg->M(qubit);           // Single qubit measurement
bitCapInt result = qReg->MAll();        // All qubits measurement
real1 prob = qReg->Prob(qubit);         // Probability of |1>
real1 prob = qReg->ProbAll(state);      // Probability of specific state
```

---

## Example Categories

| Category | Examples |
|----------|----------|
| Basic Algorithms | teleport, grovers, shors_factoring |
| Quantum ML | quantum_perceptron, quantum_associative_memory |
| Optimization | grovers_lookup, ordered_list_search, separability |
| Benchmarks | quantum_volume, qunit_separability, qbdd_separability |
| Specialized | cosmology |

---

## Next Steps
Continue reviewing test/ and scripts/ directories.
