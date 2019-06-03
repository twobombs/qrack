# This file is adapted from https://github.com/QCGPU/qcgpu/blob/master/benchmark/benchmark.py from the QCGPU project, by Adam Kelly
# Source for circuit definition of QFT is https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation

import click
import time
import random
import statistics
import csv
import os.path
import math
import random

#from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
#from qiskit.wrapper import load_qasm_file
#from qiskit import QISKitError, execute, Aer

from projectq import MainEngine
import projectq.ops as ops
from projectq.backends import Simulator

import qcgpu

# Implementation of the Quantum Fourier Transform
#def construct_circuit(num_qubits):
#    q = QuantumRegister(num_qubits)
#    circ = QuantumCircuit(q)
#
#    # Quantum Fourier Transform
#    for j in range(num_qubits):
#        for k in range(j):
#            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
#        circ.h(q[j])
#
#    return circ


# Benchmarking functions
#qiskit_backend = Aer.get_backend('statevector_simulator')
#eng = MainEngine(backend=Simulator(), engine_list=[])

# Setup the OpenCL Device
qcgpu.backend.create_context()

#def bench_qiskit(qc):
#    start = time.time()
#    job_sim = execute(qc, qiskit_backend)
#    sim_result = job_sim.result()
#    return time.time() - start

def bench_qcgpu(num_qubits):
    state = qcgpu.State(num_qubits)
    start = time.time()
    # These gates were changed from the original file,
    # to match those used by Qrack for the QFT
    # https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation
    for j in range(num_qubits):
        state.h(j)
        for k in range(1, (num_qubits - j)):
            state.cu1(j + k, j, math.pi/float(2**(k)))

    state.backend.queue.finish()
    return time.time() - start

def bench_projectq(num_qubits):
    q = eng.allocate_qureg(num_qubits)
    start = time.time()
    # These gates were changed from the original file,
    # to match those used by Qrack for the QFT
    # https://en.wikipedia.org/wiki/Quantum_Fourier_transform#Circuit_implementation
    for j in range(num_qubits):
        ops.H | q[j]
        for k in range(1, (num_qubits - j)):
            ops.CRz(math.pi/float(2**(k))) | (q[j + k], q[j]) 
    eng.flush()

    t = time.time() - start

    # measure to get rid of runtime error message
    for j in q:
        ops.Measure | j

    return t
    

# Reporting
def create_csv(filename):
    file_exists = os.path.isfile(filename)
    csvfile = open(filename, 'a')
   
    headers = ['name', 'num_qubits', 'time']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header

    return writer

def write_csv(writer, data):
    writer.writerow(data)



@click.command()
@click.option('--samples', default=100, help='Number of samples to take for each qubit.')
@click.option('--qubits', default=28, help='How many qubits you want to test for')
@click.option('--out', default='benchmark_data.csv', help='Where to store the CSV output of each test')
@click.option('--single', default=False, help='Only run the benchmark for a single amount of qubits, and print an analysis')
def benchmark(samples, qubits, out, single):
    if single:
        # QCGPU and ProjectQ were tested in two separate runs of this file.
        # functions = bench_qcgpu, bench_projectq
        functions = bench_qcgpu, 
        times = {f.__name__: [] for f in functions}

        names = []
        means = []

        #qc = construct_circuit(qubits)
        # Run the benchmarks
        for i in range(samples):
            progress = (i) / (samples)
            if samples > 1:
                print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress*100), end="", flush=True)

            func = random.choice(functions)
            if func.__name__ != 'bench_qiskit':
                t = func(qubits)
            else:
                t = func(qc)
            times[func.__name__].append(t)

        print('')

        for name, numbers in times.items():
            print('FUNCTION:', name, 'Used', len(numbers), 'times')
            print('\tMEDIAN', statistics.median(numbers))
            print('\tMEAN  ', statistics.mean(numbers))
            if len(numbers) > 1:
                print('\tSTDEV ', statistics.stdev(numbers))

        return

    
    # functions = bench_qcgpu, bench_projectq
    functions = bench_qcgpu,
    # times = {f.__name__: [] for f in functions}
    writer = create_csv(out)

    for n in range(3, qubits):
        # Progress counter
        progress = (n+1) / (qubits)
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress*100), end="", flush=True)

        # Construct the circuit
        #qc = construct_circuit(n+1)

        # Run the benchmarks
        for i in range(samples):
            func = random.choice(functions)
            if func.__name__ != 'bench_qiskit':
                t = func(n + 1)
            else:
                t = func(qc)
            # times[func.__name__].append(t)
            write_csv(writer, {'name': func.__name__, 'num_qubits': n+1, 'time': t})

if __name__ == '__main__':
    benchmark()
