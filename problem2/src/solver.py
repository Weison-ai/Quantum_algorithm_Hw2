"""Problem 2: Max-Cut with QAOA.

This module provides the reusable helpers for the assignment workflow:
- graph generation from a student-specific seed
- exact brute-force maximum cut
- QAOA objective evaluation and optimization
- simulated annealing comparison
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import dimod
import networkx as nx
import neal
import numpy as np
import pennylane as qml

DEFAULT_GRAPH_NODES = 8
DEFAULT_EDGE_PROBABILITY = 0.5
DEFAULT_QAOA_GRID_POINTS = 50


@dataclass(frozen=True)
class CutResult:
    """Container for an exact cut solution."""

    cut_value: int
    partitions: List[Tuple[int, ...]]
    bitstrings: List[Tuple[int, ...]]


def generate_student_graph(
    seed: int,
    n: int = DEFAULT_GRAPH_NODES,
    p: float = DEFAULT_EDGE_PROBABILITY,
) -> nx.Graph:
    """Generate the student-specific Erdős-Rényi graph."""

    return nx.gnp_random_graph(n=n, p=p, seed=seed)


def graph_edges(G: nx.Graph) -> List[Tuple[int, int]]:
    """Return graph edges as a sorted list."""

    return sorted((min(u, v), max(u, v)) for u, v in G.edges())


def partition_from_bitstring(bitstring: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split vertices into two partitions using a binary bitstring."""

    left = tuple(index for index, bit in enumerate(bitstring) if int(bit) == 0)
    right = tuple(index for index, bit in enumerate(bitstring) if int(bit) == 1)
    return left, right


def cut_value(G: nx.Graph, bitstring: Sequence[int]) -> int:
    """Compute the number of edges crossing a bipartition."""

    crossing = 0
    for u, v in G.edges():
        if int(bitstring[u]) != int(bitstring[v]):
            crossing += 1
    return crossing


def brute_force_max_cut(G: nx.Graph) -> CutResult:
    """Enumerate all partitions and return the exact maximum cut."""

    n = G.number_of_nodes()
    best_cut = -math.inf
    best_bitstrings: List[Tuple[int, ...]] = []
    best_partitions: List[Tuple[int, ...]] = []

    for bits in itertools.product((0, 1), repeat=n):
        current_cut = cut_value(G, bits)
        if current_cut > best_cut:
            best_cut = current_cut
            best_bitstrings = [tuple(bits)]
            best_partitions = [partition_from_bitstring(bits)[0]]
        elif current_cut == best_cut:
            best_bitstrings.append(tuple(bits))
            best_partitions.append(partition_from_bitstring(bits)[0])

    return CutResult(int(best_cut), best_partitions, best_bitstrings)


def brute_force_min_cut(G: nx.Graph) -> CutResult:
    """Backward-compatible alias. Returns the exact maximum cut."""

    return brute_force_max_cut(G)


def _edge_terms(G: nx.Graph) -> List[Tuple[float, str]]:
    """Create Pauli-Z terms for the Max-Cut optimization Hamiltonian."""

    terms: List[Tuple[float, str]] = []
    for u, v in graph_edges(G):
        terms.append((0.5, f"Z{u} @ Z{v}"))
    return terms


def build_cost_hamiltonian(G: nx.Graph) -> Tuple[qml.Hamiltonian, float]:
    """Build the Max-Cut optimization Hamiltonian used by QAOA.

    This Hamiltonian corresponds to ``-cut`` up to a constant shift.
    Minimizing it is equivalent to maximizing the cut value.
    """

    coeffs: List[float] = []
    ops: List[qml.operation.Operator] = []
    for u, v in graph_edges(G):
        coeffs.append(0.5)
        ops.append(qml.PauliZ(u) @ qml.PauliZ(v))
    constant_shift = -0.5 * G.number_of_edges()
    return qml.Hamiltonian(coeffs, ops), constant_shift


def _apply_cost_layer(gamma: float, G: nx.Graph) -> None:
    """Apply one cost layer for the Max-Cut optimization Hamiltonian."""

    for u, v in graph_edges(G):
        qml.IsingZZ(gamma, wires=[u, v])


def _apply_mixer_layer(beta: float, n_qubits: int) -> None:
    """Apply the standard X mixer."""

    for wire in range(n_qubits):
        qml.RX(2.0 * beta, wires=wire)


def _qaoa_circuit(cost_params: np.ndarray, mixer_params: np.ndarray, G: nx.Graph):
    """Create the QAOA expectation circuit for a given depth."""

    n_qubits = G.number_of_nodes()
    cost_hamiltonian, constant_shift = build_cost_hamiltonian(G)
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="autograd")
    def circuit(gammas, betas):
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)

        for layer in range(len(gammas)):
            _apply_cost_layer(gammas[layer], G)
            _apply_mixer_layer(betas[layer], n_qubits)

        return qml.expval(cost_hamiltonian), qml.expval(qml.Identity(0))

    return circuit, constant_shift


def qaoa_expectation(G: nx.Graph, gammas: Sequence[float], betas: Sequence[float]) -> float:
    """Evaluate the QAOA objective for the Max-Cut optimization Hamiltonian."""

    gammas_array = np.asarray(gammas, dtype=float)
    betas_array = np.asarray(betas, dtype=float)
    circuit, _ = _qaoa_circuit(gammas_array, betas_array, G)
    expectation, _ = circuit(gammas_array, betas_array)
    return float(expectation)


def evaluate_qaoa_landscape(
    G: nx.Graph,
    gamma_points: int = DEFAULT_QAOA_GRID_POINTS,
    beta_points: int = DEFAULT_QAOA_GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the p=1 QAOA landscape over a regular grid."""

    gammas = np.linspace(0.0, 2.0 * np.pi, gamma_points)
    betas = np.linspace(0.0, np.pi, beta_points)
    landscape = np.zeros((gamma_points, beta_points), dtype=float)

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            landscape[i, j] = qaoa_expectation(G, [gamma], [beta])

    return gammas, betas, landscape


def optimize_qaoa(
    G: nx.Graph,
    p: int,
    seed: int,
    steps: int = 120,
    learning_rate: float = 0.15,
) -> Dict[str, object]:
    """Optimize QAOA parameters with Adam and return the best solution."""

    rng = np.random.default_rng(seed)
    gammas = qml.numpy.array(rng.uniform(0.0, 2.0 * np.pi, size=p), requires_grad=True)
    betas = qml.numpy.array(rng.uniform(0.0, np.pi, size=p), requires_grad=True)

    dev = qml.device("default.qubit", wires=G.number_of_nodes())
    cost_hamiltonian, constant_shift = build_cost_hamiltonian(G)

    @qml.qnode(dev, interface="autograd")
    def circuit(gamma_params, beta_params):
        for wire in range(G.number_of_nodes()):
            qml.Hadamard(wires=wire)

        for layer in range(p):
            for u, v in graph_edges(G):
                qml.IsingZZ(gamma_params[layer], wires=[u, v])
            for wire in range(G.number_of_nodes()):
                qml.RX(2.0 * beta_params[layer], wires=wire)

        return qml.expval(cost_hamiltonian)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    best_params = (gammas, betas)
    best_energy = float(circuit(gammas, betas))

    for _ in range(steps):
        gammas, betas = optimizer.step(circuit, gammas, betas)
        energy = float(circuit(gammas, betas))
        if energy < best_energy:
            best_energy = energy
            best_params = (qml.numpy.array(gammas, requires_grad=False), qml.numpy.array(betas, requires_grad=False))

    bitstring, cut = sample_bitstring_from_qaoa(G, best_params[0], best_params[1])
    return {
        "p": p,
        "gammas": np.asarray(best_params[0], dtype=float).tolist(),
        "betas": np.asarray(best_params[1], dtype=float).tolist(),
        "objective": best_energy,
        "cut": cut,
        "bitstring": bitstring,
        "constant_shift": constant_shift,
    }


def sample_bitstring_from_qaoa(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> Tuple[Tuple[int, ...], int]:
    """Sample the most likely bitstring from the optimized QAOA circuit."""

    n_qubits = G.number_of_nodes()
    dev = qml.device("default.qubit", wires=n_qubits, shots=1000)

    @qml.qnode(dev, interface="autograd")
    def circuit(gamma_params, beta_params):
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        for layer in range(len(gamma_params)):
            for u, v in graph_edges(G):
                qml.IsingZZ(gamma_params[layer], wires=[u, v])
            for wire in range(n_qubits):
                qml.RX(2.0 * beta_params[layer], wires=wire)
        return qml.sample(wires=range(n_qubits))

    samples = circuit(np.asarray(gammas, dtype=float), np.asarray(betas, dtype=float))
    if samples.ndim == 1:
        sample = tuple(int(v) for v in samples.tolist())
    else:
        counts = {}
        for row in samples:
            key = tuple(int(v) for v in row.tolist())
            counts[key] = counts.get(key, 0) + 1
        sample = max(counts.items(), key=lambda item: item[1])[0]

    return sample, cut_value(G, sample)


def sample_simulated_annealing(
    G: nx.Graph,
    seed: int,
    num_reads: int = 1000,
) -> Dict[str, object]:
    """Solve Max-Cut approximately with simulated annealing."""

    linear = {node: 0.0 for node in G.nodes()}
    quadratic = {}
    for u, v in graph_edges(G):
        linear[u] -= 1.0
        linear[v] -= 1.0
        quadratic[(u, v)] = quadratic.get((u, v), 0.0) + 2.0

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    sampler = neal.SimulatedAnnealingSampler()
    start = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    elapsed = time.perf_counter() - start

    best = sampleset.first.sample
    bitstring = tuple(int(best[node]) for node in range(G.number_of_nodes()))
    return {
        "bitstring": bitstring,
        "cut": cut_value(G, bitstring),
        "objective": float(sampleset.first.energy),
        "time": elapsed,
        "num_reads": num_reads,
    }


def approximation_ratio(found_cut: int, optimal_cut: int) -> float:
    """Compute the approximation ratio used in the report."""

    if optimal_cut == 0:
        return 1.0 if found_cut == 0 else 0.0
    return float(found_cut) / float(optimal_cut)


def summarize_optimal_partitions(result: CutResult) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Convert stored partitions into both sides of the cut."""

    summaries = []
    for bitstring in result.bitstrings:
        summaries.append(partition_from_bitstring(bitstring))
    return summaries
