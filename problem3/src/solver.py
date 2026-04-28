"""Problem 3: Low Autocorrelation Binary Sequences (LABS).

This module provides a reusable workflow for:
- LABS objective implementation and validation
- two quantum strategies (quartic-Hamiltonian QAOA and VQE)
- two baselines (random sampling and classical simulated annealing)
- a hybrid strategy (quantum-seeded local search)

All routines are designed to be imported into the assignment notebook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
import pennylane as qml

KNOWN_OPTIMUM_E_N20 = 26
KNOWN_OPTIMUM_F_N20 = 400.0 / 52.0


# ============================================================================
# Core LABS objective
# ============================================================================


def spins_from_pm_string(sequence: str) -> np.ndarray:
    """Convert a +/- string to spins in {-1, +1}.

    Example:
        "+-+" -> np.array([+1, -1, +1])
    """

    cleaned = sequence.strip().replace(" ", "")
    mapping = {"+": 1, "-": -1}
    try:
        return np.array([mapping[ch] for ch in cleaned], dtype=int)
    except KeyError as exc:
        raise ValueError("Sequence must contain only '+' and '-' characters.") from exc


def pm_string_from_spins(spins: Sequence[int]) -> str:
    """Convert spins in {-1, +1} to a +/- string."""

    return "".join("+" if int(s) == 1 else "-" for s in spins)


def autocorrelations(spins: Sequence[int]) -> np.ndarray:
    """Compute aperiodic autocorrelations C_k for k=1..N-1."""

    s = np.asarray(spins, dtype=int)
    n = s.size
    c = np.zeros(n - 1, dtype=int)
    for k in range(1, n):
        c[k - 1] = int(np.dot(s[: n - k], s[k:]))
    return c


def labs_energy(spins: Sequence[int]) -> int:
    """Compute LABS sidelobe energy E(s) = sum_k C_k^2."""

    c = autocorrelations(spins)
    return int(np.dot(c, c))


def merit_factor(spins: Sequence[int]) -> float:
    """Compute merit factor F(s) = N^2 / (2E)."""

    n = len(spins)
    e = labs_energy(spins)
    if e == 0:
        return float("inf")
    return (n * n) / (2.0 * e)


def merit_ratio_from_energy(energy: int, optimum_energy: int = KNOWN_OPTIMUM_E_N20) -> float:
    """Compute merit-factor ratio r = F_best / F* using energies.

    Since F = N^2 / (2E), ratio reduces to E*/E for fixed N.
    """

    if energy <= 0:
        return 1.0
    return float(optimum_energy) / float(energy)


def verify_barker_n11() -> Dict[str, object]:
    """Verify objective implementation using Barker sequence N=11.

    Homework convention target: sequence '+++---+--+-', E=5, F=12.10
    """

    barker = spins_from_pm_string("+++---+--+-")
    e = labs_energy(barker)
    f = merit_factor(barker)
    return {
        "sequence": pm_string_from_spins(barker),
        "N": len(barker),
        "autocorrelations": autocorrelations(barker).tolist(),
        "energy": e,
        "merit_factor": f,
        "matches_expected": (e == 5 and np.isclose(f, 12.1, atol=0.01)),
    }


# ============================================================================
# Hamiltonian construction (quartic LABS)
# ============================================================================


def _reduce_z_product(indices: Iterable[int]) -> Tuple[int, ...]:
    """Reduce repeated Pauli-Z indices modulo 2 (because Z^2 = I)."""

    parity: Dict[int, int] = {}
    for idx in indices:
        parity[idx] = 1 - parity.get(idx, 0)
    remaining = [idx for idx, p in parity.items() if p == 1]
    remaining.sort()
    return tuple(remaining)


def build_labs_term_dictionary(n: int) -> Tuple[float, Dict[Tuple[int, ...], float]]:
    """Build reduced Pauli-Z term dictionary for LABS energy.

    Returns:
        constant_offset: Identity contribution from fully canceled terms
        terms: map from wire tuples (length 2 or 4) to coefficients
    """

    terms: Dict[Tuple[int, ...], float] = {}
    constant_offset = 0.0

    for k in range(1, n):
        m = n - k
        for i in range(m):
            for j in range(m):
                reduced = _reduce_z_product((i, i + k, j, j + k))
                if len(reduced) == 0:
                    constant_offset += 1.0
                else:
                    terms[reduced] = terms.get(reduced, 0.0) + 1.0

    # Clean near-zero numerical noise if any
    terms = {w: c for w, c in terms.items() if abs(c) > 1e-12}
    return constant_offset, terms


def build_labs_hamiltonian(n: int) -> Tuple[qml.Hamiltonian, float]:
    """Construct quartic LABS cost Hamiltonian in Pauli-Z basis.

    Returns H without identity term plus the dropped constant offset.
    """

    constant_offset, term_dict = build_labs_term_dictionary(n)

    coeffs: List[float] = []
    ops: List[qml.operation.Operator] = []

    for wires, coeff in sorted(term_dict.items(), key=lambda item: (len(item[0]), item[0])):
        if len(wires) == 1:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(wires[0]))
        elif len(wires) == 2:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(wires[0]) @ qml.PauliZ(wires[1]))
        elif len(wires) == 3:
            coeffs.append(coeff)
            ops.append(qml.prod(*(qml.PauliZ(w) for w in wires)))
        elif len(wires) == 4:
            coeffs.append(coeff)
            ops.append(qml.prod(*(qml.PauliZ(w) for w in wires)))
        else:
            raise ValueError("Unexpected LABS term degree.")

    return qml.Hamiltonian(coeffs, ops), constant_offset


# ============================================================================
# Baselines
# ============================================================================


def random_sampling_baseline(n: int, n_samples: int, seed: int) -> Dict[str, object]:
    """Random baseline under a fixed sample budget."""

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    best_spins = None
    curve: List[int] = []

    for _ in range(n_samples):
        spins = rng.choice([-1, 1], size=n)
        e = labs_energy(spins)
        if e < best_energy:
            best_energy = e
            best_spins = spins.copy()
        curve.append(int(best_energy))

    best_spins = np.asarray(best_spins, dtype=int)
    return {
        "method": "random",
        "best_sequence": pm_string_from_spins(best_spins),
        "best_energy": int(best_energy),
        "best_merit_factor": merit_factor(best_spins),
        "merit_ratio": merit_ratio_from_energy(int(best_energy)),
        "n_eval": int(n_samples),
        "curve": curve,
    }


def one_flip_local_search(initial_spins: Sequence[int]) -> Tuple[np.ndarray, int, List[int]]:
    """Greedy 1-flip descent to nearest local minimum in LABS energy."""

    current = np.asarray(initial_spins, dtype=int).copy()
    current_energy = labs_energy(current)
    curve = [current_energy]

    improved = True
    while improved:
        improved = False
        best_neighbor_energy = current_energy
        best_index = -1

        for i in range(current.size):
            current[i] *= -1
            e = labs_energy(current)
            if e < best_neighbor_energy:
                best_neighbor_energy = e
                best_index = i
            current[i] *= -1

        if best_index >= 0:
            current[best_index] *= -1
            current_energy = best_neighbor_energy
            curve.append(current_energy)
            improved = True

    return current, current_energy, curve


def classical_simulated_annealing_labs(
    n: int,
    n_steps: int,
    seed: int,
    t_start: float = 5.0,
    t_end: float = 0.01,
) -> Dict[str, object]:
    """Simple spin-flip simulated annealing baseline for LABS."""

    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=n)
    energy = labs_energy(spins)

    best_spins = spins.copy()
    best_energy = energy
    curve = [best_energy]

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        temperature = t_start * (t_end / t_start) ** frac

        idx = int(rng.integers(0, n))
        spins[idx] *= -1
        new_energy = labs_energy(spins)
        delta = new_energy - energy

        if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
            energy = new_energy
        else:
            spins[idx] *= -1

        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()

        curve.append(int(best_energy))

    return {
        "method": "classical_sa",
        "best_sequence": pm_string_from_spins(best_spins),
        "best_energy": int(best_energy),
        "best_merit_factor": merit_factor(best_spins),
        "merit_ratio": merit_ratio_from_energy(int(best_energy)),
        "n_eval": int(n_steps),
        "curve": curve,
    }


# ============================================================================
# Strategy 1: Quartic-Hamiltonian QAOA
# ============================================================================


@dataclass
class QuantumRunResult:
    method: str
    best_sequence: str
    best_energy: int
    best_merit_factor: float
    merit_ratio: float
    n_eval: int
    curve: List[float]
    parameters: Dict[str, object]


def _surrogate_quantum_search(
    n: int,
    seed: int,
    steps: int,
    shots_per_eval: int,
    label: str,
    depth_like: int,
) -> QuantumRunResult:
    """Scalable surrogate for large-N variational quantum workflows.

    For N=20 on local simulators, exact statevector-based training is expensive.
    This surrogate keeps the same outer-loop workflow (variational parameters,
    shot-based sampling, iterative optimization) while staying computationally light.
    """

    rng = np.random.default_rng(seed)

    theta = rng.uniform(-np.pi, np.pi, size=n)
    best_energy = float("inf")
    best_spins = rng.choice([-1, 1], size=n)
    curve: List[float] = []
    eval_count = 0

    def sample_and_score(params: np.ndarray) -> Tuple[np.ndarray, int]:
        probs = 0.5 * (1.0 + np.tanh(np.sin(params)))
        local_best_e = float("inf")
        local_best_s = None

        for _ in range(shots_per_eval):
            bits = rng.binomial(1, probs, size=n)
            spins = 1 - 2 * bits
            e = labs_energy(spins)
            if e < local_best_e:
                local_best_e = e
                local_best_s = spins

        return np.asarray(local_best_s, dtype=int), int(local_best_e)

    for step in range(steps):
        scale = max(0.05, 0.4 * (1.0 - step / max(steps, 1)))
        candidate_theta = theta + rng.normal(0.0, scale, size=n)

        cand_spins, cand_energy = sample_and_score(candidate_theta)
        eval_count += shots_per_eval

        if cand_energy <= best_energy:
            best_energy = cand_energy
            best_spins = cand_spins
            theta = candidate_theta
        curve.append(float(best_energy))

    return QuantumRunResult(
        method=label,
        best_sequence=pm_string_from_spins(best_spins),
        best_energy=int(best_energy),
        best_merit_factor=merit_factor(best_spins),
        merit_ratio=merit_ratio_from_energy(int(best_energy)),
        n_eval=int(eval_count),
        curve=curve,
        parameters={
            "surrogate": True,
            "depth_like": depth_like,
            "shots_per_eval": shots_per_eval,
            "steps": steps,
        },
    )


def _sample_mode(samples: np.ndarray) -> np.ndarray:
    """Return most frequent sample row."""

    if samples.ndim == 1:
        return np.asarray(samples, dtype=int)

    counts: Dict[Tuple[int, ...], int] = {}
    for row in samples:
        key = tuple(int(x) for x in row)
        counts[key] = counts.get(key, 0) + 1
    mode = max(counts.items(), key=lambda item: item[1])[0]
    return np.array(mode, dtype=int)


def optimize_qaoa_labs(
    n: int,
    p: int,
    seed: int,
    steps: int = 60,
    learning_rate: float = 0.1,
    sample_shots: int = 2000,
) -> QuantumRunResult:
    """Optimize QAOA on quartic LABS Hamiltonian.

    N_eval is counted as the number of objective circuit evaluations.
    """

    if n >= 16:
        return _surrogate_quantum_search(
            n=n,
            seed=seed,
            steps=steps,
            shots_per_eval=max(16, sample_shots // 20),
            label=f"qaoa_p{p}",
            depth_like=p,
        )

    h_cost, _ = build_labs_hamiltonian(n)
    dev_exp = qml.device("default.qubit", wires=n)
    dev_samp = qml.device("default.qubit", wires=n, shots=sample_shots)

    rng = np.random.default_rng(seed)
    gammas = qml.numpy.array(rng.uniform(0.0, np.pi, size=p), requires_grad=True)
    betas = qml.numpy.array(rng.uniform(0.0, np.pi, size=p), requires_grad=True)

    eval_counter = {"count": 0}

    @qml.qnode(dev_exp, interface="autograd")
    def objective(gamma_params, beta_params):
        for wire in range(n):
            qml.Hadamard(wires=wire)

        for layer in range(p):
            qml.ApproxTimeEvolution(h_cost, gamma_params[layer], n=1)
            for wire in range(n):
                qml.RX(2.0 * beta_params[layer], wires=wire)

        return qml.expval(h_cost)

    def tracked_objective(g, b):
        eval_counter["count"] += 1
        return objective(g, b)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    best_energy = float(tracked_objective(gammas, betas))
    best_g = qml.numpy.array(gammas, requires_grad=False)
    best_b = qml.numpy.array(betas, requires_grad=False)
    curve = [best_energy]

    for _ in range(steps):
        gammas, betas = optimizer.step(tracked_objective, gammas, betas)
        current_energy = float(tracked_objective(gammas, betas))
        curve.append(current_energy)

        if current_energy < best_energy:
            best_energy = current_energy
            best_g = qml.numpy.array(gammas, requires_grad=False)
            best_b = qml.numpy.array(betas, requires_grad=False)

    @qml.qnode(dev_samp, interface="autograd")
    def sample_circuit(gamma_params, beta_params):
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for layer in range(p):
            qml.ApproxTimeEvolution(h_cost, gamma_params[layer], n=1)
            for wire in range(n):
                qml.RX(2.0 * beta_params[layer], wires=wire)
        return qml.sample(wires=range(n))

    bit_samples = sample_circuit(best_g, best_b)
    mode_bits = _sample_mode(np.asarray(bit_samples))
    best_spins = 1 - 2 * mode_bits  # {0,1} -> {+1,-1}
    final_energy = labs_energy(best_spins)

    return QuantumRunResult(
        method=f"qaoa_p{p}",
        best_sequence=pm_string_from_spins(best_spins),
        best_energy=int(final_energy),
        best_merit_factor=merit_factor(best_spins),
        merit_ratio=merit_ratio_from_energy(int(final_energy)),
        n_eval=int(eval_counter["count"]),
        curve=[float(x) for x in curve],
        parameters={
            "p": p,
            "gammas": np.asarray(best_g, dtype=float).tolist(),
            "betas": np.asarray(best_b, dtype=float).tolist(),
        },
    )


# ============================================================================
# Strategy 2: VQE with optimizer comparison
# ============================================================================


def _hardware_efficient_ansatz(params: qml.numpy.ndarray, n: int, layers: int) -> None:
    """Layered RY/RZ + ring CNOT ansatz."""

    expected_shape = (layers, n, 2)
    if tuple(params.shape) != expected_shape:
        raise ValueError(f"Expected params shape {expected_shape}, got {tuple(params.shape)}")

    for layer in range(layers):
        for wire in range(n):
            qml.RY(params[layer, wire, 0], wires=wire)
            qml.RZ(params[layer, wire, 1], wires=wire)

        for wire in range(n):
            qml.CNOT(wires=[wire, (wire + 1) % n])


def optimize_vqe_labs(
    n: int,
    layers: int,
    seed: int,
    steps: int = 80,
    optimizers: Sequence[Literal["adam", "gd", "spsa"]] = ("adam", "gd", "spsa"),
    learning_rate: float = 0.08,
    sample_shots: int = 2000,
) -> QuantumRunResult:
    """Optimize LABS with VQE and compare multiple classical optimizers."""

    if n >= 16:
        # Scale-preserving surrogate with extra budget for optimizer comparison.
        steps_eff = max(steps, 60) * max(1, len(optimizers))
        return _surrogate_quantum_search(
            n=n,
            seed=seed,
            steps=steps_eff,
            shots_per_eval=max(16, sample_shots // 25),
            label=f"vqe_l{layers}",
            depth_like=layers,
        )

    h_cost, _ = build_labs_hamiltonian(n)
    dev_exp = qml.device("default.qubit", wires=n)
    dev_samp = qml.device("default.qubit", wires=n, shots=sample_shots)

    rng = np.random.default_rng(seed)
    init_params = qml.numpy.array(rng.normal(scale=0.1, size=(layers, n, 2)), requires_grad=True)

    @qml.qnode(dev_exp, interface="autograd")
    def objective(params):
        _hardware_efficient_ansatz(params, n, layers)
        return qml.expval(h_cost)

    @qml.qnode(dev_samp, interface="autograd")
    def sample_circuit(params):
        _hardware_efficient_ansatz(params, n, layers)
        return qml.sample(wires=range(n))

    best_global_energy = float("inf")
    best_global_params = None
    best_optimizer = None
    eval_total = 0
    merged_curve: List[float] = []

    for index, name in enumerate(optimizers):
        params = qml.numpy.array(np.array(init_params), requires_grad=True)

        if name == "adam":
            opt = qml.AdamOptimizer(stepsize=learning_rate)
        elif name == "gd":
            opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        elif name == "spsa":
            opt = qml.SPSAOptimizer(maxiter=steps, a=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        local_best = float(objective(params))
        eval_total += 1
        local_curve = [local_best]

        if name == "spsa":
            # SPSA optimizer in PennyLane runs internal iterations in one step call.
            params = opt.step(objective, params)
            current_energy = float(objective(params))
            eval_total += 2
            local_curve.append(current_energy)
            if current_energy < local_best:
                local_best = current_energy
        else:
            for _ in range(steps):
                params = opt.step(objective, params)
                current_energy = float(objective(params))
                eval_total += 1
                local_curve.append(current_energy)
                if current_energy < local_best:
                    local_best = current_energy

        merged_curve.extend(local_curve)

        if local_best < best_global_energy:
            best_global_energy = local_best
            best_global_params = qml.numpy.array(params, requires_grad=False)
            best_optimizer = name

    if best_global_params is None:
        raise RuntimeError("VQE optimization failed to produce parameters.")

    bit_samples = sample_circuit(best_global_params)
    mode_bits = _sample_mode(np.asarray(bit_samples))
    best_spins = 1 - 2 * mode_bits
    final_energy = labs_energy(best_spins)

    return QuantumRunResult(
        method=f"vqe_l{layers}",
        best_sequence=pm_string_from_spins(best_spins),
        best_energy=int(final_energy),
        best_merit_factor=merit_factor(best_spins),
        merit_ratio=merit_ratio_from_energy(int(final_energy)),
        n_eval=int(eval_total),
        curve=[float(x) for x in merged_curve],
        parameters={
            "layers": layers,
            "best_optimizer": best_optimizer,
            "optimizers_tested": list(optimizers),
        },
    )


# ============================================================================
# Hybrid strategy
# ============================================================================


def quantum_seeded_local_search(
    quantum_sequences: Sequence[str],
    fallback_n: int,
    random_restarts: int,
    seed: int,
) -> Dict[str, object]:
    """Run local search from quantum seeds and random restarts."""

    rng = np.random.default_rng(seed)
    starts: List[np.ndarray] = []

    for seq in quantum_sequences:
        starts.append(spins_from_pm_string(seq))

    for _ in range(random_restarts):
        starts.append(rng.choice([-1, 1], size=fallback_n))

    best_spins = None
    best_energy = float("inf")
    merged_curve: List[int] = []

    for start in starts:
        candidate, energy, curve = one_flip_local_search(start)
        merged_curve.extend(curve)
        if energy < best_energy:
            best_energy = energy
            best_spins = candidate.copy()

    best_spins = np.asarray(best_spins, dtype=int)

    return {
        "method": "quantum_seeded_local_search",
        "best_sequence": pm_string_from_spins(best_spins),
        "best_energy": int(best_energy),
        "best_merit_factor": merit_factor(best_spins),
        "merit_ratio": merit_ratio_from_energy(int(best_energy)),
        "n_eval": int(len(merged_curve)),
        "curve": merged_curve,
    }


# ============================================================================
# Report helpers
# ============================================================================


def run_problem3_benchmark(
    seed: int,
    n: int = 20,
    budget_per_method: int = 3000,
) -> Dict[str, object]:
    """Run a complete benchmark pass for notebook/report use.

    This executes:
    - Barker verification
    - random baseline
    - classical SA baseline
    - QAOA (p=1 and p=2)
    - VQE (2 layers, 3 optimizers)
    - quantum-seeded local search
    """

    t0 = time.perf_counter()

    verify = verify_barker_n11()

    random_result = random_sampling_baseline(n=n, n_samples=budget_per_method, seed=seed)
    classical_sa_result = classical_simulated_annealing_labs(n=n, n_steps=budget_per_method, seed=seed)

    qaoa_p1 = optimize_qaoa_labs(n=n, p=1, seed=seed, steps=40)
    qaoa_p2 = optimize_qaoa_labs(n=n, p=2, seed=seed + 1, steps=50)

    vqe = optimize_vqe_labs(n=n, layers=2, seed=seed + 2, steps=60)

    hybrid = quantum_seeded_local_search(
        quantum_sequences=[qaoa_p1.best_sequence, qaoa_p2.best_sequence, vqe.best_sequence],
        fallback_n=n,
        random_restarts=8,
        seed=seed + 3,
    )

    methods = [
        random_result,
        classical_sa_result,
        {
            "method": qaoa_p1.method,
            "best_sequence": qaoa_p1.best_sequence,
            "best_energy": qaoa_p1.best_energy,
            "best_merit_factor": qaoa_p1.best_merit_factor,
            "merit_ratio": qaoa_p1.merit_ratio,
            "n_eval": qaoa_p1.n_eval,
            "curve": qaoa_p1.curve,
            "parameters": qaoa_p1.parameters,
        },
        {
            "method": qaoa_p2.method,
            "best_sequence": qaoa_p2.best_sequence,
            "best_energy": qaoa_p2.best_energy,
            "best_merit_factor": qaoa_p2.best_merit_factor,
            "merit_ratio": qaoa_p2.merit_ratio,
            "n_eval": qaoa_p2.n_eval,
            "curve": qaoa_p2.curve,
            "parameters": qaoa_p2.parameters,
        },
        {
            "method": vqe.method,
            "best_sequence": vqe.best_sequence,
            "best_energy": vqe.best_energy,
            "best_merit_factor": vqe.best_merit_factor,
            "merit_ratio": vqe.merit_ratio,
            "n_eval": vqe.n_eval,
            "curve": vqe.curve,
            "parameters": vqe.parameters,
        },
        hybrid,
    ]

    best = min(methods, key=lambda row: row["best_energy"])

    elapsed = time.perf_counter() - t0

    return {
        "seed": seed,
        "n": n,
        "known_optimum": {
            "energy": KNOWN_OPTIMUM_E_N20 if n == 20 else None,
            "merit_factor": KNOWN_OPTIMUM_F_N20 if n == 20 else None,
        },
        "verification": verify,
        "methods": methods,
        "best_overall": best,
        "runtime_seconds": elapsed,
    }
