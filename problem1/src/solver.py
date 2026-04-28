"""
Problem 1: 01 Knapsack — QUBO and Quantum Annealing
Solver module

This module contains all solver functions for the knapsack problem.
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import dimod
import neal


# ============================================================================
# Problem Data (with corrected orientation per instructor erratum)
# ============================================================================

# Item data: (index, weight, value)
# Using corrected orientation: Weight row = values, Value row = weights
VALUES = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]  # from "Weight" row
WEIGHTS = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]  # from "Value" row
CAPACITY = 165
N_ITEMS = len(VALUES)


# ============================================================================
# Part 1: Classical Solver (Dynamic Programming)
# ============================================================================

def solve_knapsack_classical() -> Tuple[List[int], int, int]:
    """
    Solve the 01 knapsack problem using dynamic programming.
    
    Returns:
        selected_items: List of item indices (0-indexed) that are selected
        total_weight: Total weight of selected items
        total_value: Total value of selected items
    """
    # DP table: dp[i][w] = max value using items 0..i-1 with weight limit w
    dp = [[0 for _ in range(CAPACITY + 1)] for _ in range(N_ITEMS + 1)]
    
    # Fill the DP table
    for i in range(1, N_ITEMS + 1):
        for w in range(CAPACITY + 1):
            # Don't take item i-1
            dp[i][w] = dp[i - 1][w]
            
            # Take item i-1 if possible
            if WEIGHTS[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - WEIGHTS[i - 1]] + VALUES[i - 1])
    
    # Backtrack to find which items were selected
    selected_items = []
    w = CAPACITY
    for i in range(N_ITEMS, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= WEIGHTS[i - 1]
    
    selected_items.reverse()
    
    total_weight = sum(WEIGHTS[i] for i in selected_items)
    total_value = sum(VALUES[i] for i in selected_items)
    
    return selected_items, total_weight, total_value


# ============================================================================
# Part 2: QUBO Formulation with Slack Variables
# ============================================================================

def compute_qubo_slack_variables(lam: float) -> Tuple[Dict[Tuple[int, int], float], int]:
    """
    Construct the QUBO matrix using slack variables to handle the constraint.
    
    The constraint sum(w_i * x_i) <= W is converted to an equality using
    slack variables: sum(w_i * x_i) + sum(2^k * s_k) = W
    
    The objective becomes:
    min_x [ -sum(v_i * x_i) + lambda * (sum(w_i * x_i) + sum(2^k * s_k) - W)^2 ]
    
    Args:
        lam: Penalty coefficient lambda
    
    Returns:
        Q: QUBO dictionary {(i, j): Q_ij}
        M: Number of slack variables
    """
    Q = {}
    
    # Number of slack variables needed
    M = int(np.ceil(np.log2(CAPACITY)))  # M = 8 for W = 165
    
    # Objective term: -sum(v_i * x_i)
    for i in range(N_ITEMS):
        Q[(i, i)] = Q.get((i, i), 0) - VALUES[i]
    
    # Penalty term: lambda * (sum(w_i * x_i) + sum(2^k * s_k) - W)^2
    # Expanded: lambda * [(...) + 2*(...)*(...) + (...)]
    
    # Linear terms from expanding (A + B - W)^2 where A = sum(w_i*x_i), B = sum(2^k*s_k)
    # Coefficient of x_i: lambda * [2*w_i*(A+B-W)] evaluated at the x_i term
    # = lambda * 2 * w_i * (sum(w_j*x_j) + sum(2^k*s_k) - W)
    
    # This expands to quadratic terms. Let's compute systematically:
    # (sum w_i*x_i)^2 contributes: sum_i w_i^2*x_i^2 + 2*sum_{i<j} w_i*w_j*x_i*x_j
    # (sum 2^k*s_k)^2 contributes: sum_k (2^k)^2*s_k^2 + 2*sum_{k<l} 2^k*2^l*s_k*s_l
    # Cross terms: 2*(sum w_i*x_i)*(sum 2^k*s_k) = 2*sum_{i,k} w_i*2^k*x_i*s_k
    # Linear terms: -2*W*(sum w_i*x_i) + -2*W*(sum 2^k*s_k) + W^2
    
    # Quadratic x_i x_j terms (i < j)
    for i in range(N_ITEMS):
        for j in range(i + 1, N_ITEMS):
            Q[(i, j)] = Q.get((i, j), 0) + lam * 2 * WEIGHTS[i] * WEIGHTS[j]
    
    # Diagonal x_i x_i terms
    for i in range(N_ITEMS):
        Q[(i, i)] = Q.get((i, i), 0) + lam * WEIGHTS[i] ** 2 - lam * 2 * WEIGHTS[i] * CAPACITY
    
    # Slack variable indices: N_ITEMS to N_ITEMS + M - 1
    # Quadratic s_k s_l terms (k < l)
    for k in range(M):
        for l in range(k + 1, M):
            idx_k = N_ITEMS + k
            idx_l = N_ITEMS + l
            weight_k = 2 ** k
            weight_l = 2 ** l
            Q[(idx_k, idx_l)] = Q.get((idx_k, idx_l), 0) + lam * 2 * weight_k * weight_l
    
    # Diagonal s_k s_k terms
    for k in range(M):
        idx_k = N_ITEMS + k
        weight_k = 2 ** k
        Q[(idx_k, idx_k)] = Q.get((idx_k, idx_k), 0) + lam * weight_k ** 2 - lam * 2 * weight_k * CAPACITY
    
    # Cross terms x_i s_k
    for i in range(N_ITEMS):
        for k in range(M):
            idx_k = N_ITEMS + k
            weight_k = 2 ** k
            Q[(i, idx_k)] = Q.get((i, idx_k), 0) + lam * 2 * WEIGHTS[i] * weight_k
    
    # Constant term: lam * W^2 (not included in QUBO as it doesn't affect optimization)
    
    return Q, M


# ============================================================================
# Part 3: QUBO Solver with D-Wave
# ============================================================================

def solve_qubo_exact(Q: Dict, M: int) -> Tuple[List[int], int, int, float]:
    """
    Solve the QUBO using D-Wave's ExactSolver.
    
    Args:
        Q: QUBO dictionary
        M: Number of slack variables
    
    Returns:
        selected_items: Item indices selected in the best solution
        total_weight: Total weight
        total_value: Total value
        best_energy: Best energy found
    """
    bqm = dimod.BQM.from_qubo(Q)
    exact_sampler = dimod.ExactSolver()
    result = exact_sampler.sample(bqm)
    
    # Get the best solution
    best_solution = result.first.sample
    best_energy = result.first.energy
    
    # Extract selected items (x_i variables)
    selected_items = [i for i in range(N_ITEMS) if best_solution[i] == 1]
    total_weight = sum(WEIGHTS[i] for i in selected_items)
    total_value = sum(VALUES[i] for i in selected_items)
    
    return selected_items, total_weight, total_value, best_energy


def solve_qubo_simulated_annealing(
    Q: Dict,
    M: int,
    seed: int,
    num_reads: int
) -> Tuple[List[int], int, int, List[float], float]:
    """
    Solve the QUBO using simulated annealing.
    
    Args:
        Q: QUBO dictionary
        M: Number of slack variables
        seed: Random seed
        num_reads: Number of annealing runs
    
    Returns:
        selected_items: Item indices in the best solution
        total_weight: Total weight
        total_value: Total value
        energies: List of energies for all reads
        time_elapsed: Computation time
    """
    bqm = dimod.BQM.from_qubo(Q)
    sa_sampler = neal.SimulatedAnnealingSampler()
    
    t_start = time.time()
    result = sa_sampler.sample(bqm, num_reads=num_reads, seed=seed)
    t_elapsed = time.time() - t_start
    
    # Get the best solution
    best_solution = result.first.sample
    
    # Extract selected items
    selected_items = [i for i in range(N_ITEMS) if best_solution[i] == 1]
    total_weight = sum(WEIGHTS[i] for i in selected_items)
    total_value = sum(VALUES[i] for i in selected_items)
    
    # Collect all energies
    energies = [record.energy for record in result.data(fields=['energy'])]
    
    return selected_items, total_weight, total_value, energies, t_elapsed


def compute_success_probability(
    Q: Dict,
    M: int,
    seed: int,
    num_reads: int,
    classical_selection: List[int]
) -> float:
    """
    Compute the success probability for simulated annealing.
    
    Args:
        Q: QUBO dictionary
        M: Number of slack variables
        seed: Random seed
        num_reads: Number of annealing runs
        classical_selection: The optimal classical selection (0-indexed items)
    
    Returns:
        success_prob: Fraction of reads that found optimal solution
    """
    bqm = dimod.BQM.from_qubo(Q)
    sa_sampler = neal.SimulatedAnnealingSampler()
    result = sa_sampler.sample(bqm, num_reads=num_reads, seed=seed)
    
    # Create the optimal item bitstring
    optimal_items_bitstring = [1 if i in classical_selection else 0 for i in range(N_ITEMS)]
    
    success_count = 0
    for record in result.data():
        sample = record.sample
        sample_items_bitstring = [sample[i] for i in range(N_ITEMS)]
        if sample_items_bitstring == optimal_items_bitstring:
            success_count += 1
    
    success_prob = success_count / num_reads if num_reads > 0 else 0.0
    return success_prob
