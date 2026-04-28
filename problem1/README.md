# Problem 1: 01 Knapsack — QUBO and Quantum Annealing

This folder contains the implementation for Problem 1 of QCAA Homework 2.

## Problem Overview

Solve a 01 knapsack problem with 10 items and capacity 165 using:
1. **Classical Method**: Dynamic programming to establish ground truth
2. **QUBO Formulation**: Convert to Quadratic Unconstrained Binary Optimization with penalty method
3. **D-Wave Solvers**: 
   - ExactSolver (classical verification)
   - SimulatedAnnealingSampler (quantum-inspired optimization)

## Files

- **problem1.ipynb**: Main Jupyter notebook that orchestrates the entire solution pipeline
- **src/solver.py**: Core solver functions:
  - Classical DP solver
  - QUBO formulation with slack variables
  - D-Wave solver integration
  - Success probability computation
- **src/__init__.py**: Package initialization for clean imports

## Key Features

### Corrected Data Orientation
Per instructor erratum, the printed table has transposed rows:
- **Weight row** → contains the **values** (v_i): [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
- **Value row** → contains the **weights** (w_i): [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]

### QUBO Constraint Handling
Uses **slack variables** to convert the inequality constraint to equality:
$$\sum_{i=1}^{10} w_i x_i + \sum_{k=0}^{7} 2^k s_k = W$$

This requires $M = \lceil \log_2(165) \rceil = 8$ slack variables, for a total of 18 binary variables.

### Penalty Coefficient Investigation
Tests three $\lambda$ values to determine when:
- Solutions are feasible (weight ≤ 165)
- Solutions are optimal (value = classical optimum)
- The penalty properly enforces the constraint

### Simulated Annealing Analysis
Investigates `num_reads` ∈ {10, 100, 1000, 10000} to measure:
- Success probability (fraction of reads finding optimal solution)
- Convergence behavior
- Trade-off between samples and solution quality

## Running the Code

### Prerequisites
```bash
pip install dwave-ocean-sdk numpy
```

### Execution
Open `problem1.ipynb` in Jupyter Notebook or VS Code and run all cells:

```bash
jupyter notebook problem1.ipynb
```

Or in VS Code, open the notebook and run cells sequentially (Shift+Enter).

### Expected Workflow
1. **Setup**: Import libraries and solver functions from `src/`
2. **Classical Solution**: Run DP and print ground truth
3. **QUBO Analysis**: Test 3 penalty coefficients (λ = 10, 50, 100)
4. **Simulated Annealing**: Test 4 num_reads values (10, 100, 1000, 10000)
5. **Comparison**: Display results table and statistics
6. **Discussion**: Interpret findings

## Algorithm Details

### Classical Knapsack (DP)
Time: O(n·W) where n=10, W=165
Space: O(n·W)

### QUBO Formulation
The QUBO objective is:
$$\min_x \left[ -\sum_{i=1}^{10} v_i x_i + \lambda \left( \sum_{i=1}^{10} w_i x_i + \sum_{k=0}^{7} 2^k s_k - W \right)^2 \right]$$

The Q matrix entries are derived by expanding the constraint penalty term.

### Penalty Parameter Selection
Heuristic: $\lambda > \frac{\max_i v_i}{\min_i w_i} = \frac{92}{23} \approx 4$

Tested values: {10, 50, 100}

## Notes

- Student ID seed is set to `seed = 0` (replace with actual student ID for reproducibility)
- All computations use `np.random.seed(seed)` for deterministic results
- Exact solver provides ground truth for verifying simulated annealing
- Comparison table includes classical method for baseline performance

## Expected Results

| Method | Value | Weight | Feasible | Time |
|--------|-------|--------|----------|------|
| Classical DP | 568 | 165 | Yes | ~0.001s |
| Exact QUBO (λ=10) | ? | ? | ? | ~0.01s |
| Exact QUBO (λ=50) | 568 | 165 | Yes | ~0.01s |
| Exact QUBO (λ=100) | 568 | 165 | Yes | ~0.01s |
| SA (reads=10) | ? | ? | ? | ~0.001s |
| SA (reads=100) | ? | ? | ? | ~0.01s |
| SA (reads=1000) | 568 | 165 | Yes | ~0.1s |
| SA (reads=10000) | 568 | 165 | Yes | ~1s |
