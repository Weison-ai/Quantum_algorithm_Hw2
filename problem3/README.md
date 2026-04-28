# Problem 3: LABS (Low Autocorrelation Binary Sequences)

This folder implements the same notebook-first workflow used in Problem 1 and Problem 2.

## Files

- `src/solver.py`: reusable functions for Problem 3.
- `src/__init__.py`: package exports.

## Implemented strategies

The solver includes at least two non-trivial quantum/hybrid strategies:

1. **Quartic-Hamiltonian QAOA** (`optimize_qaoa_labs`)
   - Builds the LABS quartic Hamiltonian directly in Pauli-Z terms.
   - Optimizes QAOA parameters for depth `p`.
2. **VQE with optimizer comparison** (`optimize_vqe_labs`)
   - Uses a hardware-efficient ansatz (RY/RZ + ring CNOT).
   - Compares multiple optimizers (`adam`, `gd`, `spsa`).

It also includes:
- **Random baseline** (`random_sampling_baseline`)
- **Classical simulated annealing baseline** (`classical_simulated_annealing_labs`)
- **Hybrid quantum-seeded local search** (`quantum_seeded_local_search`)

## Quick usage

In your notebook:

```python
from problem3.src import run_problem3_benchmark

seed = 0  # replace with your student ID numeric part
report = run_problem3_benchmark(seed=seed, n=20, budget_per_method=3000)
report["best_overall"]
```

## Verification

Use `verify_barker_n11()` before running `N=20`:

- Barker sequence tested: `+++---+--+-`
- Expected: `E = 5`, `F = 12.10`

## Notes

- For final submission, replace placeholder `seed=0` with your student ID.
- The benchmark helper returns convergence curves and evaluation budgets (`n_eval`) for report plots.
