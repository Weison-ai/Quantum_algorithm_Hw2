# Problem 2: Max-Cut with QAOA

This folder contains the implementation scaffold for Problem 2 of QCAA Homework 2.

This folder follows the Problem 2 Max-Cut workflow in the homework sheet.

## Contents

- `src/solver.py`: graph generation, brute-force maximum cut, QAOA landscape evaluation, QAOA optimization, and simulated annealing helpers
- `src/__init__.py`: package exports for notebook imports

## Expected workflow

1. Generate the student-specific random graph with the provided seed.
2. Brute-force all 256 partitions to find the exact maximum cut.
3. Plot the `p = 1` QAOA landscape.
4. Optimize QAOA for `p in {1, 2, 3, 4}`.
5. Compare against simulated annealing.

## Dependencies

This workspace already includes the required packages in `pyproject.toml`:

- `networkx`
- `numpy`
- `matplotlib`
- `pennylane`
- `dwave-neal`

## Notes

- Replace the default seed with your student ID before producing final results.
- The helper functions are written so they can be imported into a notebook or script.
