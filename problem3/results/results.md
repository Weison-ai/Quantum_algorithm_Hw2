# Problem 3 Results (seed = 0 placeholder)

Generated from [problem3/problem3.ipynb](../problem3.ipynb) on 2026-04-27.

## (a) Cost and merit implementation check

Barker verification (`+++---+--+-`):
- `N = 11`
- `C_k = [0, -1, 0, -1, 0, -1, 0, -1, 0, -1]`
- `E = 5`
- `F = 12.10`
- Verification status: **pass**

Chosen strategies:
1. Quartic-Hamiltonian QAOA (`qaoa_p1`, `qaoa_p2`)
2. VQE with hardware-efficient ansatz + optimizer comparison (`vqe_l2`)
3. Hybrid quantum-seeded local search

Hamiltonian encoding:
- LABS objective encoded directly as quartic Pauli-Z terms from
  $E(s)=\sum_{k=1}^{N-1}\left(\sum_{i=1}^{N-k}s_is_{i+k}\right)^2$.

## (b) N = 20 strategy outcomes

Known optimum (reference):
- $E^* = 26$
- $F^* = 400/52 \approx 7.6923$

Reported metrics: best sequence, $E_{best}$, $F_{best}$, $r=F_{best}/F^*=E^*/E_{best}$, and $N_{eval}$.

- **VQE (layers = 2)**
  - Sequence: `++-++-++++-+-+---+++`
  - $E_{best}=34$, $F_{best}=5.8824$, $r=0.7647$, $N_{eval}=14400$
- **QAOA (p = 1)**
  - Sequence: `-++++----+--++--+-+-`
  - $E_{best}=50$, $F_{best}=4.0000$, $r=0.5200$, $N_{eval}=4000$
- **QAOA (p = 2)**
  - Sequence: `++++---+-+-++-++-+++`
  - $E_{best}=50$, $F_{best}=4.0000$, $r=0.5200$, $N_{eval}=5000$
- **Hybrid quantum-seeded local search**
  - Sequence: `++-++-++++-+-+---+++`
  - $E_{best}=34$, $F_{best}=5.8824$, $r=0.7647$, $N_{eval}=38$

## (c) Comparison table with two baselines

| Method | Best sequence | E_best | F_best | r | N_eval |
|---|---|---:|---:|---:|---:|
| random baseline | -+-++-+-+---+-----++ | 54 | 3.7037 | 0.4815 | 3000 |
| classical SA baseline | ++---+---+-++-+--+++ | 54 | 3.7037 | 0.4815 | 3000 |
| QAOA p=1 | -++++----+--++--+-+- | 50 | 4.0000 | 0.5200 | 4000 |
| QAOA p=2 | ++++---+-+-++-++-+++ | 50 | 4.0000 | 0.5200 | 5000 |
| VQE l=2 | ++-++-++++-+-+---+++ | 34 | 5.8824 | 0.7647 | 14400 |
| quantum-seeded local search | ++-++-++++-+-+---+++ | 34 | 5.8824 | 0.7647 | 38 |

Convergence curve was generated in the notebook ("LABS N=20 convergence" figure).

## (d) Discussion

Among tested methods, VQE and hybrid quantum-seeded local search performed best, both reaching $E=34$ ($r\approx0.7647$), clearly improving over random and classical SA baselines. QAOA at low depth ($p=1,2$) improved baseline performance but plateaued at $E=50$, suggesting insufficient expressivity or optimization depth for the glassy LABS landscape. Compared with Max-Cut, LABS has quartic/nonlocal interactions and much denser frustration, so local minima are more numerous and harder to escape. The strongest practical effect came from combining variational sampling with a local post-optimization stage; the hybrid stage quickly consolidated the best candidate but did not surpass VQE’s best energy in this run. Budget allocation and optimizer behavior strongly influenced outcomes, with deeper/longer VQE search outperforming short QAOA schedules. These observations are consistent with the assignment’s framing that LABS is substantially harder than quadratic Max-Cut and benefits from hybrid search pipelines.

## Notes

- This run used `seed = 0` placeholder. Replace with your student ID and re-run before submission.
- Target threshold $r\ge 0.85$ was **not** reached in this run.
