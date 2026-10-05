"""
28Q Convergence Test
Compare results at increasing max_bond values.
If max_bond=B and max_bond=2*B produce same result -> converged.
Cannot use full statevector (2^28 entries is infeasible).
Use observable/overlap/fidelity between max_bond levels instead.
"""
import time, sys, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu_optimized import dense_vector_tn_qu_optimized as OPT

np.random.seed(42)
nq, ng = 28, 10000

c = Circuit(nq)
for _ in range(ng):
    t = np.random.random()
    if t < 0.4:
        c.add(gates.H(np.random.randint(0, nq)))
    elif t < 0.7:
        c.add(gates.RX(np.random.randint(0, nq), np.random.uniform(0, 2*np.pi)))
    elif nq >= 2:
        qs = np.random.choice(nq, 2, replace=False)
        c.add(gates.CNOT(qs[0], qs[1]))
    else:
        c.add(gates.RZ(np.random.randint(0, nq), np.random.uniform(0, 2*np.pi)))

qasm = c.to_qasm()
depth = c.depth
print(f'28Q x {len(c.queue)}g, depth={depth}')

mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

# Test max_bond from small to large
# Cannot run unlimited or very large (256+) due to memory
bond_values = [8, 16, 32, 64]
results = {}

print(f'{"max_bond":>10s}  {"Time(s)":>10s}  {"Result shape":>15s}  Status')
print('-' * 55)

for max_bd in bond_values:
    sys.stdout.write(f'{max_bd:10d}  ')
    sys.stdout.flush()
    try:
        t0 = time.perf_counter()
        result = OPT(qasm, None, mps_opts=mps_opts, max_bond=max_bd,
                     simplify_seq='ADCR')
        t1 = time.perf_counter()
        results[max_bd] = result.flatten()
        print(f'{t1-t0:10.1f}s  {str(result.shape):>15s}  OK')
    except Exception as e:
        print(f'{"FAILED":>10s}  {"":>15s}  {str(e)[:30]}')

# Compare convergence between levels
print()
print('=== Convergence Analysis ===')
print(f'{"Compare":>20s}  {"AbsDiff":>12s}  {"RelDiff":>12s}  {"Fidelity":>12s}  {"Converged?":>12s}')
print('-' * 75)

converged = False
convergence_pair = None

for i in range(len(bond_values) - 1):
    a = bond_values[i]
    b = bond_values[i + 1]
    if a not in results or b not in results:
        continue
    vec_a = results[a]
    vec_b = results[b]
    
    abs_diff = float(np.max(np.abs(vec_a - vec_b)))
    rel_diff = float(np.max(np.abs(vec_a - vec_b) / (np.abs(vec_b) + 1e-15)))
    fid = float(np.abs(np.dot(vec_a.conj(), vec_b)) ** 2)
    
    # Convergence criterion: fidelity > 0.9999 between adjacent bond levels
    is_converged = fid > 0.9999
    label = f'{a} vs {b}'
    
    print(f'{label:>20s}  {abs_diff:12.2e}  {rel_diff:12.2e}  {fid:12.6f}  '
          f'{"YES" if is_converged else "NO":>12s}')
    
    if is_converged:
        converged = True
        convergence_pair = (a, b)

print()
if converged:
    print(f'CONCLUSION: Results converge between max_bond={convergence_pair[0]} and '
          f'max_bond={convergence_pair[1]}.')
    print(f'Convergence test PASSES for 28Q.')
else:
    print('CONCLUSION: Results do NOT converge between available max_bond levels.')
    print('28Q correctness cannot be established with this method.')
    print('This is expected — 28Q random circuits have high entanglement,')
    print('requiring bond dimension >> 64 for accuracy.')
    print('28Q remains: "performance-only stress test"')

print('\nDONE')