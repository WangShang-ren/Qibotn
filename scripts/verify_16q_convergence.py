import time, sys, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu_optimized import dense_vector_tn_qu_optimized as OPT

np.random.seed(42)
nq, ng = 16, 800
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
print(f'16Q x {ng}g, depth={c.depth}')

mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

# Test max_bond=256 vs 320 (higher = more accurate)
print('max_bond=320...')
sys.stdout.flush()
t0 = time.perf_counter()
ref = OPT(qasm, None, mps_opts=mps_opts, max_bond=320)
print(f'  {time.perf_counter()-t0:.1f}s')

print('max_bond=256...')
sys.stdout.flush()
t0 = time.perf_counter()
r256 = OPT(qasm, None, mps_opts=mps_opts, max_bond=256)
print(f'  {time.perf_counter()-t0:.1f}s')
diff_256 = float(np.max(np.abs(ref.flatten() - r256.flatten())))
fid_256 = float(np.abs(np.dot(ref.flatten().conj(), r256.flatten())) ** 2)
print(f'  vs 320: diff={diff_256:.2e}, fid={fid_256:.6f}')

print('max_bond=200...')
sys.stdout.flush()
t0 = time.perf_counter()
r200 = OPT(qasm, None, mps_opts=mps_opts, max_bond=200)
print(f'  {time.perf_counter()-t0:.1f}s')
diff_200 = float(np.max(np.abs(ref.flatten() - r200.flatten())))
fid_200 = float(np.abs(np.dot(ref.flatten().conj(), r200.flatten())) ** 2)
print(f'  vs 320: diff={diff_200:.2e}, fid={fid_200:.6f}')

if fid_256 > 0.999:
    print('\nCONCLUSION: max_bond=256 is converged/correct for 16Q')
elif fid_256 > 0.99:
    print('\nCONCLUSION: max_bond=256 is mostly correct (fid > 0.99)')
else:
    print('\nCONCLUSION: max_bond=256 is NOT correct (fid < 0.99)')

print('DONE')