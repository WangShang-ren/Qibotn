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
depth = c.depth
print(f'16Q x {ng}g, depth={depth}')

mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

# Reference: max_bond=256
print('Running max_bond=256 (reference)...')
sys.stdout.flush()
t0 = time.perf_counter()
ref = OPT(qasm, None, mps_opts=mps_opts, max_bond=256)
ref_time = time.perf_counter() - t0
ref_vec = ref.flatten()
print(f'  {ref_time:.1f}s')

print(f'{"max_bond":>10s}  {"Time(s)":>10s}  {"AbsErr":>12s}  {"RelErr":>12s}  {"Fidelity":>12s}')
print('-' * 60)

for max_bd in [128, 96, 80, 64, 48, 32]:
    t0 = time.perf_counter()
    try:
        result = OPT(qasm, None, mps_opts=mps_opts, max_bond=max_bd)
        t1 = time.perf_counter() - t0
        vec = result.flatten()
        abs_err = float(np.max(np.abs(ref_vec - vec)))
        rel_err = float(np.max(np.abs(ref_vec - vec) / (np.abs(ref_vec) + 1e-15)))
        fid = float(np.abs(np.dot(ref_vec.conj(), vec)) ** 2)
        spd = ref_time / t1 if t1 > 0 else 0
        
        status = 'CORRECT' if fid > 0.999 else ('WRONG' if fid < 0.99 else 'WARN')
        print(f'{max_bd:10d}  {t1:10.1f}s  {abs_err:12.2e}  {rel_err:12.2e}  {fid:12.6f}  {status}')
    except Exception as e:
        print(f'{max_bd:10d}  {"FAILED":>10s}  {str(e)[:40]}')

print('\nDONE')