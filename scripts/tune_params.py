import time, sys, os, numpy as np, csv, itertools
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

np.random.seed(42)

def random_circuit(nqubits, ngates):
    c = Circuit(nqubits)
    for _ in range(ngates):
        t = np.random.random()
        if t < 0.4:
            c.add(gates.H(np.random.randint(0, nqubits)))
        elif t < 0.7:
            c.add(gates.RX(np.random.randint(0, nqubits), np.random.uniform(0, 2*np.pi)))
        elif nqubits >= 2:
            qs = np.random.choice(nqubits, 2, replace=False)
            c.add(gates.CNOT(qs[0], qs[1]))
        else:
            c.add(gates.RZ(np.random.randint(0, nqubits), np.random.uniform(0, 2*np.pi)))
    return c

OUTDIR = '/home/user/HPC/qibotn/results/thread_scaling'
os.makedirs(OUTDIR, exist_ok=True)

# Test on 14Q - representative medium workload
nq, ng = 14, 500
c = random_circuit(nq, ng)
qasm = c.to_qasm()
print(f'Circuit: {nq}Q x {ng}g')

results = []

# ---- Thread tuning ----
omp_values = [1, 2, 4, 8, 16]
print('\n=== Thread Tuning (OMP_NUM_THREADS) ===')
for omp in omp_values:
    os.environ['OMP_NUM_THREADS'] = str(omp)
    os.environ['OPENBLAS_NUM_THREADS'] = str(omp)
    
    mps_opts = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}
    t0 = time.perf_counter()
    result = dense_vector_tn_qu(qasm, None, mps_opts, backend='numpy')
    t1 = time.perf_counter()
    print(f'  OMP={omp:2d}: {t1-t0:.4f}s')
    results.append({'param': f'OMP={omp}', 'value': omp, 'time': round(t1-t0, 4)})

# ---- SVD cutoff tuning ----
cutoff_values = [1e-12, 1e-10, 1e-8, 1e-6]
print('\n=== SVD Cutoff Tuning ===')
mps_opts_ref = {'method': 'svd', 'cutoff': 1e-12, 'cutoff_mode': 'abs'}
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

t0 = time.perf_counter()
ref_result = dense_vector_tn_qu(qasm, None, mps_opts_ref, backend='numpy')
t1 = time.perf_counter()
ref_time = t1 - t0

for cutoff in cutoff_values:
    mps_opts = {'method': 'svd', 'cutoff': cutoff, 'cutoff_mode': 'abs'}
    t0 = time.perf_counter()
    result = dense_vector_tn_qu(qasm, None, mps_opts, backend='numpy')
    t1 = time.perf_counter()
    
    diff = np.max(np.abs(ref_result.flatten() - result.flatten()))
    print(f'  cutoff={cutoff:.0e}: {t1-t0:.4f}s, diff={diff:.2e}')
    results.append({'param': f'cutoff={cutoff:.0e}', 'value': cutoff, 'time': round(t1-t0, 4), 'diff': f'{diff:.2e}'})

# ---- Contraction optimizer tuning ----
print('\n=== Contraction Optimizer Tuning ===')
optimizers = ['auto-hq', 'greedy', 'random-greedy', 'betweenness']

# Test via eval_qu with full_simplify then to_dense
from qibotn.backends.quimb import METHODS, CLASSES_ROOTS
import quimb.tensor as qtn

for opt in optimizers:
    try:
        circ_quimb = qtn.circuit.CircuitMPS.from_openqasm2_str(qasm, psi0=None,
            gate_opts={'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'})
        interim = circ_quimb.psi.full_simplify(seq='DRC')
        t0 = time.perf_counter()
        amplitudes = interim.to_dense(backend='numpy', optimize=opt)
        t1 = time.perf_counter()
        diff = np.max(np.abs(ref_result.flatten() - amplitudes.flatten()))
        print(f'  {opt:16s}: {t1-t0:.4f}s, diff={diff:.2e}')
        results.append({'param': f'opt={opt}', 'value': 0, 'time': round(t1-t0, 4), 'diff': f'{diff:.2e}'})
    except Exception as e:
        print(f'  {opt:16s}: FAILED - {e}')

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUTDIR, f'tuning_{ts}.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f'\nSaved to {csv_path}')
print('DONE')