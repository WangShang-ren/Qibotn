import time, sys, os, csv, json
import numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu as ORIGINAL
from qibotn.eval_qu_optimized import dense_vector_tn_qu_optimized as OPT

np.random.seed(42)
OUTDIR = '/home/user/HPC/qibotn/results/final'
os.makedirs(OUTDIR, exist_ok=True)

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

def run_3average(fn, qasm, kwargs, label):
    times = []
    for run in range(3):
        t0 = time.perf_counter()
        result = fn(qasm, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.min(times), np.max(times), result

# Benchmarks
benchmarks = [
    # (nqubits, ngates, label, max_bond)
    (4, 20, "4Q_20g", None),
    (6, 40, "6Q_40g", None),
    (8, 80, "8Q_80g", None),
    (10, 200, "10Q_200g", None),
    (12, 500, "12Q_500g", 64),
    (14, 1000, "14Q_1Kg", 64),
    (16, 2000, "16Q_2Kg", 64),
    (18, 3000, "18Q_3Kg", 64),
    (20, 5000, "20Q_5Kg", 32),  # reduced max_bond for very large
    (22, 8000, "22Q_8Kg", 32),
    (24, 10000, "24Q_10Kg", 16),  # extreme: 24Q, 10K gates!
    (26, 12000, "26Q_12Kg", 16),
]

mps_baseline = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}
mps_opt = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

results = []
print(f'{"Config":>15s}  {"Original":>10s}  {"Optimized":>10s}  {"Speedup":>8s}  {"Diff":>10s}  {"Status":>8s}')
print('-'*85)

for nq, ng, label, max_bond in benchmarks:
    c = random_circuit(nq, ng)
    qasm = c.to_qasm()
    depth = c.depth
    
    sys.stdout.write(f'{label:>15s}  ')
    sys.stdout.flush()
    
    # Original
    try:
        mean_o, min_o, max_o, ref = run_3average(ORIGINAL, qasm, 
            {'initial_state': None, 'mps_opts': mps_baseline}, 'orig')
        sys.stdout.write(f'{min_o:10.3f}  ')
        sys.stdout.flush()
    except Exception as e:
        print(f'\n  Original FAILED: {e!r}')
        results.append({'label': label, 'nqubits': nq, 'ngates': ng, 'depth': depth,
                       'original': None, 'optimized': None, 'status': f'ORIG_FAIL: {e!r}'})
        continue
    
    # Optimized
    try:
        mean_p, min_p, max_p, result = run_3average(OPT, qasm,
            {'initial_state': None, 'mps_opts': mps_opt, 'max_bond': max_bond}, 'opt')
        
        diff = float(np.max(np.abs(ref.flatten() - result.flatten())))
        speedup = min_o / min_p if min_p > 0 else 0
        status = 'PASS' if diff < 1e-3 else ('WARN' if diff < 1e-1 else 'FAIL')
        
        print(f'{min_p:10.3f}  {speedup:7.2f}x  {diff:10.2e}  {status}')
        
        results.append({
            'label': label, 'nqubits': nq, 'ngates': ng, 'depth': depth,
            'original_mean': round(mean_o, 4), 'original_min': round(min_o, 4),
            'optimized_mean': round(mean_p, 4), 'optimized_min': round(min_p, 4),
            'speedup': round(speedup, 2), 'diff': f'{diff:.2e}',
            'max_bond': max_bond, 'status': status
        })
    except Exception as e:
        print(f'  FAILED: {e!r}')
        results.append({'label': label, 'nqubits': nq, 'ngates': ng, 'depth': depth,
                       'original_min': round(min_o, 4),
                       'optimized_min': None, 'speedup': None,
                       'status': f'OPT_FAIL: {e!r}'})

# Save results
ts = time.strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUTDIR, f'final_benchmark_{ts}.csv')
with open(csv_path, 'w', newline='') as f:
    fields = ['label','nqubits','ngates','depth','original_min','optimized_min',
              'speedup','diff','max_bond','status']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(results)

json_path = os.path.join(OUTDIR, f'final_benchmark_{ts}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to {csv_path}')
print('FINAL BENCHMARK COMPLETE')