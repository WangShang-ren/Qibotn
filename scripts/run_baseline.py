#!/usr/bin/env python3
'''QiboTN Baseline Benchmark - CPU-only, Quimb backend'''

import time, os, sys, json, gc, tracemalloc, csv
import numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

OUTDIR = '/home/user/HPC/qibotn/results/baseline'
LOGDIR = '/home/user/HPC/qibotn/logs/baseline'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

def random_circuit(nqubits, ngates, seed=42):
    np.random.seed(seed)
    c = Circuit(nqubits)
    gate_1q = [
        lambda i: gates.H(i),
        lambda i: gates.RX(i, np.random.uniform(0, 2*np.pi)),
        lambda i: gates.RZ(i, np.random.uniform(0, 2*np.pi)),
    ]
    gate_2q = [
        lambda i, j: gates.CNOT(i, j),
        lambda i, j: gates.CZ(i, j),
    ]
    
    for _ in range(ngates):
        if np.random.random() < 0.7 or nqubits < 2:
            g = np.random.choice(gate_1q)
            q = np.random.randint(0, nqubits)
            c.add(g(q))
        else:
            g2 = np.random.choice(gate_2q)
            qs = np.random.choice(nqubits, 2, replace=False)
            c.add(g2(qs[0], qs[1]))
    return c

def run_benchmark(nqubits, ngates, mode='mps', n_runs=3, seed=42):
    circuit = random_circuit(nqubits, ngates, seed=seed)
    qasm = circuit.to_qasm()
    
    times = []
    peak_mem = None
    
    for run in range(n_runs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        
        if mode == 'mps':
            mps_opts = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}
            result = dense_vector_tn_qu(qasm, None, mps_opts, backend='numpy')
        else:
            result = dense_vector_tn_qu(qasm, None, None, backend='numpy')
        
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        times.append(t1 - t0)
        peak_mem = peak / (1024**2)
    
    times = np.array(times)
    return {
        'nqubits': nqubits,
        'ngates': ngates,
        'mode': mode,
        'mean_time': float(np.mean(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'std_time': float(np.std(times)),
        'peak_memory_mb': float(peak_mem),
        'cold_run': float(times[0]),
        'warm_runs_mean': float(np.mean(times[1:])) if len(times) > 1 else float(times[0]),
        'status': 'OK'
    }

def check_correctness(nqubits, seed=99):
    c = random_circuit(nqubits, ngates=nqubits*2, seed=seed)
    qasm = c.to_qasm()
    
    result_flat = dense_vector_tn_qu(qasm, None, None, backend='numpy')
    result_mps = dense_vector_tn_qu(qasm, None, 
        {'method': 'svd', 'cutoff': 1e-12, 'cutoff_mode': 'abs'}, backend='numpy')
    
    diff = float(np.max(np.abs(result_flat.flatten() - result_mps.flatten())))
    return diff < 1e-8, diff

print('='*60)
print('QiboTN Baseline Benchmark')
print('='*60)
print(f'Python: {sys.version.split()[0]}')
print(f'QiboTN: {__import__("qibotn").__version__}')
print()

# Correctness check
print('--- Correctness Check ---')
for nq in [4, 5, 6]:
    ok, diff = check_correctness(nq)
    status = "PASS" if ok else "FAIL"
    print(f'  {nq}Q: {status} (diff={diff:.2e})')

print()

# Benchmark: MPS mode
configs = [
    (4, 20), (6, 40), (8, 80),
    (10, 200), (12, 400), (14, 800),
    (16, 1600), (18, 2000), (20, 2500),
]

results = []
print('--- MPS Mode Benchmark ---')
for nq, ng in configs:
    try:
        r = run_benchmark(nq, ng, mode='mps', n_runs=3)
        results.append(r)
        print(f'  {nq}Q x {ng}g: mean={r["mean_time"]:.4f}s, min={r["min_time"]:.4f}s, '
              f'cold={r["cold_run"]:.4f}s, mem={r["peak_memory_mb"]:.1f}MB')
    except Exception as e:
        print(f'  {nq}Q x {ng}g: FAILED - {e}')
        results.append({
            'nqubits': nq, 'ngates': ng, 'mode': 'mps',
            'mean_time': None, 'status': f'FAILED: {str(e)[:200]}'
        })

print()
print('--- Non-MPS Mode Benchmark ---')
for nq, ng in [(4, 20), (6, 40), (8, 80)]:
    try:
        r = run_benchmark(nq, ng, mode='flat', n_runs=3)
        results.append(r)
        print(f'  {nq}Q x {ng}g: mean={r["mean_time"]:.4f}s, min={r["min_time"]:.4f}s, '
              f'mem={r["peak_memory_mb"]:.1f}MB')
    except Exception as e:
        print(f'  {nq}Q x {ng}g: FAILED - {e}')
        results.append({
            'nqubits': nq, 'ngates': ng, 'mode': 'flat',
            'mean_time': None, 'status': f'FAILED: {str(e)[:200]}'
        })

ts = time.strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUTDIR, f'baseline_{ts}.csv')
with open(csv_path, 'w', newline='') as f:
    fields = ['nqubits','ngates','mode','mean_time','min_time','max_time','std_time',
              'cold_run','warm_runs_mean','peak_memory_mb','status']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(results)

json_path = os.path.join(OUTDIR, f'baseline_{ts}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to {csv_path}')