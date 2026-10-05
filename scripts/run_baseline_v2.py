#!/usr/bin/env python3
'''QiboTN Baseline Benchmark v2 - CPU-only, Quimb backend'''
import time, os, sys, json, gc, csv
import numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

OUTDIR = '/home/user/HPC/qibotn/results/baseline'
os.makedirs(OUTDIR, exist_ok=True)

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

def run_one(nqubits, ngates, mode='mps', seed=42):
    circuit = random_circuit(nqubits, ngates, seed=seed)
    qasm = circuit.to_qasm()
    t0 = time.perf_counter()
    if mode == 'mps':
        mps_opts = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}
        result = dense_vector_tn_qu(qasm, None, mps_opts, backend='numpy')
    else:
        result = dense_vector_tn_qu(qasm, None, None, backend='numpy')
    t1 = time.perf_counter()
    return t1 - t0

results = []
configs = [
    (4, 20, 'flat'), (6, 40, 'flat'), (8, 80, 'flat'),
    (4, 20, 'mps'), (6, 40, 'mps'), (8, 80, 'mps'),
    (10, 200, 'mps'), (12, 400, 'mps'), (14, 800, 'mps'),
    (16, 1600, 'mps'), (18, 2000, 'mps'), (20, 2500, 'mps'),
]

print('Config     Mode  Time(s)')
print('-'*40)

for nq, ng, mode in configs:
    times = []
    for run_idx in range(3):
        try:
            t = run_one(nq, ng, mode=mode)
            times.append(t)
            print(f'{nq:2d}Qx{ng:4d}g {mode:4s}  r{run_idx+1}: {t:.4f}s')
        except Exception as e:
            print(f'{nq:2d}Qx{ng:4d}g {mode:4s}  FAILED: {e!r}')
            times = []
            break
    if times:
        r = {
            'nqubits': nq, 'ngates': ng, 'mode': mode,
            'mean': round(np.mean(times), 4),
            'min': round(np.min(times), 4),
            'max': round(np.max(times), 4),
            'std': round(np.std(times), 6),
            'cold': round(times[0], 4),
        }
        results.append(r)
        print(f'  => mean={r["mean"]:.4f}s, min={r["min"]:.4f}s\n')

ts = time.strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUTDIR, f'baseline_{ts}.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['nqubits','ngates','mode','mean','min','max','std','cold'])
    w.writeheader()
    w.writerows(results)
print(f'Results saved to {csv_path}')