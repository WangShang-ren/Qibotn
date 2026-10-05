import time, sys, os, numpy as np, csv
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

def random_circuit(nqubits, ngates, seed=42):
    np.random.seed(seed)
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

OUTDIR = '/home/user/HPC/qibotn/results/baseline'
os.makedirs(OUTDIR, exist_ok=True)

configs = [
    # scaling MPS mode
    (10, 200), (12, 300), (14, 500), (16, 800), (18, 1200), (20, 1800),
    (22, 2500), (24, 3500), (26, 5000),
]

results = []
for nq, ng in configs:
    sys.stdout.write(f'Testing {nq:2d}Q x {ng:4d}g MPS... ')
    sys.stdout.flush()
    try:
        c = random_circuit(nq, ng)
        qasm = c.to_qasm()
        
        times = []
        for run in range(3):
            t0 = time.perf_counter()
            result = dense_vector_tn_qu(qasm, None, 
                {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}, backend='numpy')
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        print(f'mean={np.mean(times):.4f}s, min={np.min(times):.4f}s, '
              f'max={np.max(times):.4f}s')
        results.append({
            'nqubits': nq, 'ngates': ng, 'mode': 'mps',
            'mean': round(np.mean(times), 4), 'min': round(np.min(times), 4),
            'max': round(np.max(times), 4), 'status': 'OK'
        })
    except Exception as e:
        print(f'FAILED: {str(e)[:100]}')
        results.append({
            'nqubits': nq, 'ngates': ng, 'mode': 'mps',
            'mean': None, 'status': f'FAILED: {str(e)[:100]}'
        })

# Save
ts = time.strftime('%Y%m%d_%H%M%S')
csv_path = os.path.join(OUTDIR, f'baseline_{ts}.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['nqubits','ngates','mode','mean','min','max','status'])
    w.writeheader()
    w.writerows(results)
print(f'\nSaved to {csv_path}')
print('DONE')