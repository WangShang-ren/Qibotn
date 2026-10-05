import time, sys, os, numpy as np
from qibo import Circuit, gates
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

configs = [
    (28, 5000, 8),
    (28, 10000, 8),
    (30, 5000, 8),
    (30, 10000, 8),
]

mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

for nq, ng, max_bd in configs:
    sys.stdout.write(f'\n{nq:2d}Q x {ng:5d}g (max_bond={max_bd})... ')
    sys.stdout.flush()
    
    try:
        c = random_circuit(nq, ng)
        qasm = c.to_qasm()
        depth = c.depth
        
        t0 = time.perf_counter()
        result = OPT(qasm, None, mps_opts=mps_opts, max_bond=max_bd)
        t1 = time.perf_counter()
        t = t1 - t0
        
        print(f'{t:.3f}s, depth={depth}')
        
        csv_path = os.path.join(OUTDIR, 'large_circuits.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if write_header:
                f.write('nqubits,ngates,depth,max_bond,time,status\n')
            f.write(f'{nq},{ng},{depth},{max_bd},{t:.4f},OK\n')
        
    except Exception as e:
        print(f'FAILED: {e!r}')
        csv_path = os.path.join(OUTDIR, 'large_circuits.csv')
        with open(csv_path, 'a') as f:
            f.write(f'{nq},{ng},0,{max_bd},,FAILED:{str(e)[:100]}\n')
        if 'Memory' in str(e) or 'cannot allocate' in str(e).lower():
            break

print('\nDONE')