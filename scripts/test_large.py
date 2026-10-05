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

# Test large circuits with aggressive max_bond
# Parameters chosen based on profiling: max_bond controls SVD matrix sizes
configs = [
    (16, 2000, 32),    # 16Q x 2K gates, max_bond=32
    (18, 3000, 32),    # 18Q x 3K gates
    (20, 5000, 16),    # 20Q x 5K gates, lower max_bond
    (22, 8000, 16),    # 22Q x 8K gates
    (24, 10000, 8),    # 24Q x 10K gates! Very aggressive
    (26, 12000, 8),    # 26Q x 12K gates! Extreme
]

mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}

for nq, ng, max_bd in configs:
    sys.stdout.write(f'\n{nq:2d}Q x {ng:5d}g (max_bond={max_bd})... ')
    sys.stdout.flush()
    
    try:
        c = random_circuit(nq, ng)
        qasm = c.to_qasm()
        depth = c.depth
        gate_count = len(c.queue)
        
        t0 = time.perf_counter()
        result = OPT(qasm, None, mps_opts=mps_opts, max_bond=max_bd)
        t1 = time.perf_counter()
        t = t1 - t0
        
        print(f'{t:.3f}s, depth={depth}, gates={gate_count}')
        
        # Append to CSV
        csv_path = os.path.join(OUTDIR, 'large_circuits.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if write_header:
                f.write('nqubits,ngates,depth,max_bond,time,status\n')
            f.write(f'{nq},{ng},{depth},{max_bd},{t:.4f},OK\n')
        
    except Exception as e:
        print(f'FAILED: {e!r}')
        csv_path = os.path.join(OUTDIR, 'large_circuits.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if write_header:
                f.write('nqubits,ngates,depth,max_bond,time,status\n')
            f.write(f'{nq},{ng},0,{max_bd},,FAILED:{str(e)[:100]}\n')
        
        # If OOM, stop trying larger circuits
        if 'Memory' in str(e) or 'cannot allocate' in str(e).lower():
            print('Memory limit reached, stopping larger circuits')
            break

print('\nDONE')