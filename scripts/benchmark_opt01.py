import time, sys, os, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu as ORIGINAL
from qibotn.eval_qu_optimized import dense_vector_tn_qu_optimized as OPTIMIZED

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

def benchmark(label, fn, qasm, kwargs):
    times = []
    for run in range(3):
        t0 = time.perf_counter()
        result = fn(qasm, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times), np.min(times), result

configs = [
    (8, 80), (10, 200), (12, 300), (14, 500),
]

print('='*70)
print('OPT-01: Comparison Benchmark')
print('='*70)
print(f'{"Config":>15s}  {"Original":>10s}  {"Optimized":>10s}  {"Speedup":>8s}  {"Correct":>8s}')
print('-'*70)

mps = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}

for nq, ng in configs:
    c = random_circuit(nq, ng)
    qasm = c.to_qasm()
    
    # Original
    sys.stdout.write(f'{nq:3d}Q x {ng:4d}g  ')
    sys.stdout.flush()
    
    mean_orig, min_orig, ref = benchmark('orig', ORIGINAL, qasm, 
        {'initial_state': None, 'mps_opts': mps})
    sys.stdout.write(f'{min_orig:10.3f}  ')
    sys.stdout.flush()
    
    # Optimized (better cutoff, simplified seq)
    opt_kwargs = {'initial_state': None, 
                  'mps_opts': {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'},
                  'simplify_seq': 'ADCR'}
    mean_opt, min_opt, result = benchmark('opt', OPTIMIZED, qasm, opt_kwargs)
    
    # Correctness
    diff = np.max(np.abs(ref.flatten() - result.flatten()))
    correct = 'PASS' if diff < 1e-6 else f'FAIL({diff:.1e})'
    speedup = min_orig / min_opt if min_opt > 0 else 0
    
    print(f'{min_opt:10.3f}  {speedup:7.2f}x  {correct}')
    
print('DONE')