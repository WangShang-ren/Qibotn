import time, sys, os, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu_optimized import dense_vector_tn_qu_optimized

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

print('=== Max Bond Dimension Impact on Performance ===')

for nq, ng in [(12, 300), (14, 500)]:
    c = random_circuit(nq, ng)
    qasm = c.to_qasm()
    
    # Reference: no max_bond
    t0 = time.perf_counter()
    ref = dense_vector_tn_qu_optimized(qasm, None,
        mps_opts={'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'})
    ref_time = time.perf_counter() - t0
    
    print(f'\n{nq}Q x {ng}g (reference no max_bond: {ref_time:.3f}s)')
    print(f'{"max_bond":>12s}  {"Time":>10s}  {"Speedup":>8s}  {"Diff":>12s}')
    print('-'*50)
    
    for max_bond in [1024, 512, 256, 128, 64, 32, 16, 8]:
        try:
            t0 = time.perf_counter()
            result = dense_vector_tn_qu_optimized(qasm, None,
                mps_opts={'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'},
                max_bond=max_bond)
            t = time.perf_counter() - t0
            diff = np.max(np.abs(ref.flatten() - result.flatten()))
            speedup = ref_time / t if t > 0 else 0
            print(f'{max_bond:12d}  {t:10.3f}s  {speedup:7.2f}x  {diff:12.2e}')
        except Exception as e:
            print(f'{max_bond:12d}  {"FAILED":>10s}  {str(e)[:30]}')
    
print('\nDONE')