import time, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

c = Circuit(4)
c.add(gates.H(0))
c.add(gates.H(1))
c.add(gates.CNOT(0,2))
c.add(gates.CNOT(1,3))
qasm = c.to_qasm()
print('QASM generated, running...')
t0 = time.perf_counter()
result = dense_vector_tn_qu(qasm, None, None, backend='numpy')
t1 = time.perf_counter()
print(f'Done in {t1-t0:.4f}s, shape: {result.shape}')
print('First 4 amplitudes:', result.flatten()[:4])

# test MPS
print('Testing MPS...')
t0 = time.perf_counter()
result2 = dense_vector_tn_qu(qasm, None, 
    {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}, backend='numpy')
t1 = time.perf_counter()
print(f'Done in {t1-t0:.4f}s, shape: {result2.shape}')
diff = np.max(np.abs(result.flatten() - result2.flatten()))
print(f'Max diff: {diff:.2e}')
