import time, os, sys, json, traceback
import numpy as np
from qibo import Circuit, gates, set_backend

RESULTS = []

def log(msg):
    print(f'[SMOKE] {msg}')
    RESULTS.append(msg)

log(f'Python: {sys.version}')
log(f'Numpy: {np.__version__}')

# Test 1: basic import
try:
    import qibotn
    log(f'QiboTN: {qibotn.__version__} from {qibotn.__file__}')
except Exception as e:
    log(f'FATAL: import qibotn failed: {e}')
    sys.exit(1)

# Test 2: backend list
try:
    from qibotn.backends import MetaBackend
    avail = MetaBackend().list_available()
    log(f'Available backends: {avail}')
except Exception as e:
    log(f'FATAL: backend listing failed: {e}')

# Test 3: Qibo backend registration and basic circuit (4 qubits, small)
try:
    set_backend(backend='qibotn', platform='quimb')
    c = Circuit(4)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 3))
    t0 = time.perf_counter()
    result = c()
    t1 = time.perf_counter()
    log(f'Qibo backend circuit (4Q, 4 gates): {t1-t0:.4f}s')
except Exception as e:
    log(f'FATAL: circuit execution through Qibo failed: {e}')
    traceback.print_exc()

# Test 4: direct quimb TN evaluation (small scale)
try:
    from qibotn.eval_qu import dense_vector_tn_qu
    c2 = Circuit(6)
    for i in range(6):
        c2.add(gates.H(i))
    c2.add(gates.CNOT(0, 3))
    c2.add(gates.CNOT(1, 4))
    c2.add(gates.CNOT(2, 5))
    qasm = c2.to_qasm()
    t0 = time.perf_counter()
    result_tn = dense_vector_tn_qu(qasm, None, None, backend='numpy')
    t1 = time.perf_counter()
    log(f'Direct TN eval (6Q, 9 gates): {t1-t0:.4f}s, output shape: {result_tn.shape}')
except Exception as e:
    log(f'FATAL: direct TN eval failed: {e}')
    traceback.print_exc()

# Test 5: MPS mode
try:
    mps_opts = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}
    c3 = Circuit(8)
    for i in range(8):
        c3.add(gates.H(i))
    for i in range(7):
        c3.add(gates.CNOT(i, i+1))
    qasm3 = c3.to_qasm()
    t0 = time.perf_counter()
    result_mps = dense_vector_tn_qu(qasm3, None, mps_opts, backend='numpy')
    t1 = time.perf_counter()
    log(f'MPS mode (8Q, 15 gates): {t1-t0:.4f}s, output shape: {result_mps.shape}')
except Exception as e:
    log(f'FATAL: MPS eval failed: {e}')
    traceback.print_exc()

# Test 6: Correctness check - compare MPS vs non-MPS for small circuit
try:
    c4 = Circuit(5)
    c4.add(gates.H(0))
    c4.add(gates.CNOT(0, 1))
    c4.add(gates.RX(0, np.pi/4))
    c4.add(gates.CZ(1, 2))
    qasm4 = c4.to_qasm()
    
    result_flat = dense_vector_tn_qu(qasm4, None, None, backend='numpy')
    result_mps2 = dense_vector_tn_qu(qasm4, None, 
        {'method': 'svd', 'cutoff': 1e-12, 'cutoff_mode': 'abs'}, backend='numpy')
    
    max_diff = np.max(np.abs(result_flat.flatten() - result_mps2.flatten()))
    log(f'Correctness: flat vs MPS max diff = {max_diff:.2e}')
    if max_diff < 1e-8:
        log('CORRECTNESS PASSED')
    else:
        log(f'CORRECTNESS FAILED: diff={max_diff}')
except Exception as e:
    log(f'FATAL: correctness check failed: {e}')

# Write results
with open('/home/user/HPC/qibotn/logs/smoke/smoke_test.log', 'w') as f:
    f.write('\n'.join(RESULTS))
print('Smoke test complete. Results saved to logs/smoke/smoke_test.log')
