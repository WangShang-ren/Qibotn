import time, sys, numpy as np
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

configs = [
    (4, 20, 'flat'), (6, 40, 'flat'), (8, 80, 'flat'),
    (4, 20, 'mps'), (6, 40, 'mps'), (8, 80, 'mps'),
    (10, 200, 'mps'),
]

for nq, ng, mode in configs:
    sys.stdout.write(f'Testing {nq}Q x {ng}g ({mode})... ')
    sys.stdout.flush()
    try:
        c = random_circuit(nq, ng)
        qasm = c.to_qasm()
        t0 = time.perf_counter()
        if mode == 'mps':
            result = dense_vector_tn_qu(qasm, None, 
                {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}, backend='numpy')
        else:
            result = dense_vector_tn_qu(qasm, None, None, backend='numpy')
        t1 = time.perf_counter()
        print(f'{t1-t0:.4f}s, shape={result.shape}')
    except Exception as e:
        print(f'FAILED: {e}')
print('DONE')