import time, cProfile, pstats, sys, os, numpy as np
from qibo import Circuit, gates
from qibotn.eval_qu import dense_vector_tn_qu

np.random.seed(42)

# 12Q x 300g circuit
nq, ng = 12, 300
c = Circuit(nq)
for _ in range(ng):
    t = np.random.random()
    if t < 0.4:
        c.add(gates.H(np.random.randint(0, nq)))
    elif t < 0.7:
        c.add(gates.RX(np.random.randint(0, nq), np.random.uniform(0, 2*np.pi)))
    elif nq >= 2:
        qs = np.random.choice(nq, 2, replace=False)
        c.add(gates.CNOT(qs[0], qs[1]))
    else:
        c.add(gates.RZ(np.random.randint(0, nq), np.random.uniform(0, 2*np.pi)))

qasm = c.to_qasm()
print(f'Circuit: {nq}Q x {len(c.queue)}g')
print(f'Depth: {c.depth}')
print(f'QASM size: {len(qasm)} chars')
print()

mps_opts = {'method': 'svd', 'cutoff': 1e-10, 'cutoff_mode': 'abs'}

profiler = cProfile.Profile()
profiler.enable()
t0 = time.perf_counter()
result = dense_vector_tn_qu(qasm, None, mps_opts, backend='numpy')
t1 = time.perf_counter()
profiler.disable()

print(f'Runtime: {t1-t0:.4f}s')
print(f'Output shape: {result.shape}')
print()

# Save profiling results
outdir = '/home/user/HPC/qibotn/results/baseline'
os.makedirs(outdir, exist_ok=True)
prof_path = os.path.join(outdir, 'profile_12q_300g.prof')
profiler.dump_stats(prof_path)
print(f'Profile saved to {prof_path}')
print()

# Print top 20 functions by tottime
stats = pstats.Stats(profiler)
stats.sort_stats('tottime')
stats.print_stats(30)