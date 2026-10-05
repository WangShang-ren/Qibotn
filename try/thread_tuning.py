import time
import os
import subprocess
import sys

# 测试用的基础代码（作为字符串传递给子进程，以隔离环境变量）
TEST_CODE = """
import qibo
from qibo import Circuit, gates

computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
}
qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)

c = Circuit(20)
for i in range(20): c.add(gates.H(i))
for i in range(0, 19, 2): c.add(gates.CNOT(i, i+1))
for i in range(1, 19, 2): c.add(gates.CNOT(i, i+1))
result = c()
"""

def run_benchmark(n_threads):
    """在指定线程数下运行基准测试"""
    env = os.environ.copy()
    # 核心调优参数：限制所有底层数学库的线程数
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["NUMEXPR_NUM_THREADS"] = str(n_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads)
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", TEST_CODE],
        env=env,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        return None, result.stderr
    return elapsed, None

if __name__ == "__main__":
    print("Starting System-Level Thread Tuning...")
    print("-" * 50)
    
    # 测试不同的线程数
    thread_counts = [1, 2, 4, 8, 16]
    
    print(f"{'Threads':<10} | {'Time (s)':<15} | {'Status'}")
    print("-" * 50)
    
    for n in thread_counts:
        elapsed, err = run_benchmark(n)
        if err:
            print(f"{n:<10} | {'N/A':<15} | Failed: {err[:30]}")
        else:
            print(f"{n:<10} | {elapsed:<15.4f} | Success")