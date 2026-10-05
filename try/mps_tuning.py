import time
import qibo
from qibo import Circuit, gates
import numpy as np

def create_local_entangling_circuit(n_qubits, depth):
    """创建一个具有局部纠缠的电路"""
    c = Circuit(n_qubits)
    for _ in range(depth):
        for i in range(n_qubits):
            c.add(gates.H(i))
        for i in range(0, n_qubits - 1, 2):
            c.add(gates.CZ(i, i+1))
        for i in range(n_qubits):
            c.add(gates.RZ(i, theta=0.5))
        for i in range(1, n_qubits - 1, 2):
            c.add(gates.CZ(i, i+1))
        for i in range(n_qubits):
            c.add(gates.RZ(i, theta=0.5))
    return c

def benchmark_mps(max_bond_dim, n_qubits, depth, trials=3):
    """使用指定 max_bond_dim 的 MPS 模式进行模拟"""
    times = []
    
    # 配置 MPS 参数 (适配 qibotn 0.3.4)
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": {
            "max_bond_dimension": max_bond_dim,  # 关键参数：最大键维度
            "svd_min_cutoff": 1e-12,             # SVD 截断阈值
        },
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)
    
    for _ in range(trials):
        circuit = create_local_entangling_circuit(n_qubits, depth)
        start_time = time.time()
        result = circuit()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times)

if __name__ == "__main__":
    print("Starting MPS Tuning Benchmark...")
    print("-" * 50)
    
    N_QUBITS = 20
    DEPTH = 5
    bond_dims = [10, 20, 50, 100, 200]
    
    results = []
    for dim in bond_dims:
        print(f"\nTesting Max Bond Dimension={dim}...")
        try:
            t = benchmark_mps(dim, N_QUBITS, DEPTH)
            print(f"[Bond Dim={dim}] Avg Time = {t:.4f} s")
            results.append((dim, t))
        except Exception as e:
            print(f"[Bond Dim={dim}] Failed ({e})")
            
    print("\n" + "=" * 50)
    print("MPS Tuning Summary:")
    print("-" * 50)
    print(f"{'Bond Dim':<12} | {'Time (s)':<15}")
    print("-" * 50)
    for dim, t in results:
        print(f"{dim:<12} | {t:<15.4f}")