import time
import qibo
from qibo import Circuit, gates
import numpy as np

def create_circuit(n_qubits):
    """创建一个简单的纠缠电路"""
    c = Circuit(n_qubits)
    # 第一层：所有量子比特加 H 门
    for i in range(n_qubits):
        c.add(gates.H(i))
    # 第二层：相邻比特加 CNOT 门 (形成 GHZ 态类结构)
    for i in range(0, n_qubits - 1, 2):
        c.add(gates.CNOT(i, i+1))
    return c

def benchmark_backend(backend_name, n_qubits, trials=3):
    """对指定后端进行多次运行取平均时间"""
    times = []
    for _ in range(trials):
        circuit = create_circuit(n_qubits)
        
        # 设置后端
        if backend_name == "qibotn-quimb":
            computation_settings = {
                "MPI_enabled": False,
                "MPS_enabled": False,
                "NCCL_enabled": False,
                "expectation_enabled": False,
            }
            qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)
        else:
            qibo.set_backend(backend=backend_name)
            
        start_time = time.time()
        result = circuit()
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"[{backend_name}] N={n_qubits}: Avg Time = {avg_time:.4f} s")
    return avg_time

if __name__ == "__main__":
    print("Starting Benchmark...")
    print("-" * 50)
    
    # 测试范围：从 8 到 20 个量子比特
    # 注意：numpy 在超过 25-30 比特时会因为内存爆炸而崩溃，
    # 但 qibotn-quimb 通常能跑得更远。
    qubit_counts = [8, 10, 12, 14, 16, 18, 20]
    
    results = {}
    
    for n in qubit_counts:
        print(f"\nTesting N={n}...")
        
        # 1. 测试 numpy 后端
        try:
            t_numpy = benchmark_backend("numpy", n)
            results["numpy"] = results.get("numpy", {})
            results["numpy"][n] = t_numpy
        except Exception as e:
            print(f"[numpy] N={n}: Failed ({e})")
            results["numpy"][n] = None
            
        # 2. 测试 qibotn-quimb 后端
        try:
            t_quimb = benchmark_backend("qibotn-quimb", n)
            results["qibotn-quimb"] = results.get("qibotn-quimb", {})
            results["qibotn-quimb"][n] = t_quimb
        except Exception as e:
            print(f"[qibotn-quimb] N={n}: Failed ({e})")
            results["qibotn-quimb"][n] = None
            
    print("\n" + "=" * 50)
    print("Benchmark Summary:")
    print("-" * 50)
    
    # 打印对比表格
    print(f"{'Qubits':<10} | {'NumPy (s)':<15} | {'Quimb (s)':<15} | {'Speedup'}")
    print("-" * 50)
    
    for n in qubit_counts:
        t_np = results["numpy"].get(n)
        t_qb = results["qibotn-quimb"].get(n)
        
        if t_np and t_qb:
            speedup = t_np / t_qb
            print(f"{n:<10} | {t_np:<15.4f} | {t_qb:<15.4f} | {speedup:.2f}x")
        elif t_np:
            print(f"{n:<10} | {t_np:<15.4f} | {'N/A':<15} | N/A")
        elif t_qb:
            print(f"{n:<10} | {'N/A':<15} | {t_qb:<15.4f} | N/A")
        else:
            print(f"{n:<10} | {'Failed':<15} | {'Failed':<15} | N/A")