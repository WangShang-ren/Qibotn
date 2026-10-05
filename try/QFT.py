import numpy as np
from qibo import Circuit, gates
import qibo

# 配置计算参数
computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
}

# 正确写法：backend="qibotn", platform="qutensornet"（qutensornet 就是 quimb 引擎）
qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings)

# 构建电路
c = Circuit(4)
c.add(gates.H(0))
c.add(gates.H(1))

# 执行
result = c()
print(result.state())