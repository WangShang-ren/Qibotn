import time
import numpy as np
import quimb.tensor as qtn
from qibo import gates as qibo_gates


def merge_adjacent_single_qubit_gates(circuit):
    """Merge adjacent single-qubit gates on the same qubit to reduce gate count.
    
    This is a circuit optimization that reduces the number of gates passed to
    the tensor network simulator, reducing SVD operations in MPS mode.
    """
    nqubits = circuit.nqubits
    merged_gates = [[] for _ in range(nqubits)]
    
    matrix_map = {
        'h': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, -1j], [1j, 0]]),
        'z': np.array([[1, 0], [0, -1]]),
        's': np.array([[1, 0], [0, 1j]]),
        't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    }
    
    for gate in circuit.queue:
        gate_name = getattr(gate, 'name', None)
        qubits = getattr(gate, 'qubits', ())
        
        if len(qubits) != 1 or gate_name not in ('h', 'x', 'y', 'z', 's', 't', 'rz', 'rx', 'ry'):
            yield gate
        else:
            target = qubits[0]
            merged_gates[target].append(gate)
    
    # Emit merged single-qubit gates
    for q in range(nqubits):
        if merged_gates[q]:
            combined = np.eye(2, dtype=complex)
            for gate in merged_gates[q]:
                gate_name = getattr(gate, 'name', None)
                if gate_name in matrix_map:
                    combined = matrix_map[gate_name] @ combined
                elif gate_name in ('rz', 'rx', 'ry'):
                    params = getattr(gate, 'parameters', ())
                    theta = params[0] if params else 0
                    if gate_name == 'rz':
                        mat = np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])
                    elif gate_name == 'rx':
                        c, s = np.cos(theta/2), np.sin(theta/2)
                        mat = np.array([[c, -1j*s], [-1j*s, c]])
                    elif gate_name == 'ry':
                        c, s = np.cos(theta/2), np.sin(theta/2)
                        mat = np.array([[c, -s], [s, c]])
                    combined = mat @ combined
                else:
                    combined = np.eye(2, dtype=complex)
                    break
            # Apply the combined unitary as a U3 matrix
            if not np.allclose(combined, np.eye(2)):
                from scipy.linalg import expm
                # Decompose to RZ-RY-RZ
                yield qibo_gates.U3(q, *decompose_u3(combined))


def decompose_u3(U):
    """Decompose a 2x2 unitary into U3(theta, phi, lambda) parameters."""
    det = np.linalg.det(U)
    phase = np.angle(det) / 2
    U_norm = U / np.sqrt(det)
    
    if abs(U_norm[0, 0]) > 1e-15:
        theta = 2 * np.arccos(np.clip(abs(U_norm[0, 0]), 0, 1))
    else:
        theta = np.pi
    
    if abs(U_norm[0, 0]) > 1e-15:
        phi_minus_lambda = np.angle(U_norm[1, 0])
        phi_plus_lambda = np.angle(U_norm[0, 0])
        phi = (phi_plus_lambda + phi_minus_lambda) / 2 + phase
        lam = (phi_plus_lambda - phi_minus_lambda) / 2 + phase
    else:
        phi_minus_lambda = np.angle(U_norm[1, 1])
        phi_plus_lambda = np.angle(U_norm[0, 1])
        phi = (phi_plus_lambda + phi_minus_lambda) / 2 + phase
        lam = (phi_plus_lambda - phi_minus_lambda) / 2 + phase
    
    return theta % (2*np.pi), phi % (2*np.pi), lam % (2*np.pi)


def dense_vector_tn_qu_optimized(qasm, initial_state=None, mps_opts=None, 
                                  backend='numpy', contraction_optimizer='auto-hq',
                                  max_bond=None, simplify_seq='ADCR'):
    """Optimized version of dense_vector_tn_qu with better defaults.
    
    Args:
        qasm: QASM circuit string
        initial_state: Initial MPS state (optional)
        mps_opts: MPS gate application options dict
        backend: 'numpy', 'cupy', or 'jax'
        contraction_optimizer: Contraction path optimizer
        max_bond: Maximum bond dimension for MPS
        simplify_seq: Tensor network simplification sequence
    """
    if mps_opts is None:
        mps_opts = {'method': 'svd', 'cutoff': 1e-8, 'cutoff_mode': 'abs'}
    
    if max_bond is not None:
        mps_opts['max_bond'] = max_bond
    
    if initial_state is not None:
        nqubits = int(np.log2(len(initial_state)))
        dims = tuple(2 * np.ones(nqubits, dtype=int))
        initial_state = qtn.tensor_1d.MatrixProductState.from_dense(initial_state, dims)
    
    circ_cls = qtn.circuit.CircuitMPS if mps_opts else qtn.circuit.Circuit
    circ_quimb = circ_cls.from_openqasm2_str(
        qasm, psi0=initial_state, gate_opts=mps_opts
    )
    
    interim = circ_quimb.psi.full_simplify(seq=simplify_seq)
    amplitudes = interim.to_dense(backend=backend, optimize=contraction_optimizer)
    
    return amplitudes