# QiboTN Profiling Report

**Date:** 2026-10-04 | **Workload:** 12Q x 300g MPS

## Profiling Method

cProfile on `dense_vector_tn_qu` MPS execution path, 950K function calls in 1.316s.

## Top CPU Functions (by tottime)

| # | Function | Calls | Time (s) | % |
|---|----------|-------|----------|------|
| 1 | svd_truncated_numba | 716 | 0.485 | 36.8 |
| 2 | numpy.tensordot | 1081 | 0.254 | 19.3 |
| 3 | llvmlite ffi __call__ | 4476 | 0.061 | 4.6 |
| 4 | autoray.do | 19419 | 0.042 | 3.2 |
| 5 | tensor_core.modify | 7449 | 0.026 | 2.0 |
| 6 | tensor_core.tensor_split | 961 | 0.021 | 1.6 |
| 7 | numpy.ndarray.reshape | 6465 | 0.017 | 1.3 |
| 8 | autoray._default_infer_from_sig | 24161 | 0.011 | 0.8 |
| 9 | transpose_like | 1922 | 0.011 | 0.8 |
| 10 | swap_sites_with_compress | 626 | 1.001 | 76.1* |
| 11 | canonicalize | 716 | 0.155 | 11.8* |
| 12 | svd_truncated_numpy | 716 | 0.004+0.528 | 40.1* |

(*) Cumulative time — includes time spent in called functions.

## Key Observations

1. **SVD dominates**: 36.8% self-time, 40.1% cumulative
2. **tensordot is #2**: 19.3% — this is the tensor contraction after SVD
3. **Numba JIT overhead**: 4.6% in llvmlite calls — JIT compilation cost
4. **autoray dispatch**: 3.2% + 0.8% — Python-level function dispatch
5. **MPS gate application**: `swap_sites_with_compress` at 76.1% cumulative

## Execution Path Breakdown

```
dense_vector_tn_qu (1.316s total)
  ├── CircuitMPS.from_openqasm2_str (1.20s, 91%)
  │   └── Per gate: swap_sites_with_compress
  │       └── svd_truncated
  │           └── np.linalg.svd (gesdd)
  ├── psi.full_simplify("DRC") (0.08s, 6%)
  └── psi.to_dense (0.04s, 3%)
```

## Bottleneck Conclusion

The MPS circuit construction dominates (91% of runtime). Within that, SVD operations on bond matrices consume the largest fraction. The bottleneck is algorithmic — each two-qubit gate requires at least one SVD, and gates are applied sequentially.