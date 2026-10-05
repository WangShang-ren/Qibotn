# SVD Backend Analysis

**Date:** 2026-10-04

## 1. Call Path

```
qibotn.eval_qu_optimized.dense_vector_tn_qu_optimized()
  -> quimb.tensor.circuit.CircuitMPS.from_openqasm2_str()
    -> gate_opts["method"] = "svd"
      -> quimb.tensor.tensor_1d.MPS.gate_()
        -> swap_sites_with_compress()
          -> quimb.tensor.decomp.svd_truncated()
            -> svd_truncated_numpy (registered for numpy backend)
              -> svd_truncated_numba()
                -> numpy.linalg.svd(x, full_matrices=False)  [DOMINANT]
                -> _trim_and_renorm_svd_result_numba()        [truncation]
              -> [fallback on ValueError]
                -> scipy.linalg.svd(lapack_driver="gesvd")
```

## 2. Library Stack

| Layer | Component | Details |
|-------|-----------|---------|
| Python | `np.linalg.svd` | NumPy 2.2.6 |
| C extension | `_umath_linalg` | NumPy's LAPACK wrapper |
| LAPACK routine | `?gesdd` | Divide-and-conquer SVD |
| BLAS/LAPACK library | **OpenBLAS 0.3.29** | scipy-openblas64 bundled |
| Threading | OpenBLAS internal | DYNAMIC_ARCH, NO_AFFINITY |

## 3. Evidence

- **Source code**: `quimb/tensor/decomp.py:348` — `U, s, VH = np.linalg.svd(x, full_matrices=False)`
- **NumPy config**: OpenBLAS 0.3.29, Haswell target, MAX_THREADS=64
- **Registered dispatch**: `@svd_truncated.register("numpy")` → `svd_truncated_numpy`
- **Profiling**: `svd_truncated_numba` = 36.8% of total runtime (cProfile on 12Qx300g)

## 4. Threading Behavior

Measured on 14Q x 500g, max_bond=512:

| OMP Threads | Mean Runtime (s) | Min (s) | Fidelity |
|-------------|-------------------|---------|----------|
| 1 | 54.7 | 53.3 | 1.000000 |
| 2 | 56.0 | 53.5 | 1.000000 |
| 4 | 56.9 | 55.4 | 1.000000 |
| 8 | 57.7 | 56.6 | 1.000000 |
| 16 | 53.7 | 52.2 | 1.000000 |

**Conclusion**: No significant thread scaling effect. The variance within each group (3-7s) exceeds the mean difference between groups. The SVD matrix sizes (bond_dim × bond_dim) are too small for OpenBLAS multi-threading to provide benefit.

## 5. SVD Runtime Share

From cProfile on 12Q x 300g (1.3s total runtime):
- `svd_truncated_numba`: 0.485s (36.8% of total)
- `svd_truncated_numpy`: 0.004s (cumulative includes numba)
- Total SVD-related: ~37%

The remaining time is in tensor reshaping, autoray dispatch, and gate application logic.

## 6. Alternative SVD Comparison

Benchmarked on representative matrix sizes (10 reps each):

| Matrix | NumPy gesdd (s) | SciPy gesdd (s) | SciPy gesvd (s) |
|--------|-----------------|------------------|-----------------|
| 64x64 | 0.0020 | 0.0094 | 0.0021 |
| 128x128 | 0.0768 | 0.0166 | 0.0223 |
| 256x256 | 0.0345 | 0.0511 | 0.2536 |
| 512x512 | 0.3764 | 0.2980 | 2.4867 |

**Conclusion**: NumPy gesdd is the fastest implementation for the matrix sizes encountered. The quimb scipy fallback (gesvd) is slower. No faster alternative exists within the current environment.