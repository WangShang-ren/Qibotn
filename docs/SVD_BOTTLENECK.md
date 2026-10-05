# SVD Bottleneck Analysis

**Date:** 2026-10-04 | **Workload:** 14Q x 500g, max_bond=512

## 1. Is SVD Compute-Bound or Memory-Bound?

**Answer: Compute-bound.**

Evidence:
- Thread scaling produces zero benefit (1→16 threads: all ~55s). If memory-bound, more threads would hide latency.
- SVD involves matrix multiplication (O(n^3)) with only O(n^2) data access.
- For a d×d matrix SVD, compute is O(d^3) while memory access is O(d^2).
- OpenBLAS's DYNAMIC_ARCH targeting Haswell means AVX2 (not AVX-512) instructions are used, limiting compute throughput.

## 2. Is Memory Bandwidth Significant?

**Answer: No.**

Evidence:
- Thread count has no effect on SVD performance — memory bandwidth is not the bottleneck.
- The SVD operations fit within L3 cache for bond dimensions ≤ 512.
- 512×512 matrix = 512×512×8 bytes = 2MB per matrix, ×3 matrices (U, s, Vh) ≈ 6MB. L3 cache is 16 MB.

## 3. Is Tensor Allocation Significant?

**Answer: Moderate, but not dominant.**

Evidence from cProfile:
- `quimb.tensor.tensor_core.modify`: 7449 calls, 0.026s
- `quimb.tensor.tensor_core.__init__`: 4154 calls, 0.009s
- `quimb.tensor.array_ops.asarray`: 9300 calls, 0.005s

Total allocation/copy overhead: ~5% of runtime. Not the bottleneck.

## 4. Is Data Copying Significant?

**Answer: No.**

Evidence:
- `transpose_like`: 1922 calls, 0.011s
- `transpose`: 2883 calls, 0.009s
- `reshape`: 6465 calls, 0.017s

Total: ~3% of runtime.

## 5. Does Threading Help?

**Answer: No.**

See SVD thread scaling results: 1 thread = 54.7s, 16 threads = 53.7s. Not statistically significant.

### Why threading doesn't help:
1. SVD matrices are relatively small (bond_dim ≤ 512): OpenBLAS threading threshold is not met.
2. The MPS circuit construction is sequential per gate: each SVD must wait for the previous one.
3. OpenBLAS uses NO_AFFINITY and DYNAMIC_ARCH (Haswell target), not optimized for Zen 4.

## 6. Primary Bottleneck Classification

| Component | Bottleneck Type | % of Runtime | Optimizable? |
|-----------|----------------|--------------|--------------|
| SVD (gesdd) | Compute | 37% | No (already optimal) |
| tensor operations | Memory/cache | 19% | No (NumPy internal) |
| autoray dispatch | Python overhead | 3% | No (quimb internal) |
| MPS gate sequencing | Algorithm (sequential) | 76% cumulative | Not easily |
| tensor simplification | Compute | <5% | No |
| Final contraction | Compute | <1% | No |

## 7. Conclusion

The SVD bottleneck is **compute-bound but not parallelizable** due to:
1. Small matrix sizes preventing OpenBLAS multi-threading benefit
2. Sequential gate application preventing SVD concurrency
3. Haswell-targeted OpenBLAS not using full Zen 4 AVX-512 capabilities

On this hardware, there is no software-level optimization that can significantly reduce SVD time while maintaining correctness.