# QiboTN Environment Audit

**Generated:** 2026-10-03
**Project:** ASC Supercomputing Team Selection - QiboTN Performance Optimization

---

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 7 7840HS (Zen 4) |
| Physical Cores | 8 |
| Logical Threads | 16 |
| L1d Cache | 256 KiB x 8 instances |
| L1i Cache | 256 KiB x 8 instances |
| L2 Cache | 8 MiB x 8 instances |
| L3 Cache | 16 MiB (shared) |
| SIMD | SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2/AVX/AVX2/FMA3/AVX-512 |
| Memory | 15.5 GB (WSL visible), ~14 GB available |
| Swap | 4 GB |
| NUMA | 1 node |
| GPU | None |

## Software

| Component | Version | Source |
|-----------|---------|--------|
| OS | Ubuntu 26.04 LTS | WSL2 |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 | Microsoft |
| Python | 3.10.21 | conda-forge |
| QiboTN (repo) | 0.0.7 | GitHub qiboteam/qibotn |
| Qibo | 0.3.4 | PyPI |
| qibojit | 0.1.16 | PyPI |
| Quimb | 1.11.2 | PyPI |
| NumPy | 2.2.6 | conda-forge |
| SciPy | 1.15.3 | conda-forge |
| Numba | 0.61.2 | conda-forge |
| BLAS | OpenBLAS 0.3.29 | scipy-openblas (bundled) |

## Repository

| Field | Value |
|-------|-------|
| Path | ~/HPC/qibotn |
| Remote | https://github.com/qiboteam/qibotn.git |
| Branch | main |
| HEAD | 3aeae48 |
| Status | Clean (untracked: try/) |

## Python Environment

- Manager: conda (miniforge3), env: qibotn
- Install: editable (`pip install -e .`)
- Python constraint: relaxed to >=3.10 for conda env

## Available Backends

- cutensornet: No (no GPU)
- quimb (numpy): YES - primary
- quimb (torch/jax): No
- qmatchatea: No