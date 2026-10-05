# Environment Setup

**Date:** 2026-10-04

## Hardware

- AMD Ryzen 7 7840HS (8C/16T, Zen 4)
- 16 GB RAM (15.5 GB WSL visible)
- CPU-only (no GPU in WSL)
- WSL2, Ubuntu 26.04 LTS

## Software

- Python 3.10.21 (miniforge3 conda)
- QiboTN 0.0.7 (editable install from ~/HPC/qibotn)
- Qibo 0.3.4
- Quimb 1.11.2
- NumPy 2.2.6 (OpenBLAS 0.3.29 bundled)
- SciPy 1.15.3
- Numba 0.61.2

## Setup Commands

```bash
# Activate conda environment
source ~/miniforge3/bin/activate qibotn

# Editable install of QiboTN
cd ~/HPC/qibotn
pip install -e .

# Verify
python -c "import qibotn; print(qibotn.__version__, qibotn.__file__)"
# Expected: 0.0.7 ...HPC/qibotn/src/qibotn/__init__.py
```

## Key Configuration

- Python constraint relaxed: `>=3.10,<3.14` (was `>=3.11`)
- Backend: quimb (numpy), CPU-only
- No CUDA, no cuQuantum, no MPI

## Running Benchmarks

```bash
# Small circuit test
python scripts/quick_test.py

# Profiling
python scripts/profile_12q.py

# Full validation
python scripts/validate_complete.py

# SVD analysis
python scripts/svd_scaling_and_compare.py
```

## Reproducibility

All experiments use `np.random.seed(42)` for deterministic circuit generation. Scripts are self-contained and produce CSV output in `results/final/`.