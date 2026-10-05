import numpy as np
import os, sys

print('=== SVD Backend Analysis ===')
print()

x = np.random.rand(100, 100)

print('1. NumPy config:')
np.show_config()

print('\n2. Thread pool info:')
try:
    from threadpoolctl import threadpool_info
    for p in threadpool_info():
        print(f'  prefix={p["prefix"]}, api={p["user_api"]}, threads={p["num_threads"]}, internal={p["internal_api"]}')
except:
    print('  threadpoolctl not available')

print('\n3. Environment variables:')
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    print(f'  {v}={os.environ.get(v, "(not set)")}')

print('\n4. LDD on numpy linalg module:')
import numpy.linalg
linalg_path = numpy.linalg.__file__
print(f'  numpy.linalg at: {linalg_path}')

print('\n5. SVD function identity:')
print(f'  np.linalg.svd is: {np.linalg.svd}')
print(f'  type: {type(np.linalg.svd)}')

print('\n6. Testing SVD with timer probe:')
import time
t0 = time.perf_counter()
U, s, Vh = np.linalg.svd(x, full_matrices=False)
t1 = time.perf_counter()
print(f'  100x100 SVD: {t1-t0:.6f}s')
print(f'  U shape: {U.shape}, s shape: {s.shape}, Vh shape: {Vh.shape}')

# Check if scipy gesvd produces same result
try:
    from scipy import linalg as scla
    t0 = time.perf_counter()
    U2, s2, Vh2 = scla.svd(x, full_matrices=False, lapack_driver='gesvd')
    t1 = time.perf_counter()
    print(f'\n  scipy gesvd 100x100: {t1-t0:.6f}s')
    print(f'  diff vs numpy: {np.max(np.abs(s - s2)):.2e}')
    
    # Also test gesdd
    t0 = time.perf_counter()
    U3, s3, Vh3 = scla.svd(x, full_matrices=False, lapack_driver='gesdd')
    t1 = time.perf_counter()
    print(f'  scipy gesdd 100x100: {t1-t0:.6f}s')
    print(f'  diff gesdd vs numpy: {np.max(np.abs(s - s3)):.2e}')
except Exception as e:
    print(f'\n  scipy SVD test failed: {e}')

print('\n7. Quimb SVD call path:')
print('  quimb.tensor.decomp.svd_truncated')
print('    -> svd_truncated_numpy (registered for numpy arrays)')
print('    -> svd_truncated_numba')
print('    -> np.linalg.svd(x, full_matrices=False)')
print('    -> NumPy uses LAPACK ?gesdd via OpenBLAS')
print('    -> _trim_and_renorm_svd_result_numba (Numba JIT for truncation)')
print('  Fallback (on ValueError): scipy.linalg.svd with lapack_driver="gesvd"')

print('\nDONE')