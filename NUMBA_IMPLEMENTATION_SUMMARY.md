# Numba Implementation Summary

## What Was Created

I've developed a complete Numba-accelerated implementation of R2M2 with three new files:

### 1. `r2m2_numba.py` - Core Implementation

The main Numba-accelerated module with:

- **`compute_r2m2_kernel()`**: Numba JIT-compiled kernel with parallel execution
  - Uses `@jit(nopython=True, parallel=True)` for maximum performance
  - Implements MSE, Correlation, and approximate MI in pure numpy
  - Parallelizes across the X dimension using `prange`

- **`compute_r2m2_numba()`**: Drop-in replacement for `compute_r2m2()`
  - Same API as original function
  - Two modes: full Numba (`use_numba_mi=True`) or hybrid (`use_numba_mi=False`)
  - Returns ANTs images for compatibility

- **Helper functions**:
  - `compute_mse()`: Mean Squared Error in pure numpy
  - `compute_correlation()`: Pearson correlation in pure numpy
  - `compute_mutual_information_approx()`: Fast histogram-based MI approximation
  - `compute_mi_with_ants()`: Optimized ANTs MI computation (hybrid mode)

### 2. `benchmark_numba.py` - Performance Testing

Comprehensive benchmark script that:

- Creates synthetic test data matching MNI152 dimensions
- Compares three implementations:
  1. Original (optional, very slow)
  2. Full Numba with approximate MI
  3. Hybrid (Numba for MSE/CORR, ANTs for MI)
- Reports timing and speedup
- Validates accuracy by comparing results

### 3. `test_numba_quick.py` - Smoke Test

Quick validation script that:

- Tests basic functionality on small data (20×20×20)
- Verifies output structure and data validity
- Tests both Numba and hybrid modes
- Provides clear pass/fail results

### 4. `NUMBA_OPTIMIZATION.md` - Documentation

Complete usage guide covering:

- Installation instructions
- Three usage patterns (drop-in, direct call, pipeline integration)
- Performance benchmarking
- Accuracy considerations
- Parallelism configuration
- Troubleshooting guide

### 5. Updated `requirements.txt`

Added:
- `numba>=0.56.0` for JIT compilation
- `scipy>=1.7.0` for benchmark utilities

## Expected Performance Improvements

| Metric | Original | Numba (approx MI) | Numba (hybrid) |
|--------|----------|-------------------|----------------|
| **Time** (91×109×91, r=3) | 300-600s | 10-30s | 30-60s |
| **Speedup** | 1x | **15-40x** | **8-15x** |
| **MI Accuracy** | Exact | Approximate | Exact |
| **MSE/CORR Accuracy** | Exact | Exact | Exact |

## Key Optimizations Implemented

1. **JIT Compilation**: Numba compiles Python to native machine code
2. **Parallelization**: Uses `prange` for multi-core execution
3. **Reduced Overhead**: Eliminates Python→ANTs boundary crossings in inner loop
4. **Pre-allocation**: Allocates output arrays once instead of repeatedly
5. **Masked-voxel filtering**: Only processes voxels within the mask
6. **Vectorized metrics**: Pure numpy implementations avoid function call overhead

## How to Use

### Quick Start (After Installing Dependencies)

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test (validates implementation)
python test_numba_quick.py

# Run benchmark (compares performance)
python benchmark_numba.py

# Use in your code
python
from r2m2_numba import compute_r2m2_numba

# Fastest mode (approximate MI)
results = compute_r2m2_numba(image_dict, radius=3, use_numba_mi=True)

# Recommended mode (ANTs MI for accuracy)
results = compute_r2m2_numba(image_dict, radius=3, use_numba_mi=False)
```

### Integration Options

#### Option A: Minimal Change (Recommended)

Add to `r2m2_base.py`:

```python
# At top of file
try:
    from r2m2_numba import compute_r2m2_numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

# In main() function, replace compute_r2m2() call:
if USE_NUMBA:
    r2m2 = compute_r2m2_numba(img_dict, radius=radius, subsess=subsess, use_numba_mi=False)
else:
    r2m2 = compute_r2m2(img_dict, radius=radius, subsess=subsess)
```

This provides automatic fallback if Numba isn't installed.

#### Option B: Command-Line Flag

Add a `--use-numba` flag to `get_args()`:

```python
parser.add_argument(
    "--use-numba",
    action="store_true",
    help="Use Numba-accelerated implementation (10-40x faster)"
)
```

#### Option C: Environment Variable

```python
USE_NUMBA = os.environ.get('R2M2_USE_NUMBA', 'true').lower() == 'true'
```

## Testing Status

⚠️ **Not yet tested** - ANTs is not currently installed in your environment.

To test:

```bash
# Install ANTs
pip install antspyx

# Run quick test
python test_numba_quick.py

# Run benchmark
python benchmark_numba.py
```

## Next Steps

1. **Install ANTs**: `pip install antspyx` (may take a while)
2. **Run tests**: Validate the implementation works
3. **Benchmark**: Measure actual speedup on your data
4. **Validate accuracy**: Compare results on a few test subjects
5. **Integrate**: Add to your pipeline using one of the options above
6. **Tune parallelism**: Adjust `NUMBA_NUM_THREADS` for your system

## Advanced: Parallelism Configuration

For optimal performance, balance subject-level and voxel-level parallelism:

```bash
# Example: 8-core machine, processing 20 subjects

# Option 1: Subject-level parallelism (better for many subjects)
export NUMBA_NUM_THREADS=2
python r2m2_base.py --list_path subjects.txt --num_python_jobs 4
# Total: 4 subjects × 2 threads = 8 cores

# Option 2: Voxel-level parallelism (better for few subjects)
export NUMBA_NUM_THREADS=8
python r2m2_base.py --list_path subjects.txt --num_python_jobs 1
# Total: 1 subject × 8 threads = 8 cores
```

## Future Optimization Paths

If you need even more speed:

1. **GPU acceleration** (CuPy/JAX): 100-500x speedup, requires GPU
2. **Julia rewrite**: 50-100x speedup, no Python interop overhead
3. **Spatial subsampling**: 5-10x speedup, some precision loss
4. **Integral images**: Faster for large radii (r > 5)

## Questions or Issues?

Common issues and solutions are documented in `NUMBA_OPTIMIZATION.md`.

The implementation is designed to be:
- **Conservative**: Hybrid mode uses ANTs MI for accuracy
- **Compatible**: Same API as original function
- **Flexible**: Easy to switch between modes
- **Robust**: Comprehensive error handling

You can start with hybrid mode (`use_numba_mi=False`) for safety, then switch to approximate MI mode (`use_numba_mi=True`) after validating accuracy for your specific use case.
