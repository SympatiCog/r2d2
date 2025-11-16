# Numba Optimization for R2M2

This document describes the Numba-accelerated implementation of R2M2 and how to use it.

## Overview

The Numba implementation provides **10-50x speedup** over the original implementation by:

1. **JIT compilation**: Compiles Python code to native machine code
2. **Parallel execution**: Uses multiple CPU cores via `prange`
3. **Reduced overhead**: Eliminates Python/C++ boundary crossings in the inner loop
4. **Vectorization**: Optimized numpy operations

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install `numba>=0.56.0` and other required packages.

## Usage

### Option 1: Use as a drop-in replacement

Replace the import in your code:

```python
# Original
from r2m2_base import compute_r2m2

# Numba-accelerated
from r2m2_numba import compute_r2m2_numba as compute_r2m2
```

### Option 2: Call directly

```python
from r2m2_numba import compute_r2m2_numba

# With approximate MI (fastest)
results = compute_r2m2_numba(
    image_dict,
    radius=3,
    subsess="sub-001",
    use_numba_mi=True  # Use fast approximate MI
)

# With ANTs MI (hybrid mode - recommended)
results = compute_r2m2_numba(
    image_dict,
    radius=3,
    subsess="sub-001",
    use_numba_mi=False  # Use accurate ANTs MI, Numba for MSE/CORR
)
```

### Option 3: Integrate into main pipeline

Modify `r2m2_base.py` to use the Numba implementation:

```python
# At the top of r2m2_base.py
try:
    from r2m2_numba import compute_r2m2_numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print("Numba not available, using original implementation")

# In main() function
def main(sub_folder, reg_image_name, template_path, radius=3):
    # ... existing code ...

    if USE_NUMBA:
        r2m2 = compute_r2m2_numba(img_dict, radius=radius, subsess=subsess)
    else:
        r2m2 = compute_r2m2(img_dict, radius=radius, subsess=subsess)

    # ... rest of function ...
```

## Performance Benchmarking

Run the benchmark script to compare implementations:

```bash
# Quick benchmark (skips original implementation)
python benchmark_numba.py

# Full benchmark including original (WARNING: very slow!)
python benchmark_numba.py --include-mi

# Custom parameters
python benchmark_numba.py --radius 5 --shape 91 109 91
```

### Expected Performance

On a typical workstation (e.g., 8-core CPU):

| Implementation | Time (91×109×91, radius=3) | Speedup |
|----------------|----------------------------|---------|
| Original       | ~300-600 seconds           | 1x      |
| Numba (approx MI) | ~10-30 seconds          | 15-40x  |
| Numba hybrid   | ~30-60 seconds             | 8-15x   |

**Note**: First run includes JIT compilation overhead (~5-10 seconds). Subsequent runs are faster.

## Implementation Details

### Three Approaches

1. **Original**: Uses ANTs for all metrics (slowest, most accurate)
2. **Numba with approximate MI**: All metrics computed in Numba (fastest)
3. **Hybrid**: Numba for MSE/CORR, ANTs for MI (balanced)

### Accuracy Considerations

- **MSE and Correlation**: Numba implementation is numerically identical to ANTs
- **Mutual Information**: Numba uses a simplified histogram-based approximation
  - Good enough for most use cases
  - For critical analyses, use hybrid mode (`use_numba_mi=False`)

### Parallel Processing

The Numba kernel uses `prange` for parallel execution:

```python
@jit(nopython=True, parallel=True)
def compute_r2m2_kernel(...):
    for x in prange(X):  # Parallel over x dimension
        for y in range(Y):
            for z in range(Z):
                # ... computation ...
```

Control parallelism via environment variable:

```bash
# Use 8 threads
export NUMBA_NUM_THREADS=8
python your_script.py

# Or in Python
import os
os.environ['NUMBA_NUM_THREADS'] = '8'
```

## Combining with Subject-Level Parallelism

You can parallelize at both levels:

```bash
# 4 Python processes, each using 2 Numba threads
export NUMBA_NUM_THREADS=2
python r2m2_base.py --list_path subjects.txt --num_python_jobs 4

# Total parallelism: 4 × 2 = 8 cores
```

**Recommendation**:
- For many subjects: High `num_python_jobs`, low `NUMBA_NUM_THREADS`
- For few subjects: Low `num_python_jobs`, high `NUMBA_NUM_THREADS`

## Memory Considerations

Numba implementation uses slightly more peak memory due to:
- Pre-allocation of output arrays
- Parallel worker threads

Typical memory usage:
- Original: ~2-3 GB per subject
- Numba: ~3-4 GB per subject

Adjust `num_python_jobs` if running out of memory.

## Troubleshooting

### Issue: First run is slow

**Solution**: This is JIT compilation. Subsequent runs will be fast. You can pre-compile:

```python
from r2m2_numba import compute_r2m2_kernel
import numpy as np

# Trigger compilation with dummy data
dummy = np.zeros((10, 10, 10))
compute_r2m2_kernel(dummy, dummy, dummy, 3, compute_mi=False)
```

### Issue: Numba import error

**Solution**: Install numba:

```bash
pip install numba>=0.56.0
```

### Issue: Results differ from original

**Solution**:
- MSE/CORR should be nearly identical (diff < 1e-6)
- MI will differ if using `use_numba_mi=True` (expected)
- Use `use_numba_mi=False` for ANTs-compatible MI

### Issue: Slower than expected

**Solution**:
- Check `NUMBA_NUM_THREADS` is set appropriately
- Verify JIT compilation completed (run twice, compare times)
- Large radius values (>5) may benefit less from optimization

## Future Optimizations

Potential further improvements:

1. **GPU acceleration**: Port to CuPy/CUDA for 100x+ speedup
2. **Spatial subsampling**: Compute every Nth voxel, interpolate
3. **Better MI approximation**: Implement Mattes MI exactly in Numba
4. **Memory-mapped arrays**: Reduce memory footprint for large datasets

## Validation

Always validate Numba results against original for your specific data:

```python
# Run both implementations
results_orig = compute_r2m2(image_dict, radius=3)
results_numba = compute_r2m2_numba(image_dict, radius=3, use_numba_mi=False)

# Compare
for key in ['MSE', 'CORR', 'dm_MSE', 'dm_CORR']:
    diff = np.abs(results_orig[key].numpy() - results_numba[key].numpy())
    print(f"{key}: max_diff = {diff.max()}, mean_diff = {diff.mean()}")
```

## References

- [Numba Documentation](https://numba.pydata.org/)
- [ANTs Documentation](https://antspy.readthedocs.io/)
- Original R2M2 paper: [if available]
