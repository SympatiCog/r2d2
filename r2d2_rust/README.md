# r2d2_rust

Rust-accelerated kernel for the voxelwise R2D2 (Regional Registration Mismatch
Metric) computation. It replaces the triple-nested Python loop in
`compute_r2d2` with a parallel Rust kernel exposed to Python via
[PyO3](https://pyo3.rs/) + [maturin](https://www.maturin.rs/).

The kernel operates purely on numpy arrays (no ANTs dependency); a thin Python
wrapper handles ANTs image <-> numpy conversion so it stays a drop-in for the
existing pipeline.

## Why Rust (vs. the Numba path)

`r2d2_numba.py` already JIT-accelerates the same loop. Rust adds:

- **No JIT warmup** — compiled ahead of time, so no per-process compile cost.
- **Redistributable wheels** — `pip install` a precompiled binary; users need
  no Rust toolchain, no LLVM, no Numba at runtime.
- **Tiny dependency surface** for use from other scripts: `import r2d2_rust`
  pulls in only numpy.
- **True threads** — the loop runs with the GIL released via rayon, composing
  with Python-side process/thread parallelism.

On top of that, the default kernel uses **summed-area tables (integral images)**
so MSE and Correlation cost O(1) per voxel *regardless of radius* — this stacks
with the language speedup (see "Summed-area-table kernel" below).

MSE and Correlation match ANTs/numpy to floating point. MI uses the same fast
histogram approximation as the Numba path (see "Mutual information" below).

## Layout

```
r2d2_rust/
├── Cargo.toml              # Rust crate (cdylib + rlib)
├── pyproject.toml          # maturin build config
├── src/lib.rs              # kernel + PyO3 bindings + Rust unit tests
├── python/r2d2_rust/       # Python package (ANTs-aware wrapper)
│   └── __init__.py
├── tests/test_kernel.py    # validation vs. a pure-numpy reference
└── benchmark_sat.py        # SAT vs. direct-kernel timing
```

## Build & install

```bash
# Dev install into the active virtualenv/conda env
cd r2d2_rust
maturin develop --release

# Or build a redistributable wheel
maturin build --release
pip install target/wheels/r2d2_rust-*.whl
```

## Usage

Array-in / array-out (no ANTs):

```python
import numpy as np, r2d2_rust
# reg/tmplt/mask are 3D float64 numpy arrays of identical shape
MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(
    reg, tmplt, mask, radius=3, bins=32, compute_mi=True, use_sat=True
)
```

Drop-in for `compute_r2d2` / `compute_r2d2_numba` (ANTs images in/out):

```python
from r2d2_rust import compute_r2d2_rust as compute_r2d2
results = compute_r2d2(image_dict, radius=3)   # dict of ANTsImages
```

To wire it into `r2d2_base.py`:

```python
try:
    from r2d2_rust import compute_r2d2_rust as compute_r2d2
except ImportError:
    pass  # fall back to the pure-Python / Numba implementation
```

## Testing

```bash
# Rust kernel unit tests
cargo test --release

# Python validation (MSE/CORR must match the numpy reference exactly)
maturin develop --release
pytest tests/test_kernel.py -v
```

## Summed-area-table kernel

The default kernel (`use_sat=True`) builds five 3D prefix-sum tables
(`sum r`, `sum t`, `sum r²`, `sum t²`, `sum r·t`) once, then derives every
window's MSE, Correlation, and demeaned variants from eight corner lookups —
O(1) per voxel instead of O(radius³). Each image is centered by its global mean
before squaring so the variance/covariance stay numerically stable; raw MSE is
restored exactly via a mean-difference term.

MI is the one metric that can't use prefix sums (it needs the per-window joint
histogram), so it still extracts each window when `compute_mi=True`. The SAT win
is largest with `compute_mi=False` or large radius. Pass `use_sat=False` to fall
back to the direct per-window kernel (used as the validation reference).

Benchmark (`benchmark_sat.py`, 91×109×91 volume, ~650k masked voxels,
`compute_mi=False`):

| radius | direct (s) | SAT (s) | speedup |
|-------:|-----------:|--------:|--------:|
| 2 | 0.54 | 0.13 | 4.1x |
| 3 | 1.24 | 0.13 | 9.8x |
| 5 | 4.44 | 0.12 | 36.5x |
| 8 | 15.37 | 0.13 | 117.2x |

SAT wall-clock is flat in radius; the direct kernel grows ~radius³. (Numbers
will vary with core count.)

## Mutual information

`compute_r2d2` computes a histogram-based **approximation** of MI, matching
`r2d2_numba.compute_mutual_information_approx` — not ANTs' Parzen-windowed
Mattes MI. Because each window is rescaled to its own range, the approximation
is shift-invariant, so the demeaned MI equals the raw MI here.

If you need exact ANTs agreement, two paths:

1. **Hybrid**: run the Rust kernel with `compute_mi=False` and compute MI/dm_MI
   separately with ANTs (see `r2d2_numba.compute_mi_with_ants`), substituting
   those two volumes.
2. **Native Mattes**: implement Parzen-windowed Mattes MI in `src/lib.rs`. This
   is the one metric that needs care for bit-for-bit ANTs parity.

## Possible next steps

- Native Mattes MI for exact ANTs parity (the one metric still computed
  per-window).
- f32 input support to halve memory traffic.
- Parallelize the prefix-sum build (currently a single O(voxels) pass).
