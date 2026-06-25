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

MSE and Correlation match ANTs/numpy to floating point. MI offers two methods —
a fast histogram approximation (default) and a faithful ITK-Mattes
reimplementation matching `ants.image_similarity` (see "Mutual information").

## Layout

```
rust_ext/                   # crate directory (Python import name is `r2d2_rust`)
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
cd rust_ext
maturin develop --release

# Or build a redistributable wheel
maturin build --release
pip install target/wheels/r2d2_rust-*.whl
```

> **macOS note:** `cargo test` / `cargo build` link via `dynamic_lookup` (set in
> `.cargo/config.toml`) so the `extension-module` feature — which doesn't link
> libpython — doesn't fail the macOS linker with `Undefined symbols ... _Py...`.
> This is handled automatically; no action needed. `maturin build`/`develop`
> apply the same flags, and Linux/Windows ignore them.

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

### Pipeline integration

`r2d2_base.py` and `r2d2_numba.py` already detect this extension and prefer it
when installed, falling back to their pure-Python / Numba kernels otherwise.
Both expose a `--backend` flag:

```bash
# auto (default): Rust if installed, else the module's own kernel
python r2d2_base.py  --search_string './sub-*/reg.nii.gz' --template_path tmpl.nii.gz
python r2d2_base.py  ... --backend rust    # require the Rust extension
python r2d2_base.py  ... --backend python  # force pure Python
python r2d2_numba.py ... --backend numba   # force the Numba kernel
```

When the Rust backend is used, MI is computed with `mi_method="mattes"` so its
sign convention matches the ANTs-based pipelines (negative = more similar).

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

Two MI methods, selected with `mi_method`:

- **`"approx"`** (default) — a fast histogram MI matching
  `r2d2_numba.compute_mutual_information_approx`. Positive; cheap.
- **`"mattes"`** — a faithful reimplementation of ITK's
  `MattesMutualInformationImageToImageMetric`, the metric ANTs uses for
  `metric_type="MattesMutualInformation"`. It uses B-spline Parzen windowing
  (zero-order for the fixed/template image, cubic spread over four bins for the
  moving/registered image) and ITK's `bins - 2*padding` bin layout with two
  guard bins per side. Like ANTs it returns the **metric value**, i.e. the
  *negative* mutual information (lower = more similar), so its sign is opposite
  the approximation's.

```python
# ANTs-faithful MI (negative; lower = more similar)
MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(
    reg, tmplt, mask, radius=3, bins=32, mi_method="mattes"
)
```

Both methods are shift-invariant per window, so `dm_MI == MI`.

`bins` is the number of histogram bins; for `"mattes"` it must be > 4 (two guard
bins on each side). MI cannot use the summed-area-table shortcut — it needs the
per-window joint histogram — so it is recomputed per window regardless of
`use_sat`.

### Validation

`mattes_mutual_information` is checked against an independent pure-Python
reimplementation of the same ITK algorithm (`test_mattes_matches_python_reference`,
exact to 1e-9) and structurally (negative metric, shift-invariance, ranks
identical > independent). `tests/test_kernel.py` also includes an **opt-in**
parity test against ANTs itself (`test_mattes_matches_ants_if_available`),
skipped unless ANTsPy is installed — run it on a machine with ANTs to confirm
end-to-end agreement. Exact agreement depends on matching the bin count and
ANTs' sampling settings.

## Possible next steps

- f32 input support to halve memory traffic.
- Parallelize the prefix-sum build (currently a single O(voxels) pass).
- Optionally match ANTs' default sampling for the Mattes metric if sub-1e-2
  parity is needed.
