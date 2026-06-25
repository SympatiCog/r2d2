//! Rust-accelerated voxelwise R2D2 metrics.
//!
//! The kernel mirrors `compute_r2d2` / `compute_r2d2_numba`: for every masked
//! voxel it crops an NxNxN neighborhood (radius `r`, so side `2r+1`, clipped at
//! the image borders) out of the registered and template volumes and computes
//! six local similarity metrics.
//!
//! Only numpy arrays cross the Python boundary; all NIfTI/ANTs I/O stays in
//! Python. The heavy loop runs over masked voxels in parallel via rayon, with
//! the GIL released, so it composes with process- or thread-level parallelism
//! on the Python side.

use ndarray::{s, Array3, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

/// The six metric volumes returned for a single subject.
struct R2d2Output {
    mi: Array3<f64>,
    mse: Array3<f64>,
    corr: Array3<f64>,
    dm_mi: Array3<f64>,
    dm_mse: Array3<f64>,
    dm_corr: Array3<f64>,
}

/// Per-voxel result, scattered back into the output volumes after the parallel
/// pass so the hot loop never needs synchronized writes.
struct VoxelResult {
    idx: usize,
    mi: f64,
    mse: f64,
    corr: f64,
    dm_mi: f64,
    dm_mse: f64,
    dm_corr: f64,
}

/// Mean of (a-b)^2 over the window, with optional per-array offsets subtracted
/// first (used for the demeaned variant). Returns 0.0 for an empty window.
fn mse(a: &ArrayView3<f64>, b: &ArrayView3<f64>, off_a: f64, off_b: f64) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - off_a) - (y - off_b);
        sum += d * d;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Mean of the window's elements (0.0 if empty).
fn mean(a: &ArrayView3<f64>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for &x in a.iter() {
        sum += x;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Pearson correlation coefficient over the window, in [-1, 1].
///
/// Correlation is invariant to a constant shift, so the demeaned variant is
/// numerically identical — the caller reuses this single result for both.
fn correlation(a: &ArrayView3<f64>, b: &ArrayView3<f64>) -> f64 {
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let cx = x - ma;
        let cy = y - mb;
        num += cx * cy;
        da += cx * cx;
        db += cy * cy;
    }
    if da == 0.0 || db == 0.0 {
        0.0
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

/// Histogram-based approximation of mutual information, matching the strategy in
/// `r2d2_numba.compute_mutual_information_approx`.
///
/// NOTE: this is *not* ANTs' Parzen-windowed Mattes MI. It is a fast, drop-in
/// approximation. Because each array is rescaled to its own [0, bins-1] range,
/// a constant offset cancels out, so the demeaned MI equals the raw MI here;
/// the caller reuses one result for both. Replace with a proper Mattes
/// implementation if exact ANTs agreement is required.
fn mutual_information_approx(a: &ArrayView3<f64>, b: &ArrayView3<f64>, bins: usize) -> f64 {
    let n = a.len();
    if n == 0 || bins == 0 {
        return 0.0;
    }

    let (mut min_a, mut max_a) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_b, mut max_b) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in a.iter() {
        min_a = min_a.min(x);
        max_a = max_a.max(x);
    }
    for &y in b.iter() {
        min_b = min_b.min(y);
        max_b = max_b.max(y);
    }

    // Degenerate (flat) window -> no shared information.
    if max_a == min_a || max_b == min_b {
        return 0.0;
    }

    let scale_a = (bins - 1) as f64 / (max_a - min_a);
    let scale_b = (bins - 1) as f64 / (max_b - min_b);

    let mut joint = vec![0.0f64; bins * bins];
    for (&x, &y) in a.iter().zip(b.iter()) {
        let ia = (((x - min_a) * scale_a) as usize).min(bins - 1);
        let ib = (((y - min_b) * scale_b) as usize).min(bins - 1);
        joint[ia * bins + ib] += 1.0;
    }

    let total = n as f64;
    let mut p_x = vec![0.0f64; bins];
    let mut p_y = vec![0.0f64; bins];
    for i in 0..bins {
        for j in 0..bins {
            let p = joint[i * bins + j] / total;
            joint[i * bins + j] = p;
            p_x[i] += p;
            p_y[j] += p;
        }
    }

    let mut mi = 0.0;
    for i in 0..bins {
        if p_x[i] <= 0.0 {
            continue;
        }
        for j in 0..bins {
            let p = joint[i * bins + j];
            if p > 0.0 && p_y[j] > 0.0 {
                mi += p * (p / (p_x[i] * p_y[j])).ln();
            }
        }
    }
    mi
}

/// Pure-Rust kernel (no Python types) so it can be unit-tested directly.
fn compute_r2d2_kernel(
    reg: ArrayView3<f64>,
    tmplt: ArrayView3<f64>,
    mask: ArrayView3<f64>,
    radius: usize,
    bins: usize,
    compute_mi: bool,
) -> R2d2Output {
    let (nx, ny, nz) = reg.dim();

    // Collect masked voxel indices up front so rayon can split a flat workload.
    let coords: Vec<(usize, usize, usize)> = mask
        .indexed_iter()
        .filter(|(_, &m)| m == 1.0)
        .map(|((x, y, z), _)| (x, y, z))
        .collect();

    let results: Vec<VoxelResult> = coords
        .par_iter()
        .map(|&(x, y, z)| {
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let z0 = z.saturating_sub(radius);
            let x1 = (x + radius + 1).min(nx);
            let y1 = (y + radius + 1).min(ny);
            let z1 = (z + radius + 1).min(nz);

            let win_reg = reg.slice(s![x0..x1, y0..y1, z0..z1]);
            let win_tmplt = tmplt.slice(s![x0..x1, y0..y1, z0..z1]);

            // Raw metrics (template vs registered, matching the Python arg order).
            let mse_raw = mse(&win_tmplt, &win_reg, 0.0, 0.0);
            let corr_raw = correlation(&win_tmplt, &win_reg);
            let mi_raw = if compute_mi {
                mutual_information_approx(&win_tmplt, &win_reg, bins)
            } else {
                0.0
            };

            // Demeaned metrics: MSE changes; CORR and approx-MI are shift-invariant.
            let mean_reg = mean(&win_reg);
            let mean_tmplt = mean(&win_tmplt);
            let dm_mse = mse(&win_tmplt, &win_reg, mean_tmplt, mean_reg);

            VoxelResult {
                idx: x * ny * nz + y * nz + z,
                mi: mi_raw,
                mse: mse_raw,
                corr: corr_raw,
                dm_mi: mi_raw,
                dm_mse,
                dm_corr: corr_raw,
            }
        })
        .collect();

    let mut out = R2d2Output {
        mi: Array3::zeros((nx, ny, nz)),
        mse: Array3::zeros((nx, ny, nz)),
        corr: Array3::zeros((nx, ny, nz)),
        dm_mi: Array3::zeros((nx, ny, nz)),
        dm_mse: Array3::zeros((nx, ny, nz)),
        dm_corr: Array3::zeros((nx, ny, nz)),
    };

    // Scatter back into the volumes via flat (C-order) slices.
    let mi = out.mi.as_slice_mut().unwrap();
    let mse_s = out.mse.as_slice_mut().unwrap();
    let corr = out.corr.as_slice_mut().unwrap();
    let dm_mi = out.dm_mi.as_slice_mut().unwrap();
    let dm_mse = out.dm_mse.as_slice_mut().unwrap();
    let dm_corr = out.dm_corr.as_slice_mut().unwrap();
    for r in results {
        mi[r.idx] = r.mi;
        mse_s[r.idx] = r.mse;
        corr[r.idx] = r.corr;
        dm_mi[r.idx] = r.dm_mi;
        dm_mse[r.idx] = r.dm_mse;
        dm_corr[r.idx] = r.dm_corr;
    }

    out
}

/// Compute the six R2D2 metric volumes for one subject.
///
/// Args (all 3D float64 arrays of identical shape):
///     reg:    registered image
///     tmplt:  template image
///     mask:   template mask (voxels == 1 are processed)
///     radius: neighborhood half-width (window side = 2*radius + 1)
///     bins:   histogram bins for the approximate MI (default 32)
///     compute_mi: if False, skip MI and return zeros for MI/dm_MI
///
/// Returns a 6-tuple of float64 arrays:
///     (MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR)
#[pyfunction]
#[pyo3(signature = (reg, tmplt, mask, radius, bins=32, compute_mi=true))]
#[allow(clippy::too_many_arguments)]
fn compute_r2d2<'py>(
    py: Python<'py>,
    reg: PyReadonlyArray3<'py, f64>,
    tmplt: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray3<'py, f64>,
    radius: usize,
    bins: usize,
    compute_mi: bool,
) -> PyResult<(
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
)> {
    let reg_dim = reg.shape().to_vec();
    if tmplt.shape() != reg_dim.as_slice() || mask.shape() != reg_dim.as_slice() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "reg, tmplt, and mask must all have the same shape",
        ));
    }

    // Own the data so the kernel can run with the GIL released.
    let reg = reg.as_array().to_owned();
    let tmplt = tmplt.as_array().to_owned();
    let mask = mask.as_array().to_owned();

    let out = py.allow_threads(move || {
        compute_r2d2_kernel(
            reg.view(),
            tmplt.view(),
            mask.view(),
            radius,
            bins,
            compute_mi,
        )
    });

    Ok((
        out.mi.into_pyarray_bound(py),
        out.mse.into_pyarray_bound(py),
        out.corr.into_pyarray_bound(py),
        out.dm_mi.into_pyarray_bound(py),
        out.dm_mse.into_pyarray_bound(py),
        out.dm_corr.into_pyarray_bound(py),
    ))
}

#[pymodule]
fn _r2d2_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_r2d2, m)?)?;
    m.add("__doc__", "Rust-accelerated voxelwise R2D2 metric kernel.")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn mse_is_zero_for_identical_windows() {
        let a = Array3::from_elem((3, 3, 3), 2.0);
        let b = Array3::from_elem((3, 3, 3), 2.0);
        assert_eq!(mse(&a.view(), &b.view(), 0.0, 0.0), 0.0);
    }

    #[test]
    fn correlation_is_one_for_identical_windows() {
        // Non-constant data so the correlation is well-defined.
        let mut a = Array3::zeros((2, 2, 2));
        for (i, v) in a.iter_mut().enumerate() {
            *v = i as f64;
        }
        let b = a.clone();
        let c = correlation(&a.view(), &b.view());
        assert!((c - 1.0).abs() < 1e-12, "expected ~1.0, got {c}");
    }

    #[test]
    fn kernel_only_fills_masked_voxels() {
        let nx = 5;
        let reg = Array3::from_shape_fn((nx, nx, nx), |(i, j, k)| (i + j + k) as f64);
        let tmplt = reg.clone();
        let mut mask = Array3::zeros((nx, nx, nx));
        mask[[2, 2, 2]] = 1.0;

        let out = compute_r2d2_kernel(reg.view(), tmplt.view(), mask.view(), 1, 32, true);

        // Identical inputs => zero MSE at the masked voxel, and untouched
        // (zero) everywhere else.
        assert_eq!(out.mse[[2, 2, 2]], 0.0);
        assert_eq!(out.mse[[0, 0, 0]], 0.0);
        // Correlation of identical non-constant windows is ~1 at the masked voxel.
        assert!((out.corr[[2, 2, 2]] - 1.0).abs() < 1e-9);
        // Unmasked voxel stays at the initialized zero.
        assert_eq!(out.corr[[0, 0, 0]], 0.0);
    }
}
