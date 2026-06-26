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

/// The three metric volumes returned for a single subject.
struct R2d2Output {
    mi: Array3<f64>,
    mse: Array3<f64>,
    corr: Array3<f64>,
}

/// Per-voxel result, scattered back into the output volumes after the parallel
/// pass so the hot loop never needs synchronized writes.
struct VoxelResult {
    idx: usize,
    mi: f64,
    mse: f64,
    corr: f64,
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

/// Cubic (third-order) B-spline kernel, as used by ITK for the moving-image
/// Parzen window. Even function; support [-2, 2]; partition of unity.
#[inline]
fn bspline3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0
    } else if a < 2.0 {
        let b = 2.0 - a;
        b * b * b / 6.0
    } else {
        0.0
    }
}

/// Mattes mutual information, faithfully reproducing ITK's
/// `MattesMutualInformationImageToImageMetric` math (the metric ANTs uses for
/// `metric_type="MattesMutualInformation"`).
///
/// Differences from the fast histogram approximation:
///   - B-spline Parzen windowing: the fixed image uses a zero-order window
///     (one bin per sample); the moving image uses a cubic window spread over
///     four bins.
///   - ITK's bin layout: `bin_size = range / (bins - 2*padding)` with
///     `padding = 2`, so two guard bins on each side keep the cubic window in
///     range.
///   - Returns the ITK **metric value**, i.e. the *negative* mutual
///     information (lower = more similar), matching `ants.image_similarity`.
///
/// `fixed` is the template, `moving` the registered image — the same argument
/// order as `ants.image_similarity(template, reg, ...)`.
///
/// Because the per-window bin layout rescales by each window's own min/max, a
/// constant intensity shift leaves the result unchanged (shift-invariant), so
/// the demeaned variant equals the raw one — as with the approximation.
fn mattes_mutual_information(
    fixed: &ArrayView3<f64>,
    moving: &ArrayView3<f64>,
    bins: usize,
) -> f64 {
    const PADDING: usize = 2;
    let n = fixed.len();
    // Need at least one real bin between the guard bins.
    if n == 0 || bins <= 2 * PADDING {
        return 0.0;
    }

    let (mut fmin, mut fmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut mmin, mut mmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in fixed.iter() {
        fmin = fmin.min(v);
        fmax = fmax.max(v);
    }
    for &v in moving.iter() {
        mmin = mmin.min(v);
        mmax = mmax.max(v);
    }
    // A flat window carries no mutual information.
    if fmax == fmin || mmax == mmin {
        return 0.0;
    }

    let usable = (bins - 2 * PADDING) as f64;
    let f_bin_size = (fmax - fmin) / usable;
    let m_bin_size = (mmax - mmin) / usable;
    let f_norm_min = fmin / f_bin_size - PADDING as f64;
    let m_norm_min = mmin / m_bin_size - PADDING as f64;

    // joint[fixed_index * bins + moving_index]
    let mut joint = vec![0.0f64; bins * bins];
    let lo = 2usize;
    let hi = bins - 3; // clamp range so the cubic window [idx-1, idx+2] is valid

    for (&fv, &mv) in fixed.iter().zip(moving.iter()) {
        let f_term = fv / f_bin_size - f_norm_min;
        let f_index = (f_term.floor() as usize).clamp(lo, hi);

        let m_term = mv / m_bin_size - m_norm_min;
        let m_index = (m_term.floor() as usize).clamp(lo, hi);
        let m_arg = m_term - m_index as f64;

        // Cubic Parzen weights for moving bins [m_index-1 .. m_index+2].
        let w = [
            bspline3(m_arg + 1.0),
            bspline3(m_arg),
            bspline3(m_arg - 1.0),
            bspline3(m_arg - 2.0),
        ];
        let base = f_index * bins + (m_index - 1);
        joint[base] += w[0];
        joint[base + 1] += w[1];
        joint[base + 2] += w[2];
        joint[base + 3] += w[3];
    }

    // Normalize to a joint PDF.
    let total: f64 = joint.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let inv = 1.0 / total;
    for p in joint.iter_mut() {
        *p *= inv;
    }

    // Marginals (row sums = fixed, column sums = moving).
    let mut fixed_marg = vec![0.0f64; bins];
    let mut moving_marg = vec![0.0f64; bins];
    for i in 0..bins {
        for j in 0..bins {
            let p = joint[i * bins + j];
            fixed_marg[i] += p;
            moving_marg[j] += p;
        }
    }

    const EPS: f64 = 1e-16;
    let mut mi = 0.0;
    for i in 0..bins {
        let fpv = fixed_marg[i];
        if fpv <= EPS {
            continue;
        }
        let log_fpv = fpv.ln();
        for j in 0..bins {
            let jpv = joint[i * bins + j];
            let mpv = moving_marg[j];
            if jpv > EPS && mpv > EPS {
                // jpv * log( jpv / (fpv * mpv) )
                mi += jpv * ((jpv / mpv).ln() - log_fpv);
            }
        }
    }

    // ITK metric convention: return the negative mutual information.
    -mi
}

/// Dispatch the per-window MI computation by method.
#[inline]
fn mi_window(
    win_tmplt: &ArrayView3<f64>,
    win_reg: &ArrayView3<f64>,
    bins: usize,
    mattes: bool,
) -> f64 {
    if mattes {
        mattes_mutual_information(win_tmplt, win_reg, bins)
    } else {
        mutual_information_approx(win_tmplt, win_reg, bins)
    }
}
fn compute_r2d2_kernel(
    reg: ArrayView3<f64>,
    tmplt: ArrayView3<f64>,
    mask: ArrayView3<f64>,
    radius: usize,
    bins: usize,
    compute_mi: bool,
    mattes: bool,
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

            // Metrics: template vs registered, matching the Python arg order.
            let mse_raw = mse(&win_tmplt, &win_reg, 0.0, 0.0);
            let corr_raw = correlation(&win_tmplt, &win_reg);
            let mi_raw = if compute_mi {
                mi_window(&win_tmplt, &win_reg, bins, mattes)
            } else {
                0.0
            };

            VoxelResult {
                idx: x * ny * nz + y * nz + z,
                mi: mi_raw,
                mse: mse_raw,
                corr: corr_raw,
            }
        })
        .collect();

    scatter(nx, ny, nz, results)
}

/// Scatter per-voxel results into three zero-initialized volumes. Shared by both
/// the direct and summed-area-table kernels.
fn scatter(nx: usize, ny: usize, nz: usize, results: Vec<VoxelResult>) -> R2d2Output {
    let mut out = R2d2Output {
        mi: Array3::zeros((nx, ny, nz)),
        mse: Array3::zeros((nx, ny, nz)),
        corr: Array3::zeros((nx, ny, nz)),
    };

    // Scatter back into the volumes via flat (C-order) slices.
    let mi = out.mi.as_slice_mut().unwrap();
    let mse_s = out.mse.as_slice_mut().unwrap();
    let corr = out.corr.as_slice_mut().unwrap();
    for r in results {
        mi[r.idx] = r.mi;
        mse_s[r.idx] = r.mse;
        corr[r.idx] = r.corr;
    }

    out
}

/// Build a 3D summed-area table (prefix sum) of shape (nx+1, ny+1, nz+1) where
/// `p[i,j,k]` is the sum of `f` over the half-open box [0,i) x [0,j) x [0,k).
fn build_sat<F: Fn(usize, usize, usize) -> f64>(
    nx: usize,
    ny: usize,
    nz: usize,
    f: F,
) -> Array3<f64> {
    let mut p = Array3::<f64>::zeros((nx + 1, ny + 1, nz + 1));
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                let v = f(i - 1, j - 1, k - 1);
                p[[i, j, k]] = v
                    + p[[i - 1, j, k]]
                    + p[[i, j - 1, k]]
                    + p[[i, j, k - 1]]
                    - p[[i - 1, j - 1, k]]
                    - p[[i - 1, j, k - 1]]
                    - p[[i, j - 1, k - 1]]
                    + p[[i - 1, j - 1, k - 1]];
            }
        }
    }
    p
}

/// Sum of the underlying values over the half-open window
/// [x0,x1) x [y0,y1) x [z0,z1), via 8-corner inclusion-exclusion. O(1).
#[inline]
fn window_sum(
    p: &Array3<f64>,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    z0: usize,
    z1: usize,
) -> f64 {
    p[[x1, y1, z1]] - p[[x0, y1, z1]] - p[[x1, y0, z1]] - p[[x1, y1, z0]]
        + p[[x0, y0, z1]]
        + p[[x0, y1, z0]]
        + p[[x1, y0, z0]]
        - p[[x0, y0, z0]]
}

/// Summed-area-table kernel: MSE/Correlation (and their demeaned variants) in
/// O(1) per voxel regardless of `radius`, instead of O(radius^3) per voxel.
///
/// MI cannot use prefix sums — it needs the per-window joint histogram — so it
/// still extracts each window when `compute_mi` is set. The big win is for
/// `compute_mi=false` (or large radius), where the whole pass becomes O(voxels).
fn compute_r2d2_kernel_sat(
    reg: ArrayView3<f64>,
    tmplt: ArrayView3<f64>,
    mask: ArrayView3<f64>,
    radius: usize,
    bins: usize,
    compute_mi: bool,
    mattes: bool,
) -> R2d2Output {
    let (nx, ny, nz) = reg.dim();

    // Center each image by its global mean before squaring/multiplying, so the
    // prefix sums stay well-conditioned (centered values are near zero, which
    // avoids catastrophic cancellation in var = E[x^2] - E[x]^2). Raw MSE is
    // restored exactly via the mean-difference term below.
    let n_total = (nx * ny * nz).max(1) as f64;
    let cr = reg.iter().sum::<f64>() / n_total;
    let ct = tmplt.iter().sum::<f64>() / n_total;
    let dmean = ct - cr;

    // Five prefix sums over the centered values: sum r', sum t', sum r'^2,
    // sum t'^2, sum r'*t'.
    let sat_r = build_sat(nx, ny, nz, |i, j, k| reg[[i, j, k]] - cr);
    let sat_t = build_sat(nx, ny, nz, |i, j, k| tmplt[[i, j, k]] - ct);
    let sat_rr = build_sat(nx, ny, nz, |i, j, k| {
        let v = reg[[i, j, k]] - cr;
        v * v
    });
    let sat_tt = build_sat(nx, ny, nz, |i, j, k| {
        let v = tmplt[[i, j, k]] - ct;
        v * v
    });
    let sat_rt =
        build_sat(nx, ny, nz, |i, j, k| (reg[[i, j, k]] - cr) * (tmplt[[i, j, k]] - ct));

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

            let n = ((x1 - x0) * (y1 - y0) * (z1 - z0)) as f64;
            let a = window_sum(&sat_r, x0, x1, y0, y1, z0, z1); // sum r'
            let b = window_sum(&sat_t, x0, x1, y0, y1, z0, z1); // sum t'
            let arr = window_sum(&sat_rr, x0, x1, y0, y1, z0, z1); // sum r'^2
            let btt = window_sum(&sat_tt, x0, x1, y0, y1, z0, z1); // sum t'^2
            let crt = window_sum(&sat_rt, x0, x1, y0, y1, z0, z1); // sum r'*t'

            let mean_r = a / n;
            let mean_t = b / n;
            let var_r = (arr / n - mean_r * mean_r).max(0.0);
            let var_t = (btt / n - mean_t * mean_t).max(0.0);
            let cov = crt / n - mean_r * mean_t;

            // Raw MSE = mean((t - r)^2). With t = t' + ct and r = r' + cr,
            // t - r = (t' - r') + dmean, so expand the square.
            let mse_raw =
                (btt - 2.0 * crt + arr) / n + 2.0 * dmean * (b - a) / n + dmean * dmean;

            let corr = if var_r > 0.0 && var_t > 0.0 {
                (cov / (var_r.sqrt() * var_t.sqrt())).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // MI still needs the actual window (no prefix-sum shortcut).
            let mi_raw = if compute_mi {
                let win_reg = reg.slice(s![x0..x1, y0..y1, z0..z1]);
                let win_tmplt = tmplt.slice(s![x0..x1, y0..y1, z0..z1]);
                mi_window(&win_tmplt, &win_reg, bins, mattes)
            } else {
                0.0
            };

            VoxelResult {
                idx: x * ny * nz + y * nz + z,
                mi: mi_raw,
                mse: mse_raw.max(0.0),
                corr,
            }
        })
        .collect();

    scatter(nx, ny, nz, results)
}

/// Compute the six R2D2 metric volumes for one subject.
///
/// Args (all 3D float64 arrays of identical shape):
///     reg:    registered image
///     tmplt:  template image
///     mask:   template mask (voxels == 1 are processed)
///     radius: neighborhood half-width (window side = 2*radius + 1)
///     bins:   histogram bins for MI (default 32; for "mattes" this is the
///             number of ITK histogram bins, must be > 4)
///     compute_mi: if False, skip MI and return zeros for MI/dm_MI
///     use_sat: if True (default), use the summed-area-table kernel — MSE/CORR
///              become O(1) per voxel regardless of radius. If False, use the
///              direct per-window kernel (handy for validation).
///     mi_method: "approx" (default; fast positive histogram MI) or "mattes"
///              (ITK-faithful Mattes MI, returns the negative-MI metric value
///              matching ants.image_similarity).
///
/// Returns a 3-tuple of float64 arrays:
///     (MI, MSE, CORR)
#[pyfunction]
#[pyo3(signature = (reg, tmplt, mask, radius, bins=32, compute_mi=true, use_sat=true, mi_method="approx"))]
#[allow(clippy::too_many_arguments)]
fn compute_r2d2<'py>(
    py: Python<'py>,
    reg: PyReadonlyArray3<'py, f64>,
    tmplt: PyReadonlyArray3<'py, f64>,
    mask: PyReadonlyArray3<'py, f64>,
    radius: usize,
    bins: usize,
    compute_mi: bool,
    use_sat: bool,
    mi_method: &str,
) -> PyResult<(
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

    let mattes = match mi_method {
        "approx" => false,
        "mattes" => true,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mi_method must be 'approx' or 'mattes', got '{other}'"
            )))
        }
    };
    if mattes && compute_mi && bins <= 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mattes MI needs bins > 4 (two guard bins on each side)",
        ));
    }

    // Own the data so the kernel can run with the GIL released.
    let reg = reg.as_array().to_owned();
    let tmplt = tmplt.as_array().to_owned();
    let mask = mask.as_array().to_owned();

    let out = py.allow_threads(move || {
        if use_sat {
            compute_r2d2_kernel_sat(
                reg.view(),
                tmplt.view(),
                mask.view(),
                radius,
                bins,
                compute_mi,
                mattes,
            )
        } else {
            compute_r2d2_kernel(
                reg.view(),
                tmplt.view(),
                mask.view(),
                radius,
                bins,
                compute_mi,
                mattes,
            )
        }
    });

    Ok((
        out.mi.into_pyarray_bound(py),
        out.mse.into_pyarray_bound(py),
        out.corr.into_pyarray_bound(py),
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

        let out = compute_r2d2_kernel(reg.view(), tmplt.view(), mask.view(), 1, 32, true, false);

        // Identical inputs => zero MSE at the masked voxel, and untouched
        // (zero) everywhere else.
        assert_eq!(out.mse[[2, 2, 2]], 0.0);
        assert_eq!(out.mse[[0, 0, 0]], 0.0);
        // Correlation of identical non-constant windows is ~1 at the masked voxel.
        assert!((out.corr[[2, 2, 2]] - 1.0).abs() < 1e-9);
        // Unmasked voxel stays at the initialized zero.
        assert_eq!(out.corr[[0, 0, 0]], 0.0);
    }

    #[test]
    fn sat_matches_direct_kernel() {
        // Deterministic, non-trivial data so variance/covariance are nonzero.
        let n = 12;
        let reg = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            ((i * 7 + j * 13 + k * 17) % 23) as f64 + 0.5 * (i as f64) - 0.25 * (k as f64)
        });
        let tmplt = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            ((i * 5 + j * 11 + k * 3) % 19) as f64 - 0.3 * (j as f64)
        });
        let mask = Array3::from_elem((n, n, n), 1.0);
        let radius = 3;

        let direct =
            compute_r2d2_kernel(reg.view(), tmplt.view(), mask.view(), radius, 32, true, false);
        let sat = compute_r2d2_kernel_sat(
            reg.view(),
            tmplt.view(),
            mask.view(),
            radius,
            32,
            true,
            false,
        );

        let max_diff = |a: &Array3<f64>, b: &Array3<f64>| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max)
        };

        // SAT computes the same MSE/CORR via prefix sums (looser tol for the
        // different summation order), and MI identically (same per-window code).
        assert!(max_diff(&sat.mse, &direct.mse) < 1e-9, "MSE mismatch");
        assert!(max_diff(&sat.corr, &direct.corr) < 1e-9, "CORR mismatch");
        assert!(max_diff(&sat.mi, &direct.mi) < 1e-12, "MI mismatch");
    }

    #[test]
    fn bspline3_is_partition_of_unity() {
        // The four cubic weights around any fractional offset sum to 1.
        for step in 0..10 {
            let arg = step as f64 / 10.0; // in [0, 1)
            let s = bspline3(arg + 1.0) + bspline3(arg) + bspline3(arg - 1.0) + bspline3(arg - 2.0);
            assert!((s - 1.0).abs() < 1e-12, "weights sum to {s}");
        }
    }

    #[test]
    fn mattes_mi_ranks_similarity() {
        // Identical images share maximal information -> most-negative metric;
        // an independent pairing -> metric near zero. So identical < independent.
        let n = 10;
        let a = Array3::from_shape_fn((n, n, n), |(i, j, k)| ((i * 3 + j * 5 + k * 7) % 11) as f64);
        let b_indep =
            Array3::from_shape_fn((n, n, n), |(i, j, k)| ((i * 13 + j * 2 + k * 17) % 11) as f64);

        let same = mattes_mutual_information(&a.view(), &a.view(), 16);
        let indep = mattes_mutual_information(&a.view(), &b_indep.view(), 16);

        assert!(same <= 0.0, "metric should be <= 0, got {same}");
        assert!(same < indep, "identical ({same}) should beat independent ({indep})");
    }

    #[test]
    fn mattes_mi_is_shift_invariant() {
        // Adding a constant to all moving voxels leaves the metric unchanged,
        // because the per-window bin layout rescales by the window's own range.
        let n = 8;
        let f = Array3::from_shape_fn((n, n, n), |(i, j, k)| ((i + 2 * j + 3 * k) % 7) as f64);
        let m = Array3::from_shape_fn((n, n, n), |(i, j, k)| ((2 * i + j + k) % 5) as f64);
        let m_shift = &m + 123.456;

        let base = mattes_mutual_information(&f.view(), &m.view(), 16);
        let shifted = mattes_mutual_information(&f.view(), &m_shift.view(), 16);
        assert!((base - shifted).abs() < 1e-9, "shift changed MI: {base} vs {shifted}");
    }

    #[test]
    fn mattes_mi_flat_window_is_zero() {
        let f = Array3::from_elem((4, 4, 4), 5.0); // constant
        let m = Array3::from_shape_fn((4, 4, 4), |(i, _, _)| i as f64);
        assert_eq!(mattes_mutual_information(&f.view(), &m.view(), 16), 0.0);
    }
}
