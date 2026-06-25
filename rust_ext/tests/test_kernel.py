"""Validate the Rust kernel against a pure-numpy reference.

Run after building/installing the extension (e.g. `maturin develop --release`):

    pytest rust_ext/tests/test_kernel.py -v

MSE and Correlation must match the reference to floating-point tolerance.
MI uses a histogram approximation, so it is only sanity-checked (finite,
non-negative, and identical between the raw and demeaned variants).
"""

import numpy as np
import pytest

r2d2_rust = pytest.importorskip("r2d2_rust")


def _reference(reg, tmplt, mask, radius):
    """Naive per-voxel reference, mirroring the original Python loop."""
    nx, ny, nz = reg.shape
    mse = np.zeros_like(reg)
    corr = np.zeros_like(reg)
    dm_mse = np.zeros_like(reg)
    coords = np.argwhere(mask == 1)
    for x, y, z in coords:
        x0, x1 = max(0, x - radius), min(nx, x + radius + 1)
        y0, y1 = max(0, y - radius), min(ny, y + radius + 1)
        z0, z1 = max(0, z - radius), min(nz, z + radius + 1)
        wr = reg[x0:x1, y0:y1, z0:z1].ravel()
        wt = tmplt[x0:x1, y0:y1, z0:z1].ravel()
        mse[x, y, z] = np.mean((wt - wr) ** 2)
        dm_mse[x, y, z] = np.mean(((wt - wt.mean()) - (wr - wr.mean())) ** 2)
        if wt.std() > 0 and wr.std() > 0:
            corr[x, y, z] = np.corrcoef(wt, wr)[0, 1]
    return mse, corr, dm_mse


def _random_volumes(shape=(20, 18, 16), seed=0):
    rng = np.random.default_rng(seed)
    reg = rng.normal(size=shape)
    tmplt = reg + 0.3 * rng.normal(size=shape)  # correlated but not identical
    mask = np.zeros(shape)
    # Mask an interior block so windows hit both borders and interior.
    mask[2:-2, 2:-2, 2:-2] = 1.0
    return (
        np.ascontiguousarray(reg),
        np.ascontiguousarray(tmplt),
        np.ascontiguousarray(mask),
    )


@pytest.mark.parametrize("use_sat", [True, False])
def test_mse_and_corr_match_reference(use_sat):
    reg, tmplt, mask = _random_volumes()
    radius = 3
    MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(
        reg, tmplt, mask, radius, use_sat=use_sat
    )
    ref_mse, ref_corr, ref_dm_mse = _reference(reg, tmplt, mask, radius)

    # The SAT kernel sums in a different order, so allow a slightly looser tol.
    atol = 1e-8 if use_sat else 1e-10
    np.testing.assert_allclose(MSE, ref_mse, atol=atol)
    np.testing.assert_allclose(dm_MSE, ref_dm_mse, atol=atol)
    np.testing.assert_allclose(CORR, ref_corr, atol=atol)


def test_sat_matches_direct():
    """The SAT and direct kernels must agree on every metric."""
    reg, tmplt, mask = _random_volumes(seed=7)
    radius = 4
    sat = r2d2_rust.compute_r2d2(reg, tmplt, mask, radius, use_sat=True)
    direct = r2d2_rust.compute_r2d2(reg, tmplt, mask, radius, use_sat=False)
    for s, d in zip(sat, direct):
        np.testing.assert_allclose(s, d, atol=1e-8)


def test_sat_radius_independence():
    """SAT results are exact for any radius, including a radius that spans the
    whole volume (every window is the full image)."""
    reg, tmplt, mask = _random_volumes(shape=(10, 10, 10), seed=9)
    big_r = 50  # larger than the volume -> every window is the entire image
    _, MSE, CORR, _, _, _ = r2d2_rust.compute_r2d2(reg, tmplt, mask, big_r)
    ref_mse, ref_corr, _ = _reference(reg, tmplt, mask, big_r)
    np.testing.assert_allclose(MSE, ref_mse, atol=1e-8)
    np.testing.assert_allclose(CORR, ref_corr, atol=1e-8)


def test_corr_is_shift_invariant():
    reg, tmplt, mask = _random_volumes(seed=1)
    _, _, CORR, _, _, dm_CORR = r2d2_rust.compute_r2d2(reg, tmplt, mask, 2)
    # Correlation is invariant to a constant shift, so the demeaned variant
    # equals the raw one.
    np.testing.assert_allclose(CORR, dm_CORR, atol=1e-12)


def test_mi_is_finite_and_nonnegative():
    reg, tmplt, mask = _random_volumes(seed=2)
    MI, _, _, dm_MI, _, _ = r2d2_rust.compute_r2d2(reg, tmplt, mask, 3, bins=16)
    assert np.all(np.isfinite(MI))
    assert np.all(MI >= -1e-12)
    # Approx-MI is shift-invariant in this implementation.
    np.testing.assert_allclose(MI, dm_MI, atol=1e-12)


def test_compute_mi_false_zeros_mi():
    reg, tmplt, mask = _random_volumes(seed=3)
    MI, MSE, _, dm_MI, _, _ = r2d2_rust.compute_r2d2(
        reg, tmplt, mask, 2, compute_mi=False
    )
    assert np.all(MI == 0.0)
    assert np.all(dm_MI == 0.0)
    assert np.any(MSE != 0.0)  # other metrics still computed


def test_shape_mismatch_raises():
    reg, tmplt, mask = _random_volumes()
    with pytest.raises(ValueError):
        r2d2_rust.compute_r2d2(reg, tmplt, mask[:-1], 2)


def test_mi_method_validation():
    reg, tmplt, mask = _random_volumes()
    with pytest.raises(ValueError):
        r2d2_rust.compute_r2d2(reg, tmplt, mask, 2, mi_method="bogus")
    # mattes needs bins > 4
    with pytest.raises(ValueError):
        r2d2_rust.compute_r2d2(reg, tmplt, mask, 2, bins=4, mi_method="mattes")


# --------------------------------------------------------------------------
# Mattes MI: validate the Rust kernel against an independent Python reference
# of the same ITK algorithm, then (optionally) against ANTs itself.
# --------------------------------------------------------------------------

def _bspline3(t):
    a = abs(t)
    if a < 1.0:
        return (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0
    if a < 2.0:
        b = 2.0 - a
        return b * b * b / 6.0
    return 0.0


def _mattes_reference(fixed, moving, bins, padding=2):
    """Pure-Python reimplementation of ITK's Mattes MI (negative MI)."""
    f = np.asarray(fixed, dtype=np.float64).ravel()
    m = np.asarray(moving, dtype=np.float64).ravel()
    if f.size == 0 or bins <= 2 * padding:
        return 0.0
    fmin, fmax = f.min(), f.max()
    mmin, mmax = m.min(), m.max()
    if fmax == fmin or mmax == mmin:
        return 0.0

    usable = bins - 2 * padding
    f_bin = (fmax - fmin) / usable
    m_bin = (mmax - mmin) / usable
    f_nmin = fmin / f_bin - padding
    m_nmin = mmin / m_bin - padding

    joint = np.zeros((bins, bins))
    lo, hi = 2, bins - 3
    for fv, mv in zip(f, m):
        fi = int(np.clip(int(np.floor(fv / f_bin - f_nmin)), lo, hi))
        mterm = mv / m_bin - m_nmin
        mi_ = int(np.clip(int(np.floor(mterm)), lo, hi))
        marg = mterm - mi_
        for d, off in enumerate((1.0, 0.0, -1.0, -2.0)):
            joint[fi, mi_ - 1 + d] += _bspline3(marg + off)

    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    fmarg = joint.sum(axis=1)
    mmarg = joint.sum(axis=0)

    eps = 1e-16
    mi = 0.0
    for i in range(bins):
        if fmarg[i] <= eps:
            continue
        for j in range(bins):
            jpv = joint[i, j]
            if jpv > eps and mmarg[j] > eps:
                mi += jpv * (np.log(jpv / mmarg[j]) - np.log(fmarg[i]))
    return -mi


def _mattes_via_kernel(fixed_win, moving_win, bins):
    """Run a single window through the kernel by masking only the center voxel.

    The kernel's center voxel sees exactly ``fixed_win`` / ``moving_win`` when
    the window spans the whole array (radius >= size).
    """
    shape = fixed_win.shape
    mask = np.zeros(shape)
    c = tuple(s // 2 for s in shape)
    mask[c] = 1.0
    big_r = max(shape)
    MI, *_ = r2d2_rust.compute_r2d2(
        np.ascontiguousarray(moving_win, dtype=np.float64),   # reg = moving
        np.ascontiguousarray(fixed_win, dtype=np.float64),    # tmplt = fixed
        np.ascontiguousarray(mask),
        big_r,
        bins=bins,
        mi_method="mattes",
    )
    return MI[c]


def test_mattes_matches_python_reference():
    rng = np.random.default_rng(11)
    bins = 16
    for _ in range(5):
        fixed = rng.normal(size=(7, 7, 7))
        moving = fixed + 0.5 * rng.normal(size=(7, 7, 7))
        got = _mattes_via_kernel(fixed, moving, bins)
        want = _mattes_reference(fixed, moving, bins)
        assert abs(got - want) < 1e-9, f"{got} vs {want}"


def test_mattes_is_negative_and_shift_invariant():
    reg, tmplt, mask = _random_volumes(seed=5)
    MI, _, _, dm_MI, _, _ = r2d2_rust.compute_r2d2(
        reg, tmplt, mask, 3, bins=32, mi_method="mattes"
    )
    # ITK convention: metric is <= 0 on the masked voxels.
    assert np.all(MI[mask == 1] <= 1e-12)
    # Demeaning is a per-window constant shift -> Mattes MI unchanged.
    np.testing.assert_allclose(MI, dm_MI, atol=1e-12)


@pytest.mark.parametrize("use_sat", [True, False])
def test_mattes_sat_and_direct_agree(use_sat):
    reg, tmplt, mask = _random_volumes(seed=6)
    a = r2d2_rust.compute_r2d2(reg, tmplt, mask, 3, mi_method="mattes", use_sat=True)
    b = r2d2_rust.compute_r2d2(reg, tmplt, mask, 3, mi_method="mattes", use_sat=False)
    np.testing.assert_allclose(a[0], b[0], atol=1e-9)  # MI matches across kernels


def test_mattes_matches_ants_if_available():
    """Opt-in parity check against ANTs itself.

    Skipped unless ANTsPy is installed. The kernel evaluates the window densely,
    so the apples-to-apples ANTs comparison uses sampling_strategy="none" (also
    dense) at ITK's default 50 bins. Under those settings the two agree to
    floating point. ANTs' *default* sampling ("regular") subsamples and differs
    by a few percent — that's a sampling choice, not an algorithmic difference.
    """
    ants = pytest.importorskip("ants")
    bins = 50  # ITK's Mattes default
    for seed in range(5):
        rng = np.random.default_rng(seed)
        fixed = rng.normal(size=(9, 9, 9))
        moving = fixed + 0.4 * rng.normal(size=(9, 9, 9))

        got = _mattes_via_kernel(fixed, moving, bins)

        f_img = ants.from_numpy(np.ascontiguousarray(fixed))
        m_img = ants.from_numpy(np.ascontiguousarray(moving))
        want = ants.image_similarity(
            f_img, m_img, metric_type="MattesMutualInformation",
            sampling_strategy="none",
        )
        assert abs(got - want) < 1e-3, f"seed={seed} kernel={got} ants={want}"


if __name__ == "__main__":
    # Allow running without pytest.
    reg, tmplt, mask = _random_volumes()
    MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(reg, tmplt, mask, 3)
    ref_mse, ref_corr, ref_dm_mse = _reference(reg, tmplt, mask, 3)
    print("MSE  max abs diff:", np.abs(MSE - ref_mse).max())
    print("CORR max abs diff:", np.abs(CORR - ref_corr).max())
    print("dm_MSE max abs diff:", np.abs(dm_MSE - ref_dm_mse).max())
    print("OK")
