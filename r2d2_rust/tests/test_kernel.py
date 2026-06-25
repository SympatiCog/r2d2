"""Validate the Rust kernel against a pure-numpy reference.

Run after building/installing the extension (e.g. `maturin develop --release`):

    pytest r2d2_rust/tests/test_kernel.py -v

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


def test_mse_and_corr_match_reference():
    reg, tmplt, mask = _random_volumes()
    radius = 3
    MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(
        reg, tmplt, mask, radius
    )
    ref_mse, ref_corr, ref_dm_mse = _reference(reg, tmplt, mask, radius)

    np.testing.assert_allclose(MSE, ref_mse, atol=1e-10)
    np.testing.assert_allclose(dm_MSE, ref_dm_mse, atol=1e-10)
    np.testing.assert_allclose(CORR, ref_corr, atol=1e-10)


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


if __name__ == "__main__":
    # Allow running without pytest.
    reg, tmplt, mask = _random_volumes()
    MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR = r2d2_rust.compute_r2d2(reg, tmplt, mask, 3)
    ref_mse, ref_corr, ref_dm_mse = _reference(reg, tmplt, mask, 3)
    print("MSE  max abs diff:", np.abs(MSE - ref_mse).max())
    print("CORR max abs diff:", np.abs(CORR - ref_corr).max())
    print("dm_MSE max abs diff:", np.abs(dm_MSE - ref_dm_mse).max())
    print("OK")
