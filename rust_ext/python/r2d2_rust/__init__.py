"""Rust-accelerated R2D2 voxelwise metrics.

The compiled kernel (``r2d2_rust._r2d2_rust.compute_r2d2``) operates purely on
numpy arrays. This package adds a thin, ANTs-aware wrapper
(:func:`compute_r2d2_rust`) that mirrors the signature and return value of
``r2d2_numba.compute_r2d2_numba`` so it can be used as a drop-in replacement:

    from r2d2_rust import compute_r2d2_rust as compute_r2d2

It can also be used array-in/array-out, with no ANTs dependency at all:

    import numpy as np, r2d2_rust
    MI, MSE, CORR = r2d2_rust.compute_r2d2(reg_arr, tmplt_arr, mask_arr, radius=3)
"""

import numpy as np

from ._r2d2_rust import compute_r2d2  # noqa: F401  (re-exported low-level kernel)

__all__ = ["compute_r2d2", "compute_r2d2_rust"]

_KEYS = ("MI", "MSE", "CORR")


def compute_r2d2_rust(
    image_dict: dict,
    radius: int = 3,
    subsess: str = "unknown",
    bins: int = None,
    compute_mi: bool = True,
    use_sat: bool = True,
    mi_method: str = "approx",
) -> dict:
    """Compute R2D2 metrics from ANTs images, returning ANTs images.

    Drop-in for ``compute_r2d2`` / ``compute_r2d2_numba``.

    Args:
        image_dict: dict with ``template_image``, ``reg_image``, ``template_mask``
            (ANTsImage values).
        radius: neighborhood half-width (window side = 2*radius + 1).
        subsess: subject/session id, used only in error messages.
        bins: number of histogram bins for MI. Defaults to 50 for
            mi_method="mattes" (ITK's default) and 32 for "approx".
        compute_mi: if False, MI comes back as zeros (cheaper).
        use_sat: if True (default), use the summed-area-table kernel so MSE/CORR
            cost O(1) per voxel regardless of radius. False uses the direct
            per-window kernel (validation/reference).
        mi_method: "approx" (default, fast histogram MI) or "mattes"
            (ITK-faithful Mattes MI). Both return natural, non-negative MI
            (higher = more similar); this is -1x ANTs' metric value.

    Returns:
        dict keyed by MI, MSE, CORR; each value is an ANTsImage carrying the
        template's geometry.

    Note:
        All metrics use natural math signs: MSE >= 0 (0 = identical), CORR is
        Pearson (+1 = identical), MI >= 0 (higher = more shared information).
        mi_method="approx" is a fast histogram MI; "mattes" reproduces ITK's
        Mattes math but returned with the natural (positive) sign, i.e. -1x
        ANTs' metric value. At the default 50 bins the mattes magnitude equals
        ants.image_similarity(..., sampling_strategy="none") (dense) to floating
        point; ANTs' default sampling ("regular") differs slightly.
    """
    import ants  # local import: kernel itself needs no ANTs

    # ITK's Mattes default is 50 bins; the approx histogram MI uses 32.
    if bins is None:
        bins = 50 if mi_method == "mattes" else 32

    template_image = image_dict.get("template_image")
    reg_image = image_dict.get("reg_image")
    template_mask = image_dict.get("template_mask")

    # ANTs -> contiguous float64 numpy (the kernel expects f64).
    reg_arr = np.ascontiguousarray(reg_image.numpy(), dtype=np.float64)
    tmplt_arr = np.ascontiguousarray(template_image.numpy(), dtype=np.float64)
    mask_arr = np.ascontiguousarray(template_mask.numpy(), dtype=np.float64)

    try:
        arrays = compute_r2d2(
            reg_arr,
            tmplt_arr,
            mask_arr,
            int(radius),
            int(bins),
            bool(compute_mi),
            bool(use_sat),
            str(mi_method),
        )
    except Exception as e:  # surface which subject failed, then re-raise
        print(f"r2d2_rust failed on {subsess}: {e}")
        raise

    # Re-wrap each volume with the template's geometry.
    results = {}
    for key, arr in zip(_KEYS, arrays):
        results[key] = ants.from_numpy(
            arr,
            origin=template_image.origin,
            spacing=template_image.spacing,
            direction=template_image.direction,
        )
    return results
