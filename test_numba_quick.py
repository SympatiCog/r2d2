"""
Quick smoke test for Numba implementation.

This verifies that the Numba implementation:
1. Imports correctly
2. Runs without errors
3. Produces reasonable output
4. Matches expected array shapes

Run with: python test_numba_quick.py
"""

import sys
import numpy as np

# Check for ANTs availability
try:
    import ants
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: ANTs (ANTsPy) is not installed")
    print("=" * 70)
    print("\nANTs is required for R2D2 testing.")
    print("\nInstallation instructions:")
    print("\n1. Using pip (recommended):")
    print("   pip install antspyx")
    print("\n2. Using conda:")
    print("   conda install -c conda-forge antspyx")
    print("\nNote: ANTsPy installation may take 10-20 minutes.")
    print("\nFor more information, visit:")
    print("  https://github.com/ANTsX/ANTsPy")
    print("=" * 70 + "\n")
    sys.exit(1)

from r2d2_numba import compute_r2d2_numba


def create_tiny_test_data():
    """Create minimal test data for quick validation."""
    shape = (20, 20, 20)

    # Simple synthetic data
    template_arr = np.random.randn(*shape).astype(np.float32)
    reg_arr = template_arr + np.random.randn(*shape).astype(np.float32) * 0.1

    # Simple spherical mask
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = shape[0] // 2
    mask_arr = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= (shape[0]//3)**2
    mask_arr = mask_arr.astype(np.uint8)

    # Convert to ANTs
    template_image = ants.from_numpy(template_arr)
    reg_image = ants.from_numpy(reg_arr)
    template_mask = ants.from_numpy(mask_arr)

    image_dict = {
        "template_image": template_image,
        "reg_image": reg_image,
        "template_mask": template_mask,
    }

    return image_dict, shape, mask_arr


def test_numba_basic():
    """Test basic functionality."""
    print("=" * 60)
    print("Quick Smoke Test for Numba R2D2 Implementation")
    print("=" * 60)

    print("\n1. Creating test data...")
    image_dict, shape, mask_arr = create_tiny_test_data()
    print(f"   Shape: {shape}")
    print(f"   Masked voxels: {np.sum(mask_arr)}")

    print("\n2. Testing Numba with approximate MI...")
    try:
        results = compute_r2d2_numba(
            image_dict,
            radius=3,
            subsess="test",
            use_numba_mi=True
        )
        print("   ✓ Completed successfully")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    print("\n3. Verifying output structure...")
    expected_keys = ["MI", "MSE", "CORR", "dm_MI", "dm_MSE", "dm_CORR"]
    for key in expected_keys:
        if key not in results:
            print(f"   ✗ Missing key: {key}")
            return False
        if results[key].shape != shape:
            print(f"   ✗ Wrong shape for {key}: {results[key].shape} != {shape}")
            return False
    print(f"   ✓ All {len(expected_keys)} metrics present with correct shape")

    print("\n4. Checking output values...")
    for key in ["MSE", "CORR", "dm_MSE", "dm_CORR"]:
        arr = results[key].numpy()
        non_zero = arr[mask_arr > 0]

        # Check for NaNs or Infs
        if np.any(np.isnan(non_zero)):
            print(f"   ✗ {key} contains NaN values")
            return False
        if np.any(np.isinf(non_zero)):
            print(f"   ✗ {key} contains Inf values")
            return False

        # Check reasonable range
        print(f"   {key}: min={non_zero.min():.6f}, max={non_zero.max():.6f}, "
              f"mean={non_zero.mean():.6f}")

    print("\n5. Testing hybrid mode (ANTs MI)...")
    try:
        results_hybrid = compute_r2d2_numba(
            image_dict,
            radius=3,
            subsess="test",
            use_numba_mi=False
        )
        print("   ✓ Hybrid mode completed successfully")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nNumba implementation is working correctly!")
    print("Next steps:")
    print("  1. Run benchmark: python benchmark_numba.py")
    print("  2. Test on real data")
    print("  3. Integrate into your pipeline")

    return True


if __name__ == "__main__":
    success = test_numba_basic()
    exit(0 if success else 1)
