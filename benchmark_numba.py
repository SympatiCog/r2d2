"""
Benchmark script to compare original vs Numba-accelerated R2D2 implementation.

This script creates synthetic test data and compares:
1. Original implementation (r2d2_base.compute_r2d2)
2. Numba implementation with approximate MI (r2d2_numba.compute_r2d2_numba)
3. Hybrid implementation (Numba for MSE/CORR, ANTs for MI)

Usage:
    python benchmark_numba.py [--radius RADIUS] [--shape X Y Z]
"""

import sys
import time
import argparse
import numpy as np

# Check for ANTs availability
try:
    import ants
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: ANTs (ANTsPy) is not installed")
    print("=" * 70)
    print("\nANTs is required for R2D2 benchmarking.")
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

from r2d2_base import compute_r2d2
from r2d2_numba import compute_r2d2_numba


def create_synthetic_test_data(shape=(91, 109, 91), noise_level=0.1):
    """
    Create synthetic test images for benchmarking.

    Args:
        shape: Image dimensions (default: standard MNI152 2mm shape)
        noise_level: Amount of noise to add (default: 0.1)

    Returns:
        Dictionary with 'template_image', 'reg_image', 'template_mask'
    """
    print(f"Creating synthetic test data with shape {shape}...")

    # Create template image with some structure
    template_arr = np.random.randn(*shape).astype(np.float32)

    # Add some smooth structure using convolution
    from scipy.ndimage import gaussian_filter
    template_arr = gaussian_filter(template_arr, sigma=2.0)

    # Normalize
    template_arr = (template_arr - template_arr.mean()) / template_arr.std()

    # Create registered image (template + noise + slight shift)
    reg_arr = template_arr.copy()
    reg_arr += np.random.randn(*shape).astype(np.float32) * noise_level

    # Simulate slight misregistration with a small shift
    reg_arr = np.roll(reg_arr, shift=1, axis=0)
    reg_arr = np.roll(reg_arr, shift=1, axis=1)

    # Create a brain-like mask (ellipsoid)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx, ry, rz = shape[0] // 3, shape[1] // 3, shape[2] // 3

    mask_arr = (
        ((x - cx) / rx) ** 2 +
        ((y - cy) / ry) ** 2 +
        ((z - cz) / rz) ** 2
    ) <= 1.0

    mask_arr = mask_arr.astype(np.uint8)

    # Convert to ANTs images
    template_image = ants.from_numpy(template_arr)
    reg_image = ants.from_numpy(reg_arr)
    template_mask = ants.from_numpy(mask_arr)

    image_dict = {
        "template_image": template_image,
        "reg_image": reg_image,
        "template_mask": template_mask,
    }

    num_masked_voxels = np.sum(mask_arr)
    print(f"  Template shape: {shape}")
    print(f"  Masked voxels: {num_masked_voxels:,}")
    print(f"  Total voxels: {np.prod(shape):,}")

    return image_dict


def compare_results(results_orig, results_numba, tolerance=1e-3):
    """
    Compare results from original and Numba implementations.

    Args:
        results_orig: Results from original implementation
        results_numba: Results from Numba implementation
        tolerance: Acceptable difference threshold

    Returns:
        Dictionary with comparison statistics
    """
    print("\nComparing results...")

    metrics = ["MSE", "CORR", "dm_MSE", "dm_CORR"]
    comparison = {}

    for metric in metrics:
        orig_arr = results_orig[metric].numpy()
        numba_arr = results_numba[metric].numpy()

        # Compute differences
        abs_diff = np.abs(orig_arr - numba_arr)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        rel_diff = abs_diff / (np.abs(orig_arr) + 1e-10)
        max_rel_diff = np.max(rel_diff)

        comparison[metric] = {
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "max_rel_diff": max_rel_diff,
            "within_tolerance": max_diff < tolerance,
        }

        print(f"\n  {metric}:")
        print(f"    Max absolute diff: {max_diff:.6e}")
        print(f"    Mean absolute diff: {mean_diff:.6e}")
        print(f"    Max relative diff: {max_rel_diff:.6e}")
        print(f"    Within tolerance ({tolerance}): {comparison[metric]['within_tolerance']}")

    return comparison


def run_benchmark(radius=3, shape=(91, 109, 91), skip_mi=True):
    """
    Run comprehensive benchmark comparing implementations.

    Args:
        radius: ROI radius for R2D2 computation
        shape: Image dimensions
        skip_mi: If True, skip MI computation in original (very slow)

    Returns:
        Dictionary with timing results
    """
    print("=" * 70)
    print("R2D2 Performance Benchmark")
    print("=" * 70)

    # Create test data
    image_dict = create_synthetic_test_data(shape=shape)

    results = {}

    # Warm-up Numba JIT compilation
    print("\n" + "-" * 70)
    print("Warming up Numba JIT compiler...")
    print("-" * 70)
    _ = compute_r2d2_numba(image_dict, radius=radius, use_numba_mi=True)
    print("  JIT compilation complete!")

    # Benchmark 1: Original implementation (without MI to save time)
    if not skip_mi:
        print("\n" + "-" * 70)
        print("Benchmark 1: Original implementation (with MI)")
        print("-" * 70)
        start = time.time()
        results_orig = compute_r2d2(image_dict, radius=radius, subsess="benchmark")
        elapsed_orig = time.time() - start
        results["original"] = elapsed_orig
        print(f"  Time: {elapsed_orig:.2f} seconds")
    else:
        print("\n" + "-" * 70)
        print("Benchmark 1: Original implementation SKIPPED (too slow)")
        print("  Use --include-mi flag to run full comparison")
        print("-" * 70)
        results_orig = None

    # Benchmark 2: Numba implementation (approximate MI)
    print("\n" + "-" * 70)
    print("Benchmark 2: Numba implementation (approximate MI)")
    print("-" * 70)
    start = time.time()
    results_numba_approx = compute_r2d2_numba(
        image_dict, radius=radius, subsess="benchmark", use_numba_mi=True
    )
    elapsed_numba_approx = time.time() - start
    results["numba_approx_mi"] = elapsed_numba_approx
    print(f"  Time: {elapsed_numba_approx:.2f} seconds")

    # Benchmark 3: Hybrid implementation (Numba + ANTs MI)
    print("\n" + "-" * 70)
    print("Benchmark 3: Hybrid implementation (Numba for MSE/CORR, ANTs for MI)")
    print("-" * 70)
    start = time.time()
    results_numba_hybrid = compute_r2d2_numba(
        image_dict, radius=radius, subsess="benchmark", use_numba_mi=False
    )
    elapsed_numba_hybrid = time.time() - start
    results["numba_hybrid"] = elapsed_numba_hybrid
    print(f"  Time: {elapsed_numba_hybrid:.2f} seconds")

    # Compare accuracy (if we have original results)
    if results_orig is not None:
        comparison = compare_results(results_orig, results_numba_approx)

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if "original" in results:
        print(f"\nOriginal implementation:        {results['original']:>8.2f} seconds")
        print(f"Numba (approx MI):              {results['numba_approx_mi']:>8.2f} seconds")
        print(f"Numba hybrid (ANTs MI):         {results['numba_hybrid']:>8.2f} seconds")
        print(f"\nSpeedup (Numba approx):         {results['original'] / results['numba_approx_mi']:>8.1f}x")
        print(f"Speedup (Numba hybrid):         {results['original'] / results['numba_hybrid']:>8.1f}x")
    else:
        print(f"\nNumba (approx MI):              {results['numba_approx_mi']:>8.2f} seconds")
        print(f"Numba hybrid (ANTs MI):         {results['numba_hybrid']:>8.2f} seconds")
        print(f"\nOriginal implementation not run (use --include-mi to enable)")

    print("\nRecommendation:")
    if results["numba_approx_mi"] < results["numba_hybrid"] * 0.5:
        print("  Use Numba with approximate MI for maximum speed")
        print("  (validate MI accuracy for your specific use case)")
    else:
        print("  Use Numba hybrid mode for best speed/accuracy balance")
        print("  (ANTs MI is more accurate, moderate performance)")

    print("=" * 70)

    return results


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark R2D2 original vs Numba implementations"
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="ROI radius for R2D2 computation (default: 3)"
    )

    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=[91, 109, 91],
        help="Image dimensions X Y Z (default: 91 109 91)"
    )

    parser.add_argument(
        "--include-mi",
        action="store_true",
        help="Include original implementation with MI (very slow!)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Run benchmark
    results = run_benchmark(
        radius=args.radius,
        shape=tuple(args.shape),
        skip_mi=not args.include_mi
    )
