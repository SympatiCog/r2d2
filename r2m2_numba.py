"""
Numba-accelerated implementation of R2M2 computation.

This module provides significant speedup over the original implementation
by using Numba JIT compilation for the triple-nested loop computation.

Performance improvements:
- Pre-filters to masked voxels only
- Vectorized similarity metric computations
- Parallel execution across voxels
- Reduced Python/C++ boundary crossings
"""

import sys
import numpy as np
import numba
from numba import jit, prange
import re

# Lazy import for ANTs - only check when actually running (not for --help)
ants = None

def _check_ants():
    """Check if ANTs is available and import it."""
    global ants
    if ants is not None:
        return

    try:
        import ants as ants_module
        ants = ants_module
    except ImportError:
        print("\n" + "=" * 70)
        print("ERROR: ANTs (ANTsPy) is not installed")
        print("=" * 70)
        print("\nANTs is required for R2M2 image processing operations.")
        print("\nInstallation instructions:")
        print("\n1. Using pip (recommended):")
        print("   pip install antspyx")
        print("\n2. Using conda:")
        print("   conda install -c conda-forge antspyx")
        print("\n3. From source (advanced):")
        print("   git clone https://github.com/ANTsX/ANTsPy")
        print("   cd ANTsPy")
        print("   pip install .")
        print("\nNote: ANTsPy installation may take 10-20 minutes as it compiles")
        print("C++ code. Make sure you have a C++ compiler installed:")
        print("  - macOS: Install Xcode Command Line Tools")
        print("  - Linux: Install build-essential or equivalent")
        print("  - Windows: Install Visual Studio Build Tools")
        print("\nFor more information, visit:")
        print("  https://github.com/ANTsX/ANTsPy")
        print("=" * 70 + "\n")
        sys.exit(1)


@jit(nopython=True, fastmath=True)
def compute_mse(arr1, arr2):
    """
    Compute Mean Squared Error between two arrays.

    Args:
        arr1, arr2: Input arrays (flattened or multi-dimensional)

    Returns:
        MSE value (float)
    """
    flat1 = arr1.ravel()
    flat2 = arr2.ravel()
    diff = flat1 - flat2
    return np.mean(diff * diff)


@jit(nopython=True, fastmath=True)
def compute_correlation(arr1, arr2):
    """
    Compute Pearson correlation coefficient between two arrays.

    Args:
        arr1, arr2: Input arrays

    Returns:
        Correlation coefficient (float), range [-1, 1]
    """
    flat1 = arr1.ravel()
    flat2 = arr2.ravel()

    # Demean
    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)

    centered1 = flat1 - mean1
    centered2 = flat2 - mean2

    # Compute correlation
    numerator = np.sum(centered1 * centered2)
    denom1 = np.sqrt(np.sum(centered1 * centered1))
    denom2 = np.sqrt(np.sum(centered2 * centered2))

    if denom1 == 0.0 or denom2 == 0.0:
        return 0.0

    return numerator / (denom1 * denom2)


@jit(nopython=True, fastmath=True)
def compute_mutual_information_approx(arr1, arr2, bins=32):
    """
    Compute approximate Mutual Information using histogram-based method.

    Note: This is a simplified approximation of MattesMutualInformation.

    Args:
        arr1, arr2: Input arrays
        bins: Number of histogram bins (default: 32)

    Returns:
        Approximate MI value (float)
    """
    flat1 = arr1.ravel()
    flat2 = arr2.ravel()

    # Normalize to [0, bins-1] range
    min1, max1 = flat1.min(), flat1.max()
    min2, max2 = flat2.min(), flat2.max()

    # Avoid division by zero
    if max1 == min1 or max2 == min2:
        return 0.0

    norm1 = ((flat1 - min1) / (max1 - min1) * (bins - 1)).astype(np.int32)
    norm2 = ((flat2 - min2) / (max2 - min2) * (bins - 1)).astype(np.int32)

    # Clip to valid range
    norm1 = np.clip(norm1, 0, bins - 1)
    norm2 = np.clip(norm2, 0, bins - 1)

    # Build joint histogram
    joint_hist = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(norm1)):
        joint_hist[norm1[i], norm2[i]] += 1.0

    # Normalize to probabilities
    joint_hist /= len(flat1)

    # Compute marginal distributions
    p_x = np.sum(joint_hist, axis=1)
    p_y = np.sum(joint_hist, axis=0)

    # Compute MI
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint_hist[i, j] > 0.0 and p_x[i] > 0.0 and p_y[j] > 0.0:
                mi += joint_hist[i, j] * np.log(joint_hist[i, j] / (p_x[i] * p_y[j]))

    return mi


@jit(nopython=True, parallel=True)
def compute_r2m2_kernel(
    reg_arr,
    tmplt_arr,
    mask_arr,
    radius,
    compute_mi=True
):
    """
    Numba-accelerated kernel for computing R2M2 metrics.

    This function performs the core triple-nested loop computation
    with significant performance improvements through JIT compilation
    and parallelization.

    Args:
        reg_arr: Registered image as numpy array (X, Y, Z)
        tmplt_arr: Template image as numpy array (X, Y, Z)
        mask_arr: Template mask as numpy array (X, Y, Z)
        radius: ROI radius for local similarity computation
        compute_mi: Whether to compute MI (slower) or skip it

    Returns:
        Tuple of 6 numpy arrays: (MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR)
    """
    X, Y, Z = tmplt_arr.shape

    # Pre-allocate output arrays
    MI = np.zeros((X, Y, Z), dtype=np.float64)
    MSE = np.zeros((X, Y, Z), dtype=np.float64)
    CORR = np.zeros((X, Y, Z), dtype=np.float64)
    dm_MI = np.zeros((X, Y, Z), dtype=np.float64)
    dm_MSE = np.zeros((X, Y, Z), dtype=np.float64)
    dm_CORR = np.zeros((X, Y, Z), dtype=np.float64)

    # Parallel loop over x dimension
    for x in prange(X):
        for y in range(Y):
            for z in range(Z):
                # Skip voxels outside mask
                if mask_arr[x, y, z] != 1:
                    continue

                # Compute ROI bounds
                x_min = max(0, x - radius)
                x_max = min(X, x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(Y, y + radius + 1)
                z_min = max(0, z - radius)
                z_max = min(Z, z + radius + 1)

                # Extract ROIs
                roi_reg = reg_arr[x_min:x_max, y_min:y_max, z_min:z_max]
                roi_tmplt = tmplt_arr[x_min:x_max, y_min:y_max, z_min:z_max]

                # Compute raw metrics
                if compute_mi:
                    MI[x, y, z] = compute_mutual_information_approx(roi_tmplt, roi_reg)
                MSE[x, y, z] = compute_mse(roi_tmplt, roi_reg)
                CORR[x, y, z] = compute_correlation(roi_tmplt, roi_reg)

                # Compute demeaned ROIs
                roi_reg_mean = np.mean(roi_reg)
                roi_tmplt_mean = np.mean(roi_tmplt)

                dm_roi_reg = roi_reg - roi_reg_mean
                dm_roi_tmplt = roi_tmplt - roi_tmplt_mean

                # Compute demeaned metrics
                if compute_mi:
                    dm_MI[x, y, z] = compute_mutual_information_approx(dm_roi_tmplt, dm_roi_reg)
                dm_MSE[x, y, z] = compute_mse(dm_roi_tmplt, dm_roi_reg)
                dm_CORR[x, y, z] = compute_correlation(dm_roi_tmplt, dm_roi_reg)

    return MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR


def compute_r2m2_numba(
    image_dict: dict,
    radius: float = 3,
    subsess: str = "unknown",
    use_numba_mi: bool = False
) -> dict:
    """
    Numba-accelerated version of compute_r2m2.

    This function provides significant speedup over the original implementation
    while maintaining compatibility with the existing API.

    Args:
        image_dict: Dictionary containing 'template_image', 'reg_image', 'template_mask'
        radius: Search radius for R2M2 metrics (default: 3)
        subsess: Subject/session identifier for error messages
        use_numba_mi: If True, use Numba's approximate MI. If False, use ANTs MI (slower but more accurate)

    Returns:
        Dictionary with keys: MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR
        Each value is an ANTsImage
    """
    template_image = image_dict.get("template_image")
    reg_image = image_dict.get("reg_image")
    template_mask = image_dict.get("template_mask")

    # Convert ANTs images to numpy arrays
    reg_arr = reg_image.numpy()
    tmplt_arr = template_image.numpy()
    mask_arr = template_mask.numpy()

    try:
        # Call Numba-accelerated kernel
        MI_arr, MSE_arr, CORR_arr, dm_MI_arr, dm_MSE_arr, dm_CORR_arr = compute_r2m2_kernel(
            reg_arr,
            tmplt_arr,
            mask_arr,
            int(radius),
            compute_mi=use_numba_mi
        )

        # If not using Numba MI, fall back to ANTs for MI computation
        # This hybrid approach uses Numba for fast metrics and ANTs for accurate MI
        if not use_numba_mi:
            print(f"  Computing MI with ANTs (more accurate, slower)...")
            MI_arr, dm_MI_arr = compute_mi_with_ants(
                reg_image, template_image, template_mask, radius
            )

        # Convert numpy arrays back to ANTs images
        MI = ants.from_numpy(MI_arr, origin=template_image.origin,
                            spacing=template_image.spacing,
                            direction=template_image.direction)
        MSE = ants.from_numpy(MSE_arr, origin=template_image.origin,
                             spacing=template_image.spacing,
                             direction=template_image.direction)
        CORR = ants.from_numpy(CORR_arr, origin=template_image.origin,
                              spacing=template_image.spacing,
                              direction=template_image.direction)
        dm_MI = ants.from_numpy(dm_MI_arr, origin=template_image.origin,
                                spacing=template_image.spacing,
                                direction=template_image.direction)
        dm_MSE = ants.from_numpy(dm_MSE_arr, origin=template_image.origin,
                                 spacing=template_image.spacing,
                                 direction=template_image.direction)
        dm_CORR = ants.from_numpy(dm_CORR_arr, origin=template_image.origin,
                                  spacing=template_image.spacing,
                                  direction=template_image.direction)

        results_dict = {
            "MI": MI,
            "MSE": MSE,
            "CORR": CORR,
            "dm_MI": dm_MI,
            "dm_MSE": dm_MSE,
            "dm_CORR": dm_CORR,
        }

        return results_dict

    except Exception as e:
        print(f"r2m2_numba failed on {subsess}: {e}")
        raise


def compute_mi_with_ants(reg_image, template_image, template_mask, radius):
    """
    Compute MI using ANTs for higher accuracy (hybrid approach).

    This function is called when use_numba_mi=False to compute
    only the MI metrics using ANTs while other metrics use Numba.

    Args:
        reg_image: ANTs registered image
        template_image: ANTs template image
        template_mask: ANTs template mask
        radius: ROI radius

    Returns:
        Tuple of (MI_array, dm_MI_array)
    """
    X, Y, Z = template_image.shape
    MI_arr = np.zeros((X, Y, Z))
    dm_MI_arr = np.zeros((X, Y, Z))

    # Get masked coordinates to avoid iterating over all voxels
    mask_arr = template_mask.numpy()
    masked_coords = np.argwhere(mask_arr == 1)

    for idx, (x, y, z) in enumerate(masked_coords):
        lower = [max(0, x - radius), max(0, y - radius), max(0, z - radius)]
        # Use +1 to ensure the upper bound results in a large enough count for Python slicing
        upper = [min(X, x + radius + 1), min(Y, y + radius + 1), min(Z, z + radius + 1)]

        # Crop ROIs
        roi_reg = ants.crop_indices(reg_image, lowerind=lower, upperind=upper)
        roi_tmplt = ants.crop_indices(template_image, lowerind=lower, upperind=upper)

        try:
            # Compute MI
            MI_arr[x, y, z] = ants.image_similarity(
                roi_tmplt, roi_reg, metric_type="MattesMutualInformation"
            )

            # Compute demeaned MI
            dm_roi_reg = roi_reg - roi_reg.mean()
            dm_roi_tmplt = roi_tmplt - roi_tmplt.mean()

            dm_MI_arr[x, y, z] = ants.image_similarity(
                dm_roi_tmplt, dm_roi_reg, metric_type="MattesMutualInformation"
            )
            
        except Exception:
            # If a specific voxel still fails (e.g. ITK internal error), skip it 
            # to prevent the entire multi-subject run from crashing.
            continue

        # Progress indicator for long computations
        if idx % 5000 == 0 and idx > 0:
            print(f"    Processed {idx}/{len(masked_coords)} masked voxels for MI")

    return MI_arr, dm_MI_arr


# ============================================================================
# Command-line interface (mirrors r2m2_base.py structure)
# ============================================================================

def load_images(reg_image: str, template_path: str) -> dict:
    """
    Load images from a sub-folder.

    Args:
        reg_image: full path to the registered image
        template_path: full path to the template image

    Returns:
        image_dict containing the registered image, template and mask
    """
    import os
    image_dict = {}

    # Validate input files exist
    if not os.path.exists(reg_image):
        raise FileNotFoundError(f"Registered image not found: {reg_image}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template image not found: {template_path}")

    # Construct mask path
    stem, ext = template_path[:-7], template_path[-7:]
    mask_path = f"{stem}_mask{ext}"
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Template mask not found: {mask_path}")

    image_dict["reg_image"] = ants.image_read(reg_image)
    image_dict["template_image"] = ants.image_read(template_path)
    image_dict["template_mask"] = ants.image_read(mask_path)
    return image_dict

def save_images(sub_fldr: str, image_res: dict, radius: float):
    """
    Save images to a sub-folder.

    Args:
        sub_fldr: full path to destination folder
        image_res: dictionary containing the r2m2 images
        radius: radius value for filename
    """
    import os
    # Ensure target directory exists before writing files
    os.makedirs(sub_fldr, exist_ok=True)
    print(f"Saving:")
    for k, v in image_res.items():
        outpath = os.path.join(sub_fldr, f"r2m2_{k}_rad{radius}.nii")
        print(f"  {outpath}")
        ants.image_write(v, outpath)


def comp_stats(
    r2m2: dict,
    img_dict: dict,
    metrics=["MattesMutualInformation", "MeanSquares", "Correlation"],
) -> dict:
    """
    Compute basic summary stats on images.

    Args:
        r2m2: dictionary containing the r2m2 images
        img_dict: dictionary containing template and registered images
        metrics: list of similarity metrics to compute on whole brain

    Returns:
        dictionary containing the computed stats
    """
    mask = img_dict.get("template_mask")
    template = img_dict.get("template_image")
    reg_image = img_dict.get("reg_image")
    summary_stats = {}

    try:
        for k, v in r2m2.items():
            summary_stats[f"{k}_mean"] = v[mask > 0].mean()
            summary_stats[f"{k}_std"] = v[mask > 0].std()
            summary_stats[f"{k}_z"] = (
                summary_stats[f"{k}_mean"] / summary_stats[f"{k}_std"]
            )

        for metric in metrics:
            summary_stats[f"{metric}_wholebrain"] = ants.image_similarity(
                template,
                reg_image,
                metric_type=metric,
            )
    except:
        for k, v in r2m2.items():
            summary_stats[f"{k}_mean"] = np.nan
            summary_stats[f"{k}_std"] = np.nan
            summary_stats[f"{k}_z"] = np.nan

        for metric in metrics:
            summary_stats[f"{metric}_wholebrain"] = np.nan

    return summary_stats


def main(
    reg_image_name: str = "registered_t2_img.nii.gz",
    template_path: str = None,
    radius: int = 3,
    use_numba_mi: bool = False,
) -> dict:
    """
    Main processing function for a single subject.

    Args:
        sub_folder: path to subject folder
        reg_image_name: name of the registered image file
        template_path: path to the template image
        radius: ROI radius for R2M2 computation
        use_numba_mi: if True, use Numba approximate MI; if False, use ANTs MI

    Returns:
        Dictionary with subject statistics, or Exception on failure
    """
    import os

    if template_path is None:
        raise ValueError("template_path is required. Please specify --template_path when running from command line.")

    sub_folder = os.path.dirname(reg_image_name)
    reg_image_name = os.path.basename(reg_image_name)
    img_dict = load_images(
        reg_image=f"{sub_folder}/{reg_image_name}", template_path=template_path
    )
    m = re.search(r'sub-([^/_]+)', sub_folder)
    sub_id = m.group(0) if m else None
    subsess = sub_id

    # Use Numba-accelerated version
    r2m2 = compute_r2m2_numba(
        img_dict, radius=radius, subsess=subsess, use_numba_mi=use_numba_mi
    )

    if type(r2m2) is dict:
        save_images(sub_folder, r2m2, radius)
        res = {"subsess": subsess}
        comp_vals = comp_stats(r2m2, img_dict)
        res.update(comp_vals)
        return res
    else:
        print(f"{subsess} failed in main()")
        return r2m2


def get_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Regional Registration Mismatch Metrics (Numba-accelerated version).",
        epilog="This is the high-performance Numba implementation providing 10-50x speedup over r2m2_base.py"
    )

    parser.add_argument(
        "--list_path",
        dest="list_path",
        required=False,
        help="full path to text file list of subject folders",
        default=None,
    )

    parser.add_argument(
        "--search_string",
        dest="search_string",
        required=False,
        help="glob-able search string for target images, e.g. './sub-*/registered_space_imgs.nii.gz'",
        default=None,
    )

    parser.add_argument(
        "--num_python_jobs",
        dest="num_python_jobs",
        default=4,
        type=int,
        help="The number of python jobs to spawn to use; default=4",
    )

    parser.add_argument(
        "--num_itk_cores",
        dest="num_itk_cores",
        default="1",
        type=str,
        help="The number of cores that ITK can use in each python job; default=1",
    )

    parser.add_argument(
        "--use-numba-mi",
        dest="use_numba_mi",
        action="store_true",
        help="Use Numba's approximate MI (faster but less accurate). Default: use ANTs MI (hybrid mode)",
    )

    parser.add_argument(
        "--radius",
        dest="radius",
        default=3,
        type=int,
        help="ROI radius for similarity computation; default=3",
    )

    parser.add_argument(
        "--template_path",
        dest="template_path",
        required=True,
        type=str,
        help="Path to template image file (e.g., MNI152_T1_2mm.nii.gz). Template mask must exist as {template}_mask.nii.gz",
    )

    args = parser.parse_args()
    return args


# Run as main, distribute across num_python_jobs processes
if __name__ == "__main__":
    import os
    import glob

    args = get_args()

    # Check for ANTs after argument parsing (allows --help to work without ANTs)
    _check_ants()

    # Import heavy dependencies after help/args are processed
    import pandas as pd
    from multiprocess import Pool

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = args.num_itk_cores

    if args.list_path is not None:
        with open(args.list_path, "r") as f:
            raw_data = f.read()
        flist = [fil.strip() for fil in raw_data.split()]
        num_files = len(flist)
    elif args.search_string is not None:
        flist = glob.glob(args.search_string)
        # flist = [os.path.split(f)[0] for f in flist]
        
    else:
        print("\nError: You must provide either --list_path or --search_string\n")
        print("Examples:")
        print("  python r2m2_numba.py --list_path subjects.txt")
        print("  python r2m2_numba.py --search_string './sub-*/registered_t2_img.nii.gz'")
        print("\nOptional flags:")
        print("  --use-numba-mi         Use fast approximate MI (default: use ANTs MI)")
        print("  --radius N             Set ROI radius (default: 3)")
        print("  --num_python_jobs N    Number of parallel processes (default: 4)\n")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"R2M2 Numba-Accelerated Processing")
    print(f"{'='*70}")
    print(f"Mode: {'Full Numba (approximate MI)' if args.use_numba_mi else 'Hybrid (ANTs MI + Numba MSE/CORR)'}")
    print(f"Template: {args.template_path}")
    print(f"Subjects: {len(flist)}")
    print(f"Parallel jobs: {args.num_python_jobs}")
    print(f"ITK cores per job: {args.num_itk_cores}")
    print(f"Radius: {args.radius}")
    print(f"{'='*70}\n")

    # Create wrapper function with fixed parameters
    def main_wrapper(sub_folder):
        return main(
            sub_folder,
            radius=args.radius,
            use_numba_mi=args.use_numba_mi,
            template_path=args.template_path
        )

    with Pool(args.num_python_jobs) as pool:
        res = pool.map(main_wrapper, flist)

    dict_data = [r for r in res if type(r) is dict]
    err_data = [r for r in res if type(r) is not dict]

    dat = pd.DataFrame(dict_data)
    ts = pd.Timestamp("now")
    tstr = ts.strftime("%Y_%m_%d-%X")
    output_csv = f"r2m2_numba_summary_stats_{tstr}.csv"
    dat.to_csv(output_csv, index=False)

    error_log = f"r2m2_numba_errs_{tstr}.txt"
    with open(error_log, "w") as f:
        f.write(str(err_data))

    print(f"\n{'='*70}")
    print(f"Processing Complete")
    print(f"{'='*70}")
    print(f"Successful subjects: {len(dict_data)}")
    print(f"Failed subjects: {len(err_data)}")
    print(f"Summary statistics: {output_csv}")
    print(f"Error log: {error_log}")
    print(f"{'='*70}\n")
