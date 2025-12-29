# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

R2D2 (Regional Registration Mismatch Metric) is a neuroimaging tool that performs voxelwise assessment of registration quality between MRI images and template space. It computes similarity metrics in NxNxN neighborhoods around each voxel to identify regional registration problems.

**Important**: This is a proof-of-concept implementation. The expensive triple-nested loop operations in native Python are intentionally not optimized for production use. Future versions will rewrite these in Julia, Numba, or similar performance-oriented frameworks.

## Core Architecture

### Main Algorithm (`r2d2_base.py`)

The codebase consists of a single module with the following pipeline:

1. **Image Loading** (`load_images`): Loads registered image, template, and template mask
2. **R2D2 Computation** (`compute_r2d2`): Triple-nested loop over all voxels (x, y, z)
   - For each masked voxel, crops a local neighborhood (radius-based ROI)
   - Computes 6 similarity metrics between template and registered image:
     - MI, MSE, CORR (raw values)
     - dm_MI, dm_MSE, dm_CORR (demeaned values)
   - Uses ANTs `image_similarity` with metric types: MattesMutualInformation, MeanSquares, Correlation
3. **Statistics** (`comp_stats`): Computes mean, std, z-scores for each metric, plus whole-brain similarity
4. **Save Results** (`save_images`): Outputs NIfTI files for each metric

### Key Dependencies

- **ANTs (ANTsPy)**: Core neuroimaging library for registration and similarity metrics
- **multiprocess**: Parallelization via process pools
- **numpy/pandas**: Data manipulation and output

### Data Flow

```
Input: folder containing registered_t2_img.nii.gz
  ↓
load_images() → {reg_image, template_image, template_mask}
  ↓
compute_r2d2() → {MI, MSE, CORR, dm_MI, dm_MSE, dm_CORR} images
  ↓
save_images() → r2d2_{metric}_rad{radius}.nii files
  ↓
comp_stats() → summary statistics dict
  ↓
Output: CSV with per-subject stats, error log
```

## Running the Code

### Command-line execution

```bash
# Using a list of subject folders
python r2d2_base.py --list_path /path/to/subject_list.txt \
                    --num_python_jobs 4 \
                    --num_itk_cores 1

# Using glob pattern to find subject folders
python r2d2_base.py --search_string './sub-*/registered_space_imgs.nii.gz' \
                    --num_python_jobs 4 \
                    --num_itk_cores 1
```

### Parallelization Parameters

- `--num_python_jobs`: Number of parallel Python processes (default: 4)
- `--num_itk_cores`: ITK threads per process (default: 1)
- Total parallelism = num_python_jobs × num_itk_cores

### Input Requirements

Each subject folder must contain:
- `registered_t2_img.nii.gz` (or custom name via `reg_image_name` parameter)
- Template and template mask must exist at specified `template_path`

### Outputs

- Per-subject NIfTI files: `r2d2_{MI,MSE,CORR,dm_MI,dm_MSE,dm_CORR}_rad{radius}.nii`
- Summary CSV: `r2d2_summary_stats_{timestamp}.csv`
- Error log: `r2d2_errs_{timestamp}.txt`

## Important Implementation Notes

### Performance Bottleneck

The `compute_r2d2` function contains a triple-nested loop iterating over all voxels. This is the primary computational bottleneck and is **intentionally unoptimized** as this is proof-of-concept code.

### Error Handling

Error handling has been improved with specific exception types and descriptive messages. The `main_wrapper()` function catches exceptions from parallel processing to prevent complete failure when individual subjects error out.

### Template Dimensions

Template dimensions are now extracted dynamically from the template image shape rather than hardcoded.

### Default Template Path

The default template path in `main()` is set to a local path (`/Users/stan/Projects/...`). Override via function parameter when calling programmatically.

## Testing

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_r2d2_base.py -v

# Run with coverage report
pytest test_r2d2_base.py -v --cov=r2d2_base --cov-report=html

# Run specific test class
pytest test_r2d2_base.py::TestROIFunctions -v

# Run specific test
pytest test_r2d2_base.py::TestROIFunctions::test_roi_min_vals_no_clipping -v
```

### Test Structure

The test suite (`test_r2d2_base.py`) includes:

- **Unit Tests**:
  - `TestROIFunctions`: Tests for `roi_min_vals()` and `roi_max_vals()` boundary calculations
  - `TestLoadImages`: Tests for file validation and image loading
  - `TestSaveImages`: Tests for directory creation and file writing
  - `TestCompStats`: Tests for statistics computation and error handling
  - `TestComputeR2D2`: Tests for main R2D2 computation logic
  - `TestMainFunction`: Tests for `main()` and `main_wrapper()` orchestration
  - `TestArgumentParsing`: Tests for command-line argument parsing

- **Integration Tests**:
  - `TestIntegration`: End-to-end tests with synthetic ANTs images (marked as slow)

### Test Coverage

Key areas covered:
- Input validation and error handling (FileNotFoundError, ValueError)
- Boundary conditions for ROI calculations
- Path handling and security (preventing path traversal)
- Exception handling and error propagation
- Parallel processing error recovery (via main_wrapper)
- Output file generation and naming
