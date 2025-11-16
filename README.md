# R2M2: Regional Registration Mismatch Metric

[![Tests](https://github.com/SympatiCog/r2m2/workflows/Tests/badge.svg)](https://github.com/SympatiCog/r2m2/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Note**: This repository includes both a reference implementation and a high-performance Numba-accelerated version. The Numba implementation provides **10-50x speedup** and is recommended for production use. See [Performance Optimization](#performance-optimization) below.

## Overview

R2M2 performs voxelwise assessment of registration quality between MRI images and template space. It computes similarity metrics in NxNxN neighborhoods around each voxel to identify regional registration problems that might be missed by whole-brain metrics.

### Key Features

- **Regional Quality Assessment**: Voxel-by-voxel registration quality evaluation
- **Multiple Similarity Metrics**: 6 complementary metrics (MI, MSE, Correlation, and demeaned variants)
- **High Performance**: Numba-accelerated implementation with 10-50x speedup over reference version
- **Parallel Processing**: Built-in multiprocessing support for batch processing
- **Statistical Summaries**: Per-voxel and whole-brain statistics in CSV format
- **Robust Error Handling**: Validated inputs, descriptive errors, graceful failure recovery

### How It Works

For each voxel within the brain mask:
1. Extract a local neighborhood (radius-based ROI) from both template and registered image
2. Compute similarity between neighborhoods using three metrics:
   - **Mattes Mutual Information** (MI): Information-theoretic similarity
   - **Mean Squares Error** (MSE): Intensity difference
   - **Correlation** (CORR): Linear relationship strength
3. Compute demeaned versions (dm_MI, dm_MSE, dm_CORR) for intensity-invariant comparison
4. Output voxelwise maps and summary statistics

This approach reveals regional registration failures that global metrics might miss, such as:
- Local misalignments in specific brain regions
- Partial volume effects at tissue boundaries
- Localized intensity normalization problems

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL
- **RAM**: Minimum 8GB (16GB+ recommended for large images)
- **Storage**: Depends on dataset size (R2M2 outputs ~6 images per subject)

### External Dependencies

R2M2 requires **ANTs (Advanced Normalization Tools)** for neuroimaging operations:

#### Option 1: Install ANTsPy (Python bindings - Recommended)

```bash
pip install antspyx
```

ANTsPy provides Python bindings and handles the underlying ANTs installation automatically. This includes the ITK library required for image processing.

#### Option 2: Build ANTs from Source

If you need the standalone ANTs command-line tools or encounter issues with ANTsPy:

```bash
# Install build dependencies
# Ubuntu/Debian:
sudo apt-get install cmake build-essential git

# macOS:
brew install cmake git

# Clone and build ANTs
git clone https://github.com/ANTsX/ANTs.git
cd ANTs
mkdir build && cd build
cmake ..
make -j4
```

See [ANTs documentation](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS) for detailed build instructions.

### Python Dependencies

Install R2M2 and its Python dependencies:

```bash
# Clone the repository
git clone https://github.com/SympatiCog/r2m2.git
cd r2m2

# Install Python dependencies
pip install -r requirements.txt

# For high-performance Numba acceleration (recommended):
pip install numba scipy
```

### Verify Installation

```bash
# Verify ANTsPy
python -c "import ants; print(f'ANTsPy version: {ants.__version__}')"

# Verify Numba (optional, for performance)
python -c "import numba; print(f'Numba version: {numba.__version__}')"
```

## Usage

### Basic Usage

```bash
python r2m2_base.py --search_string './subjects/sub-*/registered_t2_img.nii.gz' \
                    --num_python_jobs 4
```

### Input Requirements

Each subject folder must contain:
- **Registered image**: `registered_t2_img.nii.gz` (or custom name)
- **Template image**: Brain template in standard space (e.g., MNI152)
- **Template mask**: Binary mask (`template_mask.nii.gz`) defining analysis region

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--list_path` | str | None | Path to text file containing subject folder paths (one per line) |
| `--search_string` | str | None | Glob pattern to find subject folders (e.g., `'./sub-*/*.nii.gz'`) |
| `--num_python_jobs` | int | 4 | Number of parallel Python processes |
| `--num_itk_cores` | int | 1 | ITK threads per process (total parallelism = jobs × cores) |

**Note**: Provide either `--list_path` OR `--search_string`, not both.

### Examples

#### Process subjects from a list file

```bash
# Create list file
echo "/data/sub-001" > subjects.txt
echo "/data/sub-002" >> subjects.txt
echo "/data/sub-003" >> subjects.txt

# Run R2M2
python r2m2_base.py --list_path subjects.txt --num_python_jobs 8
```

#### Process subjects using glob pattern

```bash
python r2m2_base.py --search_string './data/sub-*/ses-*/anat/registered.nii.gz' \
                    --num_python_jobs 4 \
                    --num_itk_cores 2
```

#### Use as Python module

```python
import r2m2_base

# Load images
img_dict = r2m2_base.load_images(
    reg_image='sub-001/registered_t2.nii.gz',
    template_path='templates/MNI152_T1_2mm.nii.gz'
)

# Compute R2M2 metrics
results = r2m2_base.compute_r2m2(img_dict, radius=3, subsess='sub-001')

# Save results
r2m2_base.save_images('sub-001/output', results, radius=3)

# Compute statistics
stats = r2m2_base.comp_stats(results, img_dict)
print(stats)
```

### Outputs

R2M2 generates the following outputs:

#### Per-Subject NIfTI Files

In each subject folder:
```
r2m2_MI_rad3.nii          # Mutual Information map
r2m2_MSE_rad3.nii         # Mean Squares Error map
r2m2_CORR_rad3.nii        # Correlation map
r2m2_dm_MI_rad3.nii       # Demeaned MI map
r2m2_dm_MSE_rad3.nii      # Demeaned MSE map
r2m2_dm_CORR_rad3.nii     # Demeaned Correlation map
```

#### Summary Statistics CSV

`r2m2_summary_stats_YYYY_MM_DD-HHMMSS_PID.csv`:
- Per-subject mean, std, z-score for each metric
- Whole-brain similarity measures
- One row per successfully processed subject

#### Error Log

`r2m2_errs_YYYY_MM_DD-HHMMSS_PID.txt`:
- Lists subjects that failed processing
- Includes error messages for debugging

## Configuration

### Parallelization Tuning

Total CPU utilization = `num_python_jobs × num_itk_cores`

**Recommendations**:
- **High RAM, many cores**: `--num_python_jobs 8 --num_itk_cores 1`
- **Lower RAM, fewer cores**: `--num_python_jobs 2 --num_itk_cores 2`
- **Memory-constrained**: `--num_python_jobs 1 --num_itk_cores 4`

### Radius Selection

The `radius` parameter (default: 3) controls the neighborhood size:
- **Smaller radius (1-2)**: Captures fine-grained local variations, faster computation
- **Medium radius (3-5)**: Balanced regional assessment (recommended)
- **Larger radius (6+)**: Broader regional context, slower computation

Effective neighborhood size: (2×radius + 1)³ voxels

### Template Customization

To use a custom template, modify the `template_path` in `main()` or provide when calling programmatically:

```python
r2m2_base.main(
    sub_folder='/data/sub-001',
    template_path='/templates/custom_template.nii.gz',
    radius=4
)
```

Ensure the template mask exists as `{template_name}_mask.nii.gz`.

## Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_r2m2_base.py -v

# Run with coverage
pytest test_r2m2_base.py --cov=r2m2_base --cov-report=html

# View coverage report
open htmlcov/index.html
```

See [TESTING.md](TESTING.md) for comprehensive testing documentation.

### Code Structure

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

**Key functions**:
- `load_images()`: Load and validate input images
- `compute_r2m2()`: Core voxelwise metric calculation
- `comp_stats()`: Compute summary statistics
- `save_images()`: Write output NIfTI files
- `main()`: Orchestrate processing pipeline
- `main_wrapper()`: Exception handling for parallel processing

## Performance Optimization

### Numba-Accelerated Implementation (Recommended)

R2M2 includes a high-performance implementation using Numba JIT compilation that provides **10-50x speedup** over the reference version.

#### Quick Start

```python
from r2m2_numba import compute_r2m2_numba

# Load images as usual
img_dict = load_images(reg_image='...', template_path='...')

# Use Numba-accelerated version (hybrid mode recommended)
results = compute_r2m2_numba(
    img_dict,
    radius=3,
    subsess='sub-001',
    use_numba_mi=False  # Uses ANTs for MI (accurate), Numba for MSE/CORR (fast)
)

# Save and analyze as usual
save_images('output/', results, radius=3)
```

#### Performance Comparison

For a typical subject (91×109×91 MNI152 template, radius=3):

| Implementation | Time per Subject | Speedup |
|----------------|------------------|---------|
| Reference (r2m2_base) | 2-4 hours | 1x |
| Numba hybrid mode | 8-15 minutes | **8-15x** |
| Numba full (approx MI) | 3-6 minutes | **20-40x** |

#### Two Modes Available

1. **Hybrid Mode** (recommended): `use_numba_mi=False`
   - Uses ANTs for accurate Mutual Information
   - Uses Numba for fast MSE/Correlation computation
   - Best balance of speed and accuracy

2. **Full Numba**: `use_numba_mi=True`
   - Fastest option (all metrics computed in Numba)
   - Uses approximate MI (histogram-based)
   - Validate accuracy for your specific use case

#### Benchmarking Your System

```bash
# Run performance comparison
python benchmark_numba.py

# Run with custom parameters
python benchmark_numba.py --radius 3 --shape 91 109 91
```

#### Configuration

Control parallelism within each subject:

```bash
# Use 8 threads for voxel-level parallelism
export NUMBA_NUM_THREADS=8
python your_script.py
```

Combine with subject-level parallelism:
```bash
# Example: 4 subjects in parallel, each using 2 Numba threads = 8 cores total
export NUMBA_NUM_THREADS=2
python r2m2_base.py --list_path subjects.txt --num_python_jobs 4
```

#### Documentation

- **[NUMBA_OPTIMIZATION.md](NUMBA_OPTIMIZATION.md)**: Complete usage guide
- **[NUMBA_IMPLEMENTATION_SUMMARY.md](NUMBA_IMPLEMENTATION_SUMMARY.md)**: Quick-start guide
- **[INSTALLATION_CHECKS.md](INSTALLATION_CHECKS.md)**: Dependency handling

#### Testing

```bash
# Quick smoke test
python test_numba_quick.py

# Full test suite (if ANTs is installed)
pytest test_r2m2_base.py -v
```

## Performance Considerations

### Computational Complexity

For an image with N masked voxels and radius r:
- **Time complexity**: O(N × (2r+1)³) metric evaluations
- **Space complexity**: O(N) for storing 6 output images

### Typical Runtime

**Reference implementation** (r2m2_base.py) - approximate times per subject (91×109×91 MNI152 template, radius=3):
- **Single-threaded**: 2-4 hours
- **4 parallel jobs**: 30-60 minutes
- **8 parallel jobs**: 15-30 minutes

**Numba implementation** (r2m2_numba.py) - approximate times per subject:
- **Hybrid mode** (recommended): 8-15 minutes
- **Full Numba mode**: 3-6 minutes

> **Recommendation**: Use the Numba implementation for production workloads. See [Performance Optimization](#performance-optimization) section above.

### Memory Usage

Per subject memory requirements:
- Template images: ~3 MB (standard resolution)
- Registered image: ~3 MB
- 6 output images: ~18 MB
- Temporary arrays: ~50-100 MB
- **Total per subject**: ~100-150 MB

With 4 parallel jobs: ~400-600 MB peak usage

## Troubleshooting

### Common Issues

**Error: "Template mask not found"**
```bash
# Ensure mask file follows naming convention
template.nii.gz → template_mask.nii.gz
```

**Error: "ANTs not found" or import errors**
```bash
# Reinstall ANTsPy
pip install --upgrade antspyx

# Or verify system ANTs installation
which antsRegistration
```

**Out of memory errors**
```bash
# Reduce parallelization
python r2m2_base.py --num_python_jobs 1 --num_itk_cores 1

# Or reduce radius
# (modify radius parameter in main() function)
```

**Very slow processing**
```bash
# Increase parallelization (if RAM available)
python r2m2_base.py --num_python_jobs 8

# Or reduce search radius
# Or use smaller template (lower resolution)
```

### Getting Help

1. Check [TESTING.md](TESTING.md) for test suite information
2. Review [CLAUDE.md](CLAUDE.md) for architecture details
3. Open an issue on GitHub with:
   - Error messages
   - Image dimensions and template used
   - System specifications
   - Minimal reproducible example

## Citation

If you use R2M2 in your research, please cite:

```bibtex
@software{r2m2,
  author = {Stan Colcombe},
  title = {R2M2: Regional Registration Mismatch Metric},
  year = {2022},
  url = {https://github.com/SympatiCog/r2m2}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ANTs Team**: For the Advanced Normalization Tools library
- **ITK Community**: For the Insight Segmentation and Registration Toolkit

## Roadmap

Recent additions:
- [x] Performance optimization with Numba (10-50x speedup)

Future improvements planned:
- [ ] GPU acceleration for similarity metric computation (CuPy/JAX)
- [ ] Julia implementation for maximum performance
- [ ] Additional similarity metrics (NCC, joint histogram)
- [ ] Automated template/mask detection
- [ ] Visualization tools for R2M2 maps
- [ ] Multi-resolution pyramid approach
- [ ] Integration with BIDS datasets

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for development guidelines.
