# Testing Guide for R2D2

This document provides comprehensive information about testing the R2D2 codebase.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_r2d2_base.py -v

# Run with coverage
pytest test_r2d2_base.py --cov=r2d2_base --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Organization

### Unit Tests (Fast)

These tests use mocking to isolate individual functions and run quickly:

- **TestROIFunctions** - Tests ROI boundary calculations
  - Edge cases (coordinates at 0, near boundaries)
  - Clipping behavior when ROI exceeds image dimensions
  - Validation of required parameters

- **TestLoadImages** - Tests file loading and validation
  - File existence checks
  - Proper error messages for missing files
  - Correct path handling for mask files
  - Prevention of path traversal attacks

- **TestSaveImages** - Tests output file writing
  - Directory creation when needed
  - Correct filename generation with radius parameter
  - All metrics are saved

- **TestCompStats** - Tests statistics computation
  - Mean, std, z-score calculations
  - Whole-brain similarity metrics
  - Error handling with NaN fallback

- **TestComputeR2D2** - Tests core R2D2 algorithm
  - All 6 metrics are computed (MI, MSE, CORR, dm_*)
  - Proper exception handling and propagation
  - Template dimension extraction

- **TestMainFunction** - Tests orchestration layer
  - Success path with all steps
  - Error handling and propagation
  - main_wrapper exception catching for parallel processing

- **TestArgumentParsing** - Tests CLI argument parsing
  - Default values
  - Custom parallelization settings
  - Input file specification methods

### Integration Tests (Slow)

- **TestIntegration** - End-to-end tests with synthetic ANTs images
  - Currently skipped by default (marked with `@pytest.mark.slow`)
  - Run manually: `pytest test_r2d2_base.py::TestIntegration -v -m slow`

## Running Specific Tests

```bash
# Run a single test class
pytest test_r2d2_base.py::TestROIFunctions -v

# Run a single test method
pytest test_r2d2_base.py::TestLoadImages::test_load_images_file_not_found_registered -v

# Run tests matching a pattern
pytest test_r2d2_base.py -k "roi" -v

# Run with verbose output and show print statements
pytest test_r2d2_base.py -v -s
```

## Coverage Analysis

```bash
# Generate HTML coverage report
pytest test_r2d2_base.py --cov=r2d2_base --cov-report=html

# Generate terminal coverage report
pytest test_r2d2_base.py --cov=r2d2_base --cov-report=term-missing

# Generate XML coverage report (for CI/CD)
pytest test_r2d2_base.py --cov=r2d2_base --cov-report=xml
```

## Test Fixtures and Mocking

The test suite uses several strategies to avoid dependencies on actual NIfTI files:

1. **Temporary Directories** - `tempfile.mkdtemp()` for file system tests
2. **Mock ANTs Objects** - `unittest.mock.Mock()` to simulate ANTs images
3. **Patch Decorators** - `@patch()` to replace ANTs functions with mocks
4. **Synthetic Data** - Small numpy arrays for integration tests

### Example: Testing with Mocks

```python
@patch('r2d2_base.ants.image_read')
def test_load_images_success(self, mock_image_read):
    # Create temporary files
    self.create_mock_nifti_files()

    # Mock ANTs return values
    mock_image_read.side_effect = [mock_reg, mock_template, mock_mask]

    # Run test
    result = r2d2_base.load_images(reg_path, template_path)

    # Verify behavior
    assert result["reg_image"] == mock_reg
```

## Continuous Integration

The repository includes a GitHub Actions workflow (`.github/workflows/test.yml`) that:

- Runs tests on multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Generates coverage reports
- Uploads results to Codecov
- Triggers on pushes and pull requests to main/develop branches

## Writing New Tests

When adding new functionality to R2D2, follow this pattern:

1. **Create test class** in `test_r2d2_base.py`
2. **Add setup/teardown** methods if needed for test fixtures
3. **Write unit tests** for individual functions with mocks
4. **Write integration tests** if the feature requires end-to-end testing
5. **Run tests locally** and verify coverage
6. **Update CLAUDE.md** if testing approach changes

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<what_is_being_tested>`

### Best Practices

- Use descriptive test names that explain the scenario
- Test both success and failure paths
- Mock external dependencies (ANTs, file I/O)
- Clean up temporary files in teardown methods
- Add docstrings to explain complex test scenarios
- Mark slow tests with `@pytest.mark.slow`

## Known Limitations

1. **ANTs Dependency** - Some integration tests require ANTs to be installed and are skipped by default
2. **Large Images** - Tests use small synthetic images to keep runtime fast
3. **Multiprocessing** - Parallel execution is tested via main_wrapper but not full Pool.map()

## Troubleshooting

### Tests fail with "ModuleNotFoundError: No module named 'ants'"

Install ANTs: `pip install antspyx`

### Tests fail with import errors

Make sure you're in the R2D2 directory: `cd /path/to/R2D2`

### Coverage report shows low coverage

Some lines may be difficult to test (e.g., multiprocessing main block). Focus on testing core logic.

### Mocks aren't working as expected

Check that patch paths match the import in `r2d2_base.py`, not where the function is defined.
