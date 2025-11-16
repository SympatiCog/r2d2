# Installation Checks Added to R2M2 Numba Implementation

## Summary

All Numba-related scripts now include startup checks that verify ANTs (ANTsPy) is installed before attempting to run. If ANTs is not available, the scripts will gracefully exit with clear installation instructions.

## Files Modified

1. **r2m2_numba.py** - Core Numba implementation module
2. **benchmark_numba.py** - Performance benchmarking script
3. **test_numba_quick.py** - Quick smoke test script

## Behavior

When any of these scripts are run without ANTs installed, they will:

1. ✅ Detect the missing ANTs dependency
2. ✅ Display a clear, formatted error message
3. ✅ Provide multiple installation methods (pip, conda, source)
4. ✅ Include platform-specific compiler requirements
5. ✅ Exit gracefully with status code 1

## Example Output

```
======================================================================
ERROR: ANTs (ANTsPy) is not installed
======================================================================

ANTs is required for R2M2 image processing operations.

Installation instructions:

1. Using pip (recommended):
   pip install antspyx

2. Using conda:
   conda install -c conda-forge antspyx

3. From source (advanced):
   git clone https://github.com/ANTsX/ANTsPy
   cd ANTsPy
   pip install .

Note: ANTsPy installation may take 10-20 minutes as it compiles
C++ code. Make sure you have a C++ compiler installed:
  - macOS: Install Xcode Command Line Tools
  - Linux: Install build-essential or equivalent
  - Windows: Install Visual Studio Build Tools

For more information, visit:
  https://github.com/ANTsX/ANTsPy
======================================================================
```

## Implementation Details

The check is performed at import time using a try/except block:

```python
import sys

# Check for ANTs availability
try:
    import ants
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: ANTs (ANTsPy) is not installed")
    print("=" * 70)
    # ... detailed instructions ...
    sys.exit(1)
```

This ensures:
- **Immediate feedback**: Users know right away if they're missing dependencies
- **No cryptic errors**: Instead of Python ImportError tracebacks, users get helpful instructions
- **Graceful exit**: Scripts exit cleanly without leaving partial state

## Testing

All checks have been verified to work correctly:

```bash
# Test module import
python -c "from r2m2_numba import compute_r2m2_numba"
# Exit code: 1, displays installation instructions

# Test benchmark script
python benchmark_numba.py --help
# Exit code: 1, displays installation instructions

# Test quick test script
python test_numba_quick.py
# Exit code: 1, displays installation instructions
```

## User Experience

### Without ANTs installed:
- Clear error message
- Actionable installation instructions
- Exit code 1 for scripting compatibility

### With ANTs installed:
- Scripts run normally
- No additional overhead
- Seamless operation

## Future Enhancements

Potential improvements:

1. **Version checking**: Verify minimum ANTsPy version
2. **Dependency health check**: Test ANTs functionality (can load images, etc.)
3. **Auto-installation option**: Offer to install ANTsPy if missing
4. **Offline mode**: Support operation without ANTs for certain functions

## Notes

- The check happens at **import time**, not runtime
- This prevents wasted computation before discovering missing dependencies
- Exit code 1 allows shell scripts to detect installation issues
- Instructions are platform-aware (macOS/Linux/Windows compiler requirements)
