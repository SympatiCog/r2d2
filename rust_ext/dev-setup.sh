#!/usr/bin/env bash
#
# dev-setup.sh — build and test the r2d2_rust extension in one command.
#
# Creates (or reuses) a local virtualenv, builds the Rust extension into it
# with maturin, and runs the test suite. Re-run it any time after pulling to
# rebuild against new source.
#
# Usage:
#   ./dev-setup.sh              # create/reuse .venv, build, run tests
#   ./dev-setup.sh --no-tests   # build only, skip pytest
#   SKIP_VENV=1 ./dev-setup.sh  # use the already-active environment, don't create .venv
#
# Notes:
#   - Run from anywhere; the script cd's to its own directory (rust_ext/).
#   - maturin refuses to run when both VIRTUAL_ENV and CONDA_PREFIX are set, so
#     this script builds into the venv via an explicit interpreter and unsets
#     CONDA_PREFIX for the build to avoid the "unset one of them" error.
set -euo pipefail

cd "$(dirname "$0")"

RUN_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --no-tests) RUN_TESTS=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- pick the Python interpreter / environment -----------------------------
if [[ "${SKIP_VENV:-0}" == "1" ]]; then
  PY="$(command -v python3 || command -v python)"
  echo ">> Using already-active interpreter: $PY"
else
  if [[ ! -d .venv ]]; then
    echo ">> Creating virtualenv at rust_ext/.venv"
    python3 -m venv .venv
  fi
  PY="$PWD/.venv/bin/python"
  echo ">> Using venv interpreter: $PY"
fi

# --- tooling ---------------------------------------------------------------
echo ">> Installing build/test tooling (maturin, numpy, pytest)"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install maturin numpy pytest >/dev/null

# --- build + install the extension into this interpreter -------------------
# Build a wheel and install it with the SAME interpreter we test with, so the
# install and the tests can never land in different environments. CONDA_PREFIX
# is cleared for the build step only (maturin rejects venv+conda both set).
echo ">> Building the Rust extension (release) and installing the wheel"
rm -rf dist
env -u CONDA_PREFIX "$PY" -m maturin build --release --out dist --interpreter "$PY"
"$PY" -m pip install --force-reinstall --no-deps dist/*.whl

# --- sanity check + tests --------------------------------------------------
"$PY" - <<'EOF'
import r2d2_rust
print(">> import r2d2_rust OK:", r2d2_rust.__file__)
EOF

if [[ "$RUN_TESTS" == "1" ]]; then
  echo ">> Running cargo tests (pure-Rust kernel)"
  cargo test --release
  echo ">> Running Python validation tests"
  "$PY" -m pytest tests/test_kernel.py -v
  echo ">> Done. (The ANTs parity test is skipped unless ANTsPy is installed.)"
else
  echo ">> Build complete (tests skipped)."
fi
