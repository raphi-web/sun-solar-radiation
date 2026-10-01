#!/usr/bin/env bash
# Build and publish the sun wheel to PyPI.
#
# Usage:
#   scripts/publish_pypi.sh              # dry-run (build + twine check, no upload)
#   scripts/publish_pypi.sh --upload     # actually upload to PyPI
#   scripts/publish_pypi.sh --test       # upload to test.pypi.org
#
# Prerequisites:
#   - maturin installed (pip install maturin)
#   - twine installed (pip install twine)
#   - PyPI credentials configured (~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-dry-run}"

echo "==> Building manylinux wheel for CPython 3.12 (QGIS's interpreter)…"
# NOTE: PyPI only accepts manylinux wheels. We build WITHOUT --skip-auditwheel
# so auditwheel relabels the wheel as manylinux_2_39 and bundles the required
# shared libraries (GDAL, etc.) — the wheel is ~54MB but self-contained.
# For the QGIS plugin (which uses system GDAL), use scripts/build_qgis_plugin.sh
# instead, which uses --skip-auditwheel.
maturin build --release -i /usr/bin/python3

WHEEL="$(ls -t target/wheels/sun_solar_radiation-*-manylinux*.whl | head -1)"
echo "==> Built: $WHEEL"

echo "==> Checking wheel metadata with twine…"
twine check "$WHEEL"

case "$MODE" in
    --upload)
        echo "==> Uploading to PyPI (production)…"
        twine upload "$WHEEL"
        ;;
    --test)
        echo "==> Uploading to test.pypi.org…"
        twine upload --repository testpypi "$WHEEL"
        ;;
    dry-run|*)
        echo "==> Dry-run complete. Wheel is ready."
        echo "    To publish: scripts/publish_pypi.sh --upload"
        echo "    To test-publish: scripts/publish_pypi.sh --test"
        ;;
esac
