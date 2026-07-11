#!/usr/bin/env bash
# Install exact optimap dev build required by cardiac_pipeline_v3.
# This commit corresponds to version 0.3.2.dev47+g52c156ebb.d20260501.

set -euo pipefail

COMMIT="52c156ebb16442c36cce756ef1b2bfc340257204"
URL="git+https://github.com/cardiacvision/optimap.git@${COMMIT}"

echo "Installing optimap from commit ${COMMIT}..."
echo "This may take several minutes (C++ extension build)."

# Prefer pip install in active environment
python3 -m pip install --no-build-isolation "${URL}"

# Verify
python3 -c "import optimap; print('optimap', optimap.__version__)"

echo "Done."
