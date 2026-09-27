#!/usr/bin/env bash
# Download the HDF5 dataset parts and checksums from a GitHub Release into dist/ and verify them.
# Usage: scripts/download_data.sh OWNER/REPO [TAG]   (requires the GitHub CLI, gh)
set -euo pipefail
REPO="${1:?OWNER/REPO}"; TAG="${2:-v1.0.0}"
mkdir -p dist
gh release download "$TAG" --repo "$REPO" --pattern 'hev_gsp_*.h5' --pattern 'SHA256SUMS' --dir dist --clobber
cd dist && sha256sum -c SHA256SUMS
