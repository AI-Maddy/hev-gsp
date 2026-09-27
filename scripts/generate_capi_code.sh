#!/usr/bin/env bash
# Regenerate ara::com proxy/skeleton code from adaptive/*.arxml with AUTOSAR CAPI's aragen.
# Usage: scripts/generate_capi_code.sh /path/to/capi   (clone of github.com/AUTOSAR/capi, tag v1.0.0)
# aragen needs lxml==4.9.x, Jinja2, jsonpath, xmltodict (see capi/isoft/ara-gen/requirements.txt).
set -euo pipefail
CAPI="${1:?path to CAPI clone}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CAPI/isoft/ara-gen"
for e in energy_mgmt telemetry; do
  python -m generator -g CODE -e "/HEVApps/exe/$e" -o "$HERE/adaptive/generated/$e" ../arxmls "$HERE/adaptive"
done
