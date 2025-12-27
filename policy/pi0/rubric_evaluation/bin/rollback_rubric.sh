#!/usr/bin/env bash
set -euo pipefail

# Roll back to a previous rubric version.
#
# Example:
#   bash policy/pi0/rubric_evaluation/bin/rollback_rubric.sh --version 1734680000_deadbeef
#
# Default server: http://127.0.0.1:8899

HOST="127.0.0.1"
PORT="8899"
VERSION_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --version) VERSION_ID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${VERSION_ID}" ]]; then
  echo "Missing --version <VERSION_ID>"
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  curl -sS -X POST "http://${HOST}:${PORT}/api/rollback/${VERSION_ID}" | cat
  echo
else
  python - <<PY
import urllib.request
req = urllib.request.Request("http://${HOST}:${PORT}/api/rollback/${VERSION_ID}", method="POST")
print(urllib.request.urlopen(req).read().decode("utf-8"))
PY
fi