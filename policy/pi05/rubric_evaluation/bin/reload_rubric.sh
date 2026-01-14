#!/usr/bin/env bash
set -euo pipefail

# Notify the evaluation server to snapshot + reload the rubric.
# Default server: http://127.0.0.1:8899
#
# Example:
#   bash policy/pi05/rubric_evaluation/bin/reload_rubric.sh
#   bash policy/pi05/rubric_evaluation/bin/reload_rubric.sh --host 127.0.0.1 --port 8899

HOST="127.0.0.1"
PORT="8899"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if command -v curl >/dev/null 2>&1; then
  curl -sS -X POST "http://${HOST}:${PORT}/api/reload" | cat
  echo
else
  python - <<PY
import urllib.request
req = urllib.request.Request("http://${HOST}:${PORT}/api/reload", method="POST")
print(urllib.request.urlopen(req).read().decode("utf-8"))
PY
fi