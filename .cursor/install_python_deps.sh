#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP_FILE="$ROOT_DIR/.cursor/.requirements.sha256"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for this project." >&2
  exit 1
fi

REQ_HASH="$(sha256sum "$ROOT_DIR/requirements.txt" | awk '{print $1}')"
PREV_HASH=""
if [[ -f "$STAMP_FILE" ]]; then
  PREV_HASH="$(<"$STAMP_FILE")"
fi

if [[ "$PREV_HASH" != "$REQ_HASH" ]]; then
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user -r "$ROOT_DIR/requirements.txt"
  printf "%s\n" "$REQ_HASH" > "$STAMP_FILE"
else
  echo "requirements.txt unchanged; skipping pip install."
fi

# Pre-warm rembg model cache so first /remove-bg call starts faster.
python3 - <<'PY'
from rembg import new_session

try:
    new_session("isnet-general-use")
except Exception:
    new_session("u2net")

print("rembg model cache prepared")
PY
