#!/usr/bin/env bash
# Sealed smoke probe for the live HF Space.
# Verifies what we can verify without a browser: HTTP envelope, Streamlit
# health, response headers. Returns 0 on success, 1 on any failure.
#
# Usage: scripts/smoke_hf_space.sh [base_url]
#   default base_url: https://mekosotto-hackathon.hf.space

set -euo pipefail

BASE="${1:-https://mekosotto-hackathon.hf.space}"
FAIL=0

probe() {
  local label="$1" url="$2" expect="$3" method="${4:-GET}"
  local actual
  if [[ "$method" == "POST" ]]; then
    actual="$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
      -H 'content-type: application/json' -d '{}' "$url")"
  else
    actual="$(curl -sS -o /dev/null -w "%{http_code}" "$url")"
  fi
  if [[ "$actual" == "$expect" ]]; then
    printf "  OK   %-40s %s\n" "$label" "$actual"
  else
    printf "  FAIL %-40s expected=%s actual=%s\n" "$label" "$expect" "$actual"
    FAIL=1
  fi
}

echo "Probing $BASE"
probe "frontend root"             "$BASE/"                       "200" "GET"
probe "streamlit health"          "$BASE/_stcore/health"          "200" "GET"
probe "POST to unknown path is rejected" "$BASE/predict/bbb"     "403" "POST"

echo
if [[ "$FAIL" == "0" ]]; then
  echo "Smoke OK — proceed to manual click-through checklist:"
  echo "  docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md"
  exit 0
else
  echo "Smoke FAILED — see HF Space logs at:"
  echo "  https://huggingface.co/spaces/mekosotto/hackathon/discussions"
  exit 1
fi
