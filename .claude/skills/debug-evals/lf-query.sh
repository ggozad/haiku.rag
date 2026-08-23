#!/usr/bin/env bash
# Query the Logfire API. Key is read from ~/.logfire-read-key and never echoed.
# usage: lf-query.sh "<SQL>" [min_timestamp]
set -uo pipefail
KEY_FILE="$HOME/.logfire-read-key"
[ -r "$KEY_FILE" ] || { echo "missing $KEY_FILE"; exit 2; }
SQL="${1:?need SQL}"
MIN="${2:-$(date -u -v-2d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '2 days ago' +%Y-%m-%dT%H:%M:%SZ)}"
python3 - "$SQL" "$MIN" <<'PY'
import json, os, sys, urllib.request, urllib.error
sql, min_ts = sys.argv[1], sys.argv[2]
key = open(os.path.expanduser("~/.logfire-read-key")).read().strip()
# region comes from the key prefix: pylf_v2_<region>_...
parts = key.split("_")
region = parts[2] if len(parts) > 3 and parts[0] == "pylf" else "eu"
req = urllib.request.Request(
    f"https://logfire-{region}.pydantic.dev/v2/query",
    data=json.dumps({"sql": sql, "min_timestamp": min_ts}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
try:
    print(json.dumps(json.load(urllib.request.urlopen(req, timeout=120)), indent=2)[:6000])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:600]}")
PY
