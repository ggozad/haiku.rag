#!/bin/bash
# Run integration tests against a local SeaweedFS instance.
#
# Brings up SeaweedFS via docker compose, runs every test marked `integration`,
# and tears down on exit. Extra args are forwarded to pytest:
#
#   ./scripts/run-integration-tests.sh
#   ./scripts/run-integration-tests.sh tests/test_s3_integration.py::test_s3_watcher_initial_sweep
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/tests/docker/docker-compose.s3.yml"

cleanup() {
    echo "==> Tearing down SeaweedFS"
    docker compose -f "$COMPOSE_FILE" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Bringing up SeaweedFS"
docker compose -f "$COMPOSE_FILE" up -d --wait

echo "==> Running integration tests"
uv run pytest -m integration "$@"
