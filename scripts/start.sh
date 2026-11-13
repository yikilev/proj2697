#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker not found. Install Docker with Compose and start Docker." >&2
    exit 1
fi
docker compose version
if ! docker info >/dev/null; then
    echo "Docker is unavailable. Start Docker and check access to its daemon." >&2
    exit 1
fi

compose() { docker compose -p proj2697 -f compose.yaml "$@"; }
cleanup() {
    status=$?
    trap - EXIT INT TERM
    echo "Stopping backend (SQLite volume is preserved)..."
    compose stop --timeout 10 || echo "Stop failed. Run: docker compose -p proj2697 -f compose.yaml stop" >&2
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! compose up --build --wait --wait-timeout 120; then
    echo "Backend failed to start or /health did not become ready within 120 seconds. Details:" >&2
    compose ps -a || true
    compose logs --tail 80 backend || true
    exit 1
fi

url=http://localhost:8000/docs
echo "Swagger: $url"
echo "Press Ctrl+C to stop the backend and keep the data."
# Some Linux browser launchers wait until the browser closes; keep streaming logs.
(
    if [[ "$(uname -s)" == Darwin ]]; then
        open "$url" || echo "Could not open the browser. Open $url manually." >&2
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" || echo "Could not open the browser. Open $url manually." >&2
    else
        echo "No browser launcher found. Open $url manually." >&2
    fi
) &
compose logs --follow --tail 100 backend
