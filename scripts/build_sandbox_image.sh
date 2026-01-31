#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker build -t code-sandbox:py311 -f "${ROOT_DIR}/backend/sandbox/Dockerfile" "${ROOT_DIR}/backend/sandbox"
