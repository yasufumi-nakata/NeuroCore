#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
[ -x .venv/bin/python ] || { echo 'Run sh Install-NeuroCore.sh first' >&2; exit 1; }
SMAC_HOST="${1:-smac20}"
SMAC_USER="${2:-your_ssh_username}"
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=yes -N -L '127.0.0.1:18010:127.0.0.1:8011' "${SMAC_USER}@${SMAC_HOST}" &
SSH_PID=$!
trap 'kill "$SSH_PID" 2>/dev/null || true' EXIT HUP INT TERM
sleep 2
kill -0 "$SSH_PID" 2>/dev/null || { echo 'SSH failed: check VPN, key, known_hosts' >&2; exit 1; }
echo 'Open http://127.0.0.1:8788'
.venv/bin/python -m neurocore.cli serve --desktop --remote-model http://127.0.0.1:18010 --port 8788
