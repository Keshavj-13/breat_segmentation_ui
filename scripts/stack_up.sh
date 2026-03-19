#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/.runtime"
ENV_FILE="${ROOT_DIR}/.env.local"

mkdir -p "${RUNTIME_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.local.example and fill your values first."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

SSH_LOCAL_PORT="${SSH_LOCAL_PORT:-8001}"
SSH_REMOTE_PORT="${SSH_REMOTE_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-5000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
START_FRONTEND_DEV="${START_FRONTEND_DEV:-true}"
WORKERS="${WORKERS:-http://127.0.0.1:${SSH_LOCAL_PORT}}"
BACKEND_ENTRY="${BACKEND_ENTRY:-xai_server.js}"
HEALTH_RETRIES="${HEALTH_RETRIES:-45}"
HEALTH_SLEEP="${HEALTH_SLEEP:-1}"

if [[ -z "${SSH_USER:-}" || -z "${SSH_LOGIN_HOST:-}" || -z "${SSH_REMOTE_COMPUTE_HOST:-}" ]]; then
  echo "Missing SSH settings in .env.local. Required: SSH_USER, SSH_LOGIN_HOST, SSH_REMOTE_COMPUTE_HOST"
  exit 1
fi

require_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "Required command not found: $c"
    exit 1
  fi
}

require_cmd curl
require_cmd node
require_cmd npm
require_cmd ssh

if [[ -n "${SSH_PASSWORD:-}" ]]; then
  require_cmd sshpass
fi

stop_if_pid_stale() {
  local name="$1"
  local pid_file="${RUNTIME_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${name} appears to be running already (PID ${pid})."
      echo "Run: bash scripts/stack_down.sh"
      exit 1
    fi
    rm -f "${pid_file}"
  fi
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1
  else
    ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .
  fi
}

check_port_free() {
  local name="$1"
  local port="$2"
  if port_in_use "${port}"; then
    echo "${name} port ${port} is already in use."
    echo "Free it or change ${name} port in .env.local."
    exit 1
  fi
}

cleanup_on_error() {
  local code="$?"
  echo
  echo "Startup failed. Cleaning up partial processes..."

  for name in frontend backend tunnel; do
    local pid_file="${RUNTIME_DIR}/${name}.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
      rm -f "${pid_file}"
    fi
  done

  exit "${code}"
}

trap cleanup_on_error ERR INT TERM

wait_for_url() {
  local label="$1"
  local url="$2"
  local retries="${3:-$HEALTH_RETRIES}"
  local sleep_s="${4:-$HEALTH_SLEEP}"

  local i
  for ((i = 1; i <= retries; i++)); do
    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then
      echo "[ok] ${label}"
      return 0
    fi
    sleep "${sleep_s}"
  done

  echo "[fail] ${label} after ${retries} retries: ${url}"
  return 1
}

normalize_worker() {
  local first
  first="$(echo "${WORKERS}" | cut -d',' -f1 | xargs)"

  if [[ -z "${first}" ]]; then
    echo "http://127.0.0.1:${SSH_LOCAL_PORT}"
    return
  fi

  if [[ "${first}" =~ ^[0-9]+$ ]]; then
    echo "http://127.0.0.1:${first}"
    return
  fi

  if [[ "${first}" =~ ^[^/[:space:]]+:[0-9]+$ ]]; then
    echo "http://${first}"
    return
  fi

  echo "${first}"
}

stop_if_pid_stale "tunnel"
stop_if_pid_stale "backend"
stop_if_pid_stale "frontend"

check_port_free "Tunnel local" "${SSH_LOCAL_PORT}"
check_port_free "Backend" "${GATEWAY_PORT}"
if [[ "${START_FRONTEND_DEV,,}" == "true" ]]; then
  check_port_free "Frontend" "${FRONTEND_PORT}"
fi

TUNNEL_LOG="${RUNTIME_DIR}/tunnel.log"
BACKEND_LOG="${RUNTIME_DIR}/backend.log"
FRONTEND_LOG="${RUNTIME_DIR}/frontend.log"

if [[ -n "${SSH_PASSWORD:-}" ]]; then
  SSHPASS="${SSH_PASSWORD}" sshpass -e ssh \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -N -L "${SSH_LOCAL_PORT}:${SSH_REMOTE_COMPUTE_HOST}:${SSH_REMOTE_PORT}" \
    "${SSH_USER}@${SSH_LOGIN_HOST}" \
    >"${TUNNEL_LOG}" 2>&1 &
else
  ssh \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=accept-new \
    -N -L "${SSH_LOCAL_PORT}:${SSH_REMOTE_COMPUTE_HOST}:${SSH_REMOTE_PORT}" \
    "${SSH_USER}@${SSH_LOGIN_HOST}" \
    >"${TUNNEL_LOG}" 2>&1 &
fi

echo "$!" > "${RUNTIME_DIR}/tunnel.pid"
echo "Tunnel started (PID $(cat "${RUNTIME_DIR}/tunnel.pid"))"

WORKER_BASE_URL="$(normalize_worker)"
wait_for_url "Compute worker health" "${WORKER_BASE_URL}/health"

(
  cd "${ROOT_DIR}/backend"
  PORT="${GATEWAY_PORT}" WORKERS="${WORKERS}" node "${BACKEND_ENTRY}" >"${BACKEND_LOG}" 2>&1 &
  echo "$!" > "${RUNTIME_DIR}/backend.pid"
)

echo "Backend started (PID $(cat "${RUNTIME_DIR}/backend.pid"))"
wait_for_url "Gateway health" "http://127.0.0.1:${GATEWAY_PORT}/api/health"
wait_for_url "Gateway compute connectivity" "http://127.0.0.1:${GATEWAY_PORT}/api/gpu"

if [[ "${START_FRONTEND_DEV,,}" == "true" ]]; then
  (
    cd "${ROOT_DIR}/frontend"
    npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}" >"${FRONTEND_LOG}" 2>&1 &
    echo "$!" > "${RUNTIME_DIR}/frontend.pid"
  )
  echo "Frontend started (PID $(cat "${RUNTIME_DIR}/frontend.pid"))"
  wait_for_url "Frontend" "http://127.0.0.1:${FRONTEND_PORT}"
fi

trap - ERR INT TERM

echo
echo "Stack is up."
if [[ "${START_FRONTEND_DEV,,}" == "true" ]]; then
  echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
fi
echo "Backend:  http://127.0.0.1:${GATEWAY_PORT}"
echo "Worker:   ${WORKER_BASE_URL}"
echo
echo "Logs:"
echo "  ${TUNNEL_LOG}"
echo "  ${BACKEND_LOG}"
if [[ "${START_FRONTEND_DEV,,}" == "true" ]]; then
  echo "  ${FRONTEND_LOG}"
fi
echo
echo "Stop everything with: bash scripts/stack_down.sh"
