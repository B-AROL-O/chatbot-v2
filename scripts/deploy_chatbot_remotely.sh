#!/usr/bin/env bash
#
# deploy_chatbot_remotely.sh
#
# Pull and start a released chatbot image on a deployment/test host
# (e.g. deploy-host), then verify the container came up healthy.
#
# Modeled after the manual steps documented in docs/procedures/docker-deployment.md
# ("Deploying a released image to a remote host" / "Upgrading or rolling back").
#
# Run this ON the target host, from the directory containing docker-compose.yml
# and .env (see docs/procedures/docker-deployment.md for one-time setup):
#
#   ssh user@deploy-host
#   cd ~/chatbotv2-test
#   ~/chatbotv2-test/deploy_chatbot_remotely.sh 0.7.0-rc.2
#
# Or run it FROM your workstation against a remote host over SSH, without
# copying it over by hand first: set REMOTE_USER and REMOTE_HOST and the
# script re-invokes itself on that host via `ssh ... bash -s`, cd-ing into
# REMOTE_DIR (default: chatbotv2-test, relative to $HOME on the remote).
#
#   REMOTE_USER=user REMOTE_HOST=deploy-host ./deploy_chatbot_remotely.sh 0.7.0-rc.2
#
# Run with -h/--help for full usage.
#
# Requires: docker compose, curl. jq is used if present (falls back to raw output).
# For the SSH mode: ssh, plus password-less (key-based) access to the remote host.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [-h|--help] <version> [health_url]

Arguments:
  version     Image tag to deploy, e.g. 0.7.0-rc.2 (required)
  health_url  Health endpoint to poll (default: http://localhost:8000/health)

Options:
  -h, --help  Show this help message and exit

Environment (run over SSH from your workstation instead of on the host):
  REMOTE_USER  SSH user, e.g. user. Requires REMOTE_HOST too.
  REMOTE_HOST  SSH host, e.g. deploy-host. Requires REMOTE_USER too.
  REMOTE_DIR   Remote deployment directory, relative to \$HOME on the
               remote host unless absolute (default: chatbotv2-test)
  IMAGE_REPO   Image to pull, without the tag (default: ghcr.io/b-arol-o/chatbotv2)

Examples:
  $(basename "$0") 0.7.0-rc.2
  REMOTE_USER=user REMOTE_HOST=deploy-host $(basename "$0") 0.7.0-rc.2
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

VERSION="${1:-}"
HEALTH_URL="${2:-http://localhost:8000/health}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/b-arol-o/chatbotv2}"
RETRIES=30
SLEEP_SECONDS=2

if [[ -z "$VERSION" ]]; then
  usage
  exit 1
fi

REMOTE_DIR="${REMOTE_DIR:-chatbotv2-test}"

if [[ -n "${REMOTE_USER:-}" || -n "${REMOTE_HOST:-}" ]]; then
  [[ -n "${REMOTE_USER:-}" && -n "${REMOTE_HOST:-}" ]] \
    || die "Set both REMOTE_USER and REMOTE_HOST to run over SSH (got REMOTE_USER='${REMOTE_USER:-}' REMOTE_HOST='${REMOTE_HOST:-}')."
  command -v ssh >/dev/null 2>&1 || die "'ssh' is required but not installed."

  echo "=== Running on ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} via SSH ==="
  remote_args=$(printf '%q ' "$@")
  # Pipe this script's own source over SSH and run it with `bash -s` on the
  # remote host — no need to copy the file over there by hand first. Clear
  # REMOTE_USER/REMOTE_HOST so the remote copy takes the local (non-SSH)
  # branch instead of recursing.
  ssh_status=0
  # shellcheck disable=SC2029 # intentional: $REMOTE_DIR and $remote_args must
  # expand client-side, already shell-quoted (printf %q), to build the exact
  # command run remotely by `bash -s`.
  ssh "${REMOTE_USER}@${REMOTE_HOST}" \
    "unset REMOTE_USER REMOTE_HOST; cd $(printf '%q' "$REMOTE_DIR") && bash -s -- ${remote_args}" \
    < "$0" || ssh_status=$?

  # ssh itself exits 255 for connection/auth failures (unreachable host,
  # rejected key, etc.) — anything else is the remote script's own exit code,
  # already reported by its own error message, so just propagate it.
  if [[ $ssh_status -eq 255 ]]; then
    die "Could not reach or authenticate to ${REMOTE_USER}@${REMOTE_HOST} — check the host is up, DNS/VPN routing is in place, and your SSH key is authorized there (see docs/procedures/docker-deployment.md)."
  fi
  exit "$ssh_status"
fi

[[ -f docker-compose.yml ]] || die "docker-compose.yml not found in $(pwd) — cd to the deployment directory first"

command -v docker >/dev/null 2>&1 || die "'docker' is required but not installed."
command -v curl >/dev/null 2>&1 || die "'curl' is required but not installed."

export CHATBOT_IMAGE="${IMAGE_REPO}:${VERSION}"

echo "=== Pulling ${CHATBOT_IMAGE} ==="
if ! docker compose pull; then
  die "docker compose pull failed for ${CHATBOT_IMAGE}. If it said 'unauthorized' or 'pull access denied', the package may not be public or a stale ghcr.io login may be stored: see docs/procedures/release-chatbot.md 'Pulling the image'. If it couldn't reach the Docker daemon, check 'docker info' / that the Docker service is running. Otherwise confirm the tag exists in the registry."
fi

echo "=== Starting containers ==="
if ! docker compose up -d; then
  die "docker compose up -d failed. Run 'docker compose logs' for details, and check that docker-compose.yml and .env are present and valid in $(pwd)."
fi

echo "=== Waiting for ${HEALTH_URL} to report version ${VERSION} ==="
attempt=0
actual_version=""
while (( attempt < RETRIES )); do
  if response=$(curl -sf "$HEALTH_URL" 2>/dev/null); then
    actual_version=$(echo "$response" | grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' | sed -E 's/.*"([^"]+)"$/\1/')
    if [[ "$actual_version" == "$VERSION" ]]; then
      if command -v jq >/dev/null 2>&1; then
        echo "$response" | jq .
      else
        echo "$response"
      fi
      break
    fi
  fi
  attempt=$((attempt + 1))
  sleep "$SLEEP_SECONDS"
done

if [[ "$actual_version" != "$VERSION" ]]; then
  echo "--- docker compose ps ---"
  docker compose ps
  echo "--- recent logs ---"
  docker compose logs --tail=50
  die "Timed out waiting for $HEALTH_URL to report version=$VERSION (last seen: '${actual_version:-none}')"
fi

echo "=== docker compose ps ==="
docker compose ps

echo "=== Deployed ${CHATBOT_IMAGE} successfully ==="

# EOF
