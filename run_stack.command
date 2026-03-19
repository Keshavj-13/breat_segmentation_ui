#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
"${DIR}/scripts/stack_up.sh"
echo
echo "Press Enter to close..."
read -r _
