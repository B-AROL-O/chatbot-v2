#!/bin/bash
#
# backup_data_folder.sh
#
# Archives the data/ folder into data/archive/, following the same naming
# convention as existing backups, e.g.:
#   data/archive/2026-07-10-bk-chatbotv2-data-hw2482.zip
#
# Usage:
#   ./scripts/backup_data_folder.sh

set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

HOST_FILE=".devcontainer/from_host/hostname"
if [[ -s "$HOST_FILE" ]]; then
	HOST_NAME="$(cat "$HOST_FILE")"
else
	HOST_NAME="$(hostname)"
fi

ARCHIVE_DIR="data/archive"
mkdir -p "$ARCHIVE_DIR"

ZIP_NAME="$(date +%Y-%m-%d)-bk-chatbotv2-data-${HOST_NAME}.zip"
ZIP_PATH="${ARCHIVE_DIR}/${ZIP_NAME}"

if [[ -e "$ZIP_PATH" ]]; then
	echo "Error: $ZIP_PATH already exists. Remove it or wait for tomorrow's date." >&2
	exit 1
fi

echo "Archiving data/ into $ZIP_PATH ..."
zip -r "$ZIP_PATH" data/ -x "data/archive/*"

echo "Done: $ZIP_PATH"

# EOF
