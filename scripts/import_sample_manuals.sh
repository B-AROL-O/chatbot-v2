#!/bin/bash

set -e
# set -x

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_INPUT_PATH=$PWD/data/input/manuals

# Override with ZIP_FILENAME=/path/to/archive.zip to use another archive.
ZIP_FILENAME=${ZIP_FILENAME:-$PWD/data/archive/sample-manuals.zip}

echo "DEBUG: SCRIPT_PATH=$SCRIPT_PATH"
echo "DEBUG: DATA_INPUT_PATH=$DATA_INPUT_PATH"
echo "DEBUG: ZIP_FILENAME=$ZIP_FILENAME"

if [ ! -e "${ZIP_FILENAME}" ]; then
    echo "ERROR: Cannot find ${ZIP_FILENAME}"
    exit 1
fi

mkdir -p "${DATA_INPUT_PATH}"

cd "$DATA_INPUT_PATH" && unzip -o "$ZIP_FILENAME"

# EOF
