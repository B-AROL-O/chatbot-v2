#!/bin/bash

# set -xe

# echo "DEBUG: BEGIN setup-mounts.sh"
# echo "DEBUG: HOME=$HOME"

# Make a file (or directory) on the host visible inside the container
#
# $1: Real file/directory on the host
# $2: Intermediate file/directory shared with the container
mount_host() {
    local real_path="$1"
    local mount_point="$2"

    mkdir -p "$(dirname "$mount_point")"
    if [ -d "$real_path" ]; then
        # Link the real dir so the mount sees the data
        ln -sfn "$real_path" "$mount_point"
    elif [ -f "$real_path" ]; then
        # Link the real file so the mount sees the data
        # echo "DEBUG: File $real_path: $(cat "$real_path")"
        ln -sfn "$real_path" "$mount_point"
    else
        # Create an empty dummy dir so the mount doesn't error out
        mkdir -p "$mount_point"
    fi
}

mount_host "/etc/hostname"   "./.devcontainer/from_host/hostname"
mount_host "$HOME/.claude"   "./.devcontainer/from_host/.claude"
mount_host "$HOME/.config"   "./.devcontainer/from_host/.config"
mount_host "$HOME/.local"    "./.devcontainer/from_host/.local"
mount_host "$HOME/.opencode" "./.devcontainer/from_host/.opencode"
mount_host "$HOME/.pi"       "./.devcontainer/from_host/.pi"

# echo "DEBUG: END setup-mounts.sh"

# EOF
