#!/bin/bash

# set -xe

echo "DEBUG: BEGIN $0"
echo "DEBUG: HOME=$HOME"

FROM_HOST=/opt/from_host

# asdf shims (node/npm) are not on PATH in non-interactive shells
export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$PATH"

ls -la "$FROM_HOST"

echo "INFO: Setup Claude Code"

if ! command -v claude &> /dev/null; then
    echo "DEBUG: Claude Code not found in PATH, let us install it"
    npm install -g @anthropic-ai/claude-code
fi
if [ -e "$FROM_HOST/.claude/settings.json" ]; then
    if [ ! -e "$HOME/.claude/settings.json" ]; then
        echo "DEBUG: Copying Claude Code settings from $(cat $FROM_HOST/hostname)"
        mkdir -p "$HOME/.claude"
        cp "$FROM_HOST/.claude/settings.json" "$HOME/.claude/settings.json"
    fi
fi

echo "INFO: Setup OpenCode"

if ! command -v opencode &> /dev/null; then
    echo "DEBUG: OpenCode not found in PATH, let us install it"
    curl -fsSL https://opencode.ai/install | bash
fi
mkdir -p "$HOME/.config/opencode"
if [ -e "$FROM_HOST/.config/opencode/opencode.jsonc" ]; then
    if [ ! -e "$HOME/.config/opencode/opencode.jsonc" ]; then
        echo "DEBUG: Copying OpenCode configuration from $(cat $FROM_HOST/hostname)"
        cp "$FROM_HOST/.config/opencode/opencode.jsonc" "$HOME/.config/opencode/opencode.jsonc"
    fi
fi
if [ ! -e "$HOME/.config/opencode/tui.json" ]; then
    echo "DEBUG: Remap OpenCode shortcut to Ctrl-O"
    echo '{"keybinds": {"command_list": "ctrl+o"}}' > "$HOME/.config/opencode/tui.json"
fi
if [ -e "$FROM_HOST/.local/share/opencode/auth.json" ]; then
    if [ ! -e "$HOME/.local/share/opencode/auth.json" ]; then
        echo "DEBUG: Copying OpenCode authorization from $(cat $FROM_HOST/hostname)"
        mkdir -p "$HOME/.local/share/opencode"
        cp "$FROM_HOST/.local/share/opencode/auth.json" "$HOME/.local/share/opencode/auth.json"
    fi
fi
# ls -la "$HOME/.config/opencode"
# ls -la "$HOME/.local/share/opencode"

# set -x

echo "INFO: Setup pi.dev"

if ! command -v pi &> /dev/null; then
    echo "DEBUG: pi not found in PATH, let us install it"
    # curl -fsSL https://pi.dev/install.sh | sh
    npm install -g @earendil-works/pi-coding-agent
fi
if [ -e "$FROM_HOST/.pi/agent/models.json" ]; then
    if [ ! -e "$HOME/.pi/agent/models.json" ]; then
        echo "DEBUG: Copying pi.dev models.json from $(cat $FROM_HOST/hostname)"
        mkdir -p "$HOME/.pi/agent"
        cp "$FROM_HOST/.pi/agent/models.json" "$HOME/.pi/agent/"
    fi
fi
if [ -e "$FROM_HOST/.pi/agent/settings.json" ]; then
    if [ ! -e "$HOME/.pi/agent/settings.json" ]; then
        echo "DEBUG: Copying pi.dev settings.json from $(cat $FROM_HOST/hostname)"
        mkdir -p "$HOME/.pi/agent"
        cp "$FROM_HOST/.pi/agent/settings.json" "$HOME/.pi/agent/"
    fi
fi

echo "INFO: Install PrinceXML"

if ! command -v prince &> /dev/null; then
    echo "DEBUG: PrinceXML not found, installing..."
    sudo apt-get update -qq
    PRINCE_DEB=$(mktemp --suffix=.deb)
    curl -fsSL "https://www.princexml.com/download/prince_16.2-1_ubuntu24.04_amd64.deb" -o "$PRINCE_DEB"
    sudo apt-get install -y "$PRINCE_DEB"
    rm -f "$PRINCE_DEB"
fi

echo "INFO: Install rclone"

if ! command -v rclone &> /dev/null; then
    echo "DEBUG: rclone not found, installing..."
    curl -fsSL https://rclone.org/install.sh | sudo bash
fi

echo "INFO: Setup tau"

if ! command -v tau &>/dev/null; then
	echo "DEBUG: tau not found in PATH, let us install it"
	uv tool install tau-ai
fi

echo "INFO: Setup shell completions for CLI tools"

# bash-completion (already installed) lazy-loads completion scripts named after
# each command from the XDG user dir. Dropping files here means new terminals
# pick up completion automatically, no ~/.bashrc edits required.
COMPLETION_DIR="$HOME/.local/share/bash-completion/completions"
mkdir -p "$COMPLETION_DIR"

if command -v uv &> /dev/null; then
    uv generate-shell-completion bash > "$COMPLETION_DIR/uv"
fi
if command -v docker &> /dev/null; then
    docker completion bash > "$COMPLETION_DIR/docker"
fi
if command -v gh &> /dev/null; then
    gh completion -s bash > "$COMPLETION_DIR/gh"
fi
if command -v kubectl &> /dev/null; then
    kubectl completion bash > "$COMPLETION_DIR/kubectl"
fi
if command -v rclone &> /dev/null; then
    rclone completion bash "$COMPLETION_DIR/rclone"
fi

echo "INFO: Fix PATH for login shells (e.g. tmux)"

# /etc/bash.bashrc sources asdf.sh, putting node/npm/claude/pi shims on PATH.
# But /etc/profile.d/00-restore-env.sh (sourced from /etc/profile, AFTER
# /etc/bash.bashrc, for every *login* shell) unconditionally overwrites PATH
# with a snapshot taken at image build time, wiping the asdf additions back
# out. tmux starts panes as login shells by default (empty default-command),
# so `claude`/`node`/`npm` silently disappear inside tmux even though they
# work fine in VS Code's own (non-login) integrated terminal. Re-source
# asdf.sh from ~/.bashrc, which ~/.profile runs *after* the reset script, so
# login shells get the shims back too.
BASHRC_ASDF_MARKER="# asdf (chatbotv2: restore PATH after login-shell PATH reset, see 00-restore-env.sh)"
if ! grep -qF "$BASHRC_ASDF_MARKER" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "$BASHRC_ASDF_MARKER"
        # SC2016 is intentional: the single-quoted form is written literally into
        # ~/.bashrc so the login shell is the one that expands $HOME later.
        # shellcheck disable=SC2016
        echo '. "$HOME/.asdf/asdf.sh"'
    } >> "$HOME/.bashrc"
fi

echo "DEBUG: END $0"

# EOF
