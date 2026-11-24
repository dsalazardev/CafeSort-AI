#!/usr/bin/env bash
# Source this file from Ubuntu/WSL before running project commands.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Uso: source scripts/wsl_env.sh" >&2
  exit 2
fi

if [[ -f "$HOME/.local/bin/env" ]]; then
  # uv's installer adds the Linux uv binary to PATH through this file.
  source "$HOME/.local/bin/env"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/cafesort-ai}"
