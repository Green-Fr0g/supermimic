#!/usr/bin/env bash

# SuperMimic 统一入口：环境管理、任务列表、supermimic_rl 训练和回放。

set -euo pipefail

SUPERMIMIC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUPERMIMIC_PATH}"

require_command() {
    command -v "$1" >/dev/null 2>&1 || { echo "[错误] 找不到命令 '$1'。" >&2; exit 1; }
}

case "${1:-}" in
    -i|--install)
        shift
        require_command uv
        exec uv sync --locked "$@"
        ;;
    -l|--list)
        shift
        require_command uv
        exec uv run --locked isaaclab list_envs "$@"
        ;;
    -t|--train)
        shift
        require_command uv
        exec uv run --locked python scripts/supermimic/train.py "$@"
        ;;
    -p|--play)
        shift
        require_command uv
        exec uv run --locked python scripts/supermimic/play.py "$@"
        ;;
    -h|--help|"")
        cat <<'EOF'
用法：
  ./supermimic.sh -i                         安装项目环境
  ./supermimic.sh -l                         列出已注册任务
  ./supermimic.sh -t --profile smoke        使用 supermimic_rl 训练
  ./supermimic.sh -t --profile paper        使用 paper 参数训练
  ./supermimic.sh -p --checkpoint <路径>     回放 checkpoint

示例：
  ./supermimic.sh -t --profile smoke --headless
  ./supermimic.sh -p --checkpoint logs/supermimic_rl/checkpoint_final.pt --headless
EOF
        ;;
    *)
        echo "[错误] 未知选项 '$1'，使用 './supermimic.sh --help' 查看用法。" >&2
        exit 1
        ;;
esac
