#!/usr/bin/env bash

# SuperMimic 统一入口脚本。
# -i/--install: 根据 uv.lock 创建环境并安装依赖
# -l/--list:    列出已注册任务
# -t/--train:   使用 rsl_rl 运行训练
# -p/--play:    运行策略回放

set -euo pipefail

SUPERMIMIC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SUPERMIMIC_PATH}"

die() {
    echo "[错误] $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "找不到命令 '$1'，请先安装后重试。"
}

install_project() {
    require_command uv

    echo "[信息] 根据 uv.lock 同步 Python 3.12 环境及本地 RSL-RL 依赖。"
    uv sync --locked "$@"
    echo "[完成] SuperMimic 安装完成，环境位于 ${SUPERMIMIC_PATH}/.venv。"
}

run_train() {
    require_command uv
    local args=("$@")
    local has_rl_library=0
    for arg in "${args[@]}"; do
        if [[ "${arg}" == "--rl_library" || "${arg}" == --rl_library=* ]]; then
            has_rl_library=1
            break
        fi
    done
    if [[ ${has_rl_library} -eq 0 ]]; then
        args=(--rl_library rsl_rl "${args[@]}")
    fi
    exec uv run --locked isaaclab train "${args[@]}"
}

run_play() {
    require_command uv
    local args=("$@")
    local has_rl_library=0
    local has_visualizer=0
    for arg in "${args[@]}"; do
        if [[ "${arg}" == "--rl_library" || "${arg}" == --rl_library=* ]]; then
            has_rl_library=1
        fi
        if [[ "${arg}" == "--visualizer" || "${arg}" == --visualizer=* || "${arg}" == "--viz" || "${arg}" == --viz=* ]]; then
            has_visualizer=1
        fi
    done
    if [[ ${has_rl_library} -eq 0 ]]; then
        args=(--rl_library rsl_rl "${args[@]}")
    fi
    if [[ ${has_visualizer} -eq 0 ]]; then
        args+=(--visualizer newton_gl)
    fi
    exec uv run --locked isaaclab play "${args[@]}"
}

list_tasks() {
    require_command uv
    exec uv run --locked isaaclab list_envs "$@"
}

case "${1:-}" in
    -i|--install)
        shift
        install_project "$@"
        ;;
    -t|--train)
        shift
        run_train "$@"
        ;;
    -p|--play)
        shift
        run_play "$@"
        ;;
    -l|--list)
        shift
        list_tasks "$@"
        ;;
    -h|--help|"")
        cat <<'EOF'
用法：
  ./supermimic.sh -i [uv sync 参数]     根据 uv.lock 安装环境
  ./supermimic.sh -l                    列出本项目已注册任务
  ./supermimic.sh -l --all              列出所有 Isaac Lab 任务
  ./supermimic.sh -t --task <任务名称>   使用 rsl_rl 训练
  ./supermimic.sh -p --task <任务名称>   播放已训练策略，默认使用 newton_gl

示例（Isaac Lab 内置任务）：
  ./supermimic.sh -i
  ./supermimic.sh -t --task Isaac-Cartpole-Direct physics=newton_mjwarp
  ./supermimic.sh -p --task Isaac-Cartpole-Direct --checkpoint <检查点路径> physics=newton_mjwarp

安装参数传给 uv sync；训练、回放和列表参数传给对应的 Isaac Lab 入口。
可用 --rl_library 覆盖默认 RL 框架，回放时可用 --visualizer 或 --viz 覆盖默认可视化器。
EOF
        ;;
    *)
        die "未知选项 '$1'。使用 './supermimic.sh --help' 查看用法。"
        ;;
esac
