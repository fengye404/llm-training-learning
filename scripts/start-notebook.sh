#!/usr/bin/env bash
set -euo pipefail

# 一键启动 Jupyter Notebook。
# 用法：
#   ./scripts/start-notebook.sh
#   ./scripts/start-notebook.sh --dry-run
#   ./scripts/start-notebook.sh --install-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'USAGE'
一键启动 Notebook：
  ./scripts/start-notebook.sh

可选参数：
  --dry-run       只打印将要执行的命令，不启动
  --install-only  只检查并安装 notebook，不启动
  -h, --help      显示帮助
USAGE
}

DRY_RUN=false
INSTALL_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --install-only)
      INSTALL_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 $PYTHON_BIN，请先安装 Python 3。" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "项目目录: $ROOT_DIR"
echo "Python: $PYTHON_BIN"

if "$DRY_RUN"; then
  echo "[dry-run] 检查 notebook 包并执行: $PYTHON_BIN -m notebook"
  exit 0
fi

if ! "$PYTHON_BIN" -m pip show notebook >/dev/null 2>&1; then
  echo "检测到 notebook 未安装，正在安装..."
  "$PYTHON_BIN" -m pip install notebook
else
  echo "notebook 已安装。"
fi

if "$INSTALL_ONLY"; then
  echo "仅安装模式完成。"
  exit 0
fi

echo "正在启动 Jupyter Notebook（Ctrl+C 可停止）..."
exec "$PYTHON_BIN" -m notebook
