#!/usr/bin/env bash
# ============================================================
#  离线 Docker 部署包一键启动（Linux / macOS / Windows Git Bash）
#  用法：./start.sh
#
#  说明：离线包内 Ollama 容器始终启动（提供本地 bge-m3 嵌入 + qwen2.5:1.5b 聊天）；
#  也可在页面「模型设置」切换为云端 API（Base URL + Key），无需在启动时选择模式。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "[错误] 未检测到 Docker，请先安装：https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[错误] Docker 守护进程未运行，请先启动 Docker 后重试。"
  exit 1
fi

echo "=== [1/2] 加载离线镜像 ==="
for f in images/*.tar; do
  [ -e "$f" ] || { echo "[错误] images/ 目录下没有找到镜像 tar 文件"; exit 1; }
  echo "  - $(basename "$f")"
  docker load -i "$f"
done

echo "=== [2/2] 启动服务 ==="
docker compose up -d

cat <<EOF

================================================================
 部署完成，浏览器访问： http://localhost:8501

 首次启动需等待 Ollama 与 bge-m3 模型加载，查看进度：
   docker logs -f ollama
 停止服务：docker compose down
 查看应用日志：docker logs -f smart-cs-bot

 若聊天模型走云端/内网 API，请确认目标网络可达，并在页面
 「模型设置」中检查 API Key 与接口地址。
================================================================
EOF
