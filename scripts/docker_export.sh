#!/usr/bin/env bash
# ============================================================
#  离线 Docker 部署包导出脚本（在【有网络的电脑】上运行一次）
#  产物：docker_deploy/ 目录，拷贝到目标电脑后运行 ./start.sh 即可离线部署
#
#  用法：bash scripts/docker_export.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到项目根目录

if ! command -v docker >/dev/null 2>&1; then
  echo "[错误] 未检测到 Docker，请先安装：https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[错误] Docker 守护进程未运行，请先启动 Docker。"
  exit 1
fi

echo "=== [1/5] 构建应用镜像（rag-cs-agent:latest）==="
docker build -t rag-cs-agent:latest .

echo "=== [2/5] 拉取 Ollama 镜像 ==="
docker pull ollama/ollama:latest

echo "=== [3/5] 拉取本地模型（qwen2.5:1.5b 聊天 + bge-m3 嵌入，约 2GB，请耐心等待）==="
mkdir -p docker_data/ollama
docker run --rm \
  -v "$(pwd)/docker_data/ollama:/root/.ollama" \
  ollama/ollama pull qwen2.5:1.5b
docker run --rm \
  -v "$(pwd)/docker_data/ollama:/root/.ollama" \
  ollama/ollama pull bge-m3

echo "=== [4/5] 导出镜像为 tar ==="
rm -rf docker_deploy
mkdir -p docker_deploy/images
docker save -o docker_deploy/images/rag-cs-agent.tar rag-cs-agent:latest
docker save -o docker_deploy/images/ollama.tar ollama/ollama:latest

echo "=== [5/5] 归集部署文件到 docker_deploy/ ==="
cp scripts/docker/docker-compose.deploy.yml docker_deploy/docker-compose.yml
cp scripts/docker/start.sh docker_deploy/start.sh
chmod +x docker_deploy/start.sh

# 知识库与业务数据
[ -d chroma_db ] && cp -r chroma_db docker_deploy/
[ -d data ] && cp -r data docker_deploy/
[ -d docs ] && cp -r docs docker_deploy/
# 配置与会话
cp config.yaml docker_deploy/
[ -f .env ] && cp .env docker_deploy/
[ -f chat_history.db ] && cp chat_history.db docker_deploy/
# Ollama 模型（bge-m3 已拉取到 docker_data/ollama）
cp -r docker_data docker_deploy/
# Rerank 权重（可选，本地有则打包；无则目标电脑首次用 rerank 时需联网下载）
[ -d models ] && cp -r models docker_deploy/ || mkdir -p docker_deploy/models

cat <<EOF

==============================================================
 ✅ 离线部署包已生成：docker_deploy/ 目录（约 3~4GB）

 1. 把整个 docker_deploy/ 目录拷到目标电脑（U 盘需 exFAT/NTFS）
 2. 目标电脑安装 Docker 后，进入 docker_deploy/ 运行：
      ./start.sh
 3. 浏览器访问：http://localhost:8501

 ⚠️ 安全提醒：部署包内含 config.yaml 与 .env（API Key、访问令牌）
    及 chat_history.db 历史会话，属于敏感数据——仅限受控环境内
    拷贝使用，请勿上传到任何公开位置或转发给无关人员。
 说明：若 config.yaml 配置了内网/私有 API 地址，目标电脑需能
       访问对应网络才能使用聊天功能；Rerank 若未打包，首次使用需联网。
==============================================================
EOF
