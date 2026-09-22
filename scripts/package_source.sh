#!/usr/bin/env bash
# ============================================================
#  打包部署所需的「源码 + 数据」（排除虚拟环境、git、缓存等无用文件）
#  生成 smart-cs-agent-deploy.tar.gz（约 2~3MB），拷贝到目标电脑
#  解压后即可 docker compose up -d --build（在线构建部署）
#
#  用法：bash scripts/package_source.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # 切到项目根目录

OUT="smart-cs-agent-deploy.tar.gz"
rm -f "$OUT"
TMP_OUT="$(mktemp "${TMPDIR:-/tmp}/smart-cs-agent-deploy.XXXXXX")"
trap 'rm -f "$TMP_OUT"' EXIT

tar czf "$TMP_OUT" \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='logs' \
  --exclude='*.log' \
  --exclude='*.pyc' \
  --exclude='docker_deploy' \
  --exclude='docker_data' \
  --exclude='models' \
  --exclude='项目架构图.html' \
  --exclude='cover.jpg' \
  --exclude="$OUT" \
  --exclude="./$OUT" \
  .

mv "$TMP_OUT" "$OUT"
trap - EXIT

echo ""
echo "=============================================================="
echo " ✅ 打包完成：$OUT"
echo "    大小：$(du -h "$OUT" | cut -f1)"
echo ""
echo " 部署步骤："
echo "   1. 把 $OUT 拷到目标电脑并解压"
echo "   2. 目标电脑安装 Docker，进入解压目录运行："
echo "      docker compose up -d --build"
echo "   3. 浏览器访问 http://localhost:8501"
echo ""
echo " 说明：压缩包内含 .env（API Key），注意保管，勿外传。"
echo "=============================================================="
