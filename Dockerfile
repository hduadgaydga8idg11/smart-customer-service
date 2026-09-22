# syntax=docker/dockerfile:1
# =========================================================
# 智能客服 Agent「小智」应用镜像（Streamlit + RAG + Agent）
# 仅包含应用本身；Ollama 大模型由 docker-compose 独立服务提供
# =========================================================
FROM python:3.12-slim

# ---------- 系统依赖 ----------
# libgl1 / libglib2.0-0：OpenCV 运行时（RapidOCR / docling 使用）
# libgomp1：ONNX Runtime 与 PyTorch CPU 并行运算库
# 注：OCR 已从 Tesseract 切换为 RapidOCR（纯 pip + onnxruntime），无需安装 tesseract；
#     PPT 解析走 python-pptx 文本降级，无需 LibreOffice，显著控制镜像体积。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 运行时环境变量 ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    CHAT_DB_FILE=/app/persistent/chat_history.db \
    MODEL_CONFIG_PATH=/app/persistent/config.yaml

# pip 源（默认清华镜像加速国内构建；可在构建时覆盖：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple . ）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- 先装依赖（利用 Docker 层缓存，改代码不重装依赖） ----------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- 应用代码 ----------
COPY . .

# 持久化目录（compose 会把宿主机目录挂载进来）
RUN mkdir -p /app/persistent /app/docs /app/logs /app/chroma_db /app/models

EXPOSE 8501

# Streamlit 官方健康检查端点；start-period 给模型加载留时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status==200 else 1)"

# PYTHONUTF8=1 保证中文入口文件名在容器内被正确解析
CMD ["streamlit", "run", "智能客服助手.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
