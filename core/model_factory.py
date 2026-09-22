# -*- coding: utf-8 -*-
"""模型工厂：本地 Ollama 与 OpenAI 兼容 API 双来源统一接入（框架无关层）

职责：
  1. 供应商预设表（聊天 / 嵌入分开维护；DeepSeek 不提供嵌入 API，故不在嵌入预设中）
  2. build_chat_model / build_embeddings：按配置创建 LangChain 模型实例
  3. 配置持久化分级：
     - config.yaml：非敏感配置（来源 / 厂商 / base_url / 模型名）
     - API Key（BYOK 三级）：会话临时输入 > user_settings.local.json（个人，不入库）
       > .env（全局兜底）；详见 resolve_api_key
  4. 嵌入向量空间检查：嵌入模型决定向量空间，更换为不同空间的模型后必须重建
     知识库，否则新旧向量混检会导致检索结果错乱（kb_compatibility / update_kb_space）

被 Streamlit 主程序与 api.py（FastAPI）共同使用：同一份 config.yaml + .env，
保证两个入口的模型行为一致。
"""
import os
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 容器部署时通过 MODEL_CONFIG_PATH 指向挂载卷内路径，保证模型切换配置持久化
CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_PATH") or (PROJECT_ROOT / "config.yaml"))
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------- 本地模型（与原主程序配置一致） ----------
LOCAL_CHAT_MODEL = "qwen2.5:1.5b"
LOCAL_EMBEDDING_MODEL = "bge-m3"
# 注意：用 127.0.0.1 而非 localhost——Windows 下 localhost 先解析 IPv6 失败再回退 IPv4，
# 每个请求固定多耗约 2 秒（模型设置页探测模型列表要发 4+ 个请求，差距明显）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_TIMEOUT = 120  # 秒

# =========================================================
# 供应商预设（OpenAI 兼容协议）
#   api_key_env：该厂商的 Key 在 .env / 环境变量中的变量名
#   models：常用模型列表（界面下拉），空列表 = 纯手输
# =========================================================
CHAT_PRESETS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
    },
    "阿里云百炼 DashScope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct", "qwen2.5-7b-instruct"],
        "default_model": "qwen-plus",
    },
    "SiliconFlow 硅基流动": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "models": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"],
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "自定义": {
        "base_url": "",
        "api_key_env": "OPENAI_COMPAT_API_KEY",
        "models": [],
        "default_model": "",
    },
}

EMBEDDING_PRESETS = {
    "阿里云百炼 DashScope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": ["text-embedding-v4", "text-embedding-v3"],
        "default_model": "text-embedding-v4",
    },
    "SiliconFlow 硅基流动": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "models": ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5"],
        "default_model": "BAAI/bge-m3",
    },
    "自定义": {
        "base_url": "",
        "api_key_env": "OPENAI_COMPAT_API_KEY",
        "models": [],
        "default_model": "",
    },
}

# 注意：本地 bge-m3 与 SiliconFlow 的 BAAI/bge-m3 是同一模型、同一向量空间，
# 二者切换属于"兼容切换"，无需重建知识库（见 _canonical_embed_space）。


# =========================================================
# 配置持久化（config.yaml）
# =========================================================
def _default_config() -> dict:
    return {
        "version": 0,  # 每次保存 +1，作为运行时缓存失效依据
        "chat": {"source": "local", "provider": "DeepSeek", "base_url": "", "model": ""},
        "embedding": {"source": "local", "provider": "SiliconFlow 硅基流动", "base_url": "", "model": ""},
        # 模型库：只保存公开配置（名称/来源/地址/模型），API Key 始终只走 .env
        "model_profiles": {"chat": [], "embedding": []},
        # 知识库当前向量空间（由建库时的嵌入模型决定），默认 = 本地 bge-m3
        "kb_embedding_space": LOCAL_EMBEDDING_MODEL,
    }


def load_model_config() -> dict:
    """读取模型配置；文件不存在或字段缺失时回退默认值（本地 Ollama）"""
    cfg = _default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            stored = yaml.safe_load(f) or {}
    except FileNotFoundError:
        stored = {}
    except Exception as e:  # 文件损坏时不阻断启动，回退默认并告警
        import logging
        logging.getLogger("rag_app").error(f"config.yaml 读取失败，回退默认配置: {e}")
        stored = {}

    for section in ("chat", "embedding"):
        if isinstance(stored.get(section), dict):
            cfg[section].update({
                k: v for k, v in stored[section].items() if v is not None
            })
    if stored.get("kb_embedding_space"):
        # 归一化历史空间标识（去掉本地模型的 :latest 标签），避免同模型误报空间不一致
        kb_space = str(stored["kb_embedding_space"])
        cfg["kb_embedding_space"] = (
            kb_space if kb_space.startswith("api:") else _norm_local_model(kb_space)
        )
    if isinstance(stored.get("model_profiles"), dict):
        for kind in ("chat", "embedding"):
            profiles = stored["model_profiles"].get(kind)
            if isinstance(profiles, list):
                cfg["model_profiles"][kind] = [p for p in profiles if isinstance(p, dict)]
    try:
        cfg["version"] = int(stored.get("version", 0))
    except (TypeError, ValueError):
        cfg["version"] = 0

    # 校验来源取值
    for section, presets in (("chat", CHAT_PRESETS), ("embedding", EMBEDDING_PRESETS)):
        if cfg[section].get("source") not in ("local", "api"):
            cfg[section]["source"] = "local"
        if cfg[section].get("provider") not in presets:
            cfg[section]["provider"] = "自定义"
    return cfg


def _profile_id(kind: str, profile: dict) -> str:
    """基于非敏感配置生成稳定 ID；不把 API Key 写入模型库。"""
    payload = {
        "kind": kind,
        "source": profile.get("source", "local"),
        "provider": profile.get("provider", ""),
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def upsert_model_profile(cfg: dict, kind: str, name: str, profile: dict) -> dict:
    """把聊天/嵌入模型加入模型库；同来源同模型重复加入时更新显示名称。"""
    if kind not in ("chat", "embedding"):
        raise ValueError("kind 必须是 chat 或 embedding")
    clean = {
        "name": (name or "").strip() or (profile.get("model") or "未命名模型"),
        "source": profile.get("source", "local"),
        "provider": profile.get("provider", ""),
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
    }
    clean["id"] = _profile_id(kind, clean)
    profiles = cfg.setdefault("model_profiles", {}).setdefault(kind, [])
    for i, existing in enumerate(profiles):
        if existing.get("id") == clean["id"]:
            profiles[i] = clean
            return clean
    profiles.append(clean)
    return clean


def delete_model_profile(cfg: dict, kind: str, profile_id: str) -> None:
    """从模型库移除指定模型，不影响当前线上生效模型。"""
    profiles = cfg.setdefault("model_profiles", {}).setdefault(kind, [])
    cfg["model_profiles"][kind] = [p for p in profiles if p.get("id") != profile_id]


def list_local_ollama_models() -> list[str]:
    """读取当前 Ollama 已安装的模型，用于模型设置页本地模型点选。"""
    try:
        url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        with urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        return sorted(
            [str(m.get("name", "")).strip() for m in data.get("models", []) if m.get("name")]
        )
    except Exception:
        return []


# 按模型名兜底识别嵌入/重排模型（/api/show 不支持 capabilities 时的降级判断）
_EMBED_NAME_KEYWORDS = ("embed", "bge", "gte", "rerank", "minilm", "nomic")


def list_local_chat_models() -> list[str]:
    """列出 Ollama 已安装的聊天模型（排除嵌入/重排类模型，避免混入聊天下拉框）。

    优先按 /api/show 返回的 capabilities 判断（含 embedding 能力即排除）；
    探测失败时降级按模型名关键词过滤。
    """
    chat_models = []
    for name in list_local_ollama_models():
        is_embed = any(k in name.lower() for k in _EMBED_NAME_KEYWORDS)
        try:
            req = Request(
                OLLAMA_BASE_URL.rstrip("/") + "/api/show",
                data=json.dumps({"model": name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as response:
                caps = json.loads(response.read().decode("utf-8")).get("capabilities") or []
            if caps:
                is_embed = "embedding" in caps
        except Exception:
            pass  # 探测失败时保留关键词判断结果
        if not is_embed:
            chat_models.append(name)
    return chat_models


def list_local_embedding_models() -> list[str]:
    """列出 Ollama 已安装的嵌入模型（只保留有 embedding 能力的，避免聊天模型混入）。

    优先按 /api/show 返回的 capabilities 判断；探测失败时降级按模型名关键词过滤。
    """
    embed_models = []
    for name in list_local_ollama_models():
        is_embed = any(k in name.lower() for k in _EMBED_NAME_KEYWORDS)
        try:
            req = Request(
                OLLAMA_BASE_URL.rstrip("/") + "/api/show",
                data=json.dumps({"model": name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as response:
                caps = json.loads(response.read().decode("utf-8")).get("capabilities") or []
            if caps:
                is_embed = "embedding" in caps
        except Exception:
            pass  # 探测失败时保留关键词判断结果
        if is_embed:
            embed_models.append(name)
    return embed_models


def _write_config(cfg: dict, bump_version: bool) -> None:
    """原子写配置：先写临时文件再 os.replace，断电/并发保存不会损坏或互相覆盖。"""
    if bump_version:
        cfg["version"] = int(cfg.get("version", 0)) + 1
    tmp_path = CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, CONFIG_PATH)  # 同一文件系统内原子替换


def save_model_config(cfg: dict) -> int:
    """保存完整模型配置，版本号 +1（触发运行时缓存重建）。返回新版本号"""
    _write_config(cfg, bump_version=True)
    return int(cfg["version"])


def update_kb_space(space: str) -> None:
    """知识库重建完成后更新其向量空间记录（不 bump 版本，模型对象无需重建）"""
    cfg = load_model_config()
    cfg["kb_embedding_space"] = space
    _write_config(cfg, bump_version=False)


# =========================================================
# API Key 两级解析（BYOK：自带密钥）
#   优先级：本次会话临时输入 > .env/环境变量（全局兜底）
#   - 会话级：set_session_api_key，仅当前进程、不落盘（多用户部署下防止
#     互相用对方 Key 计费；长期使用请配置到 .env）
# =========================================================
_session_key_overrides: dict[str, str] = {}  # provider -> key（仅本次运行）


def api_key_env_name(provider: str) -> str:
    return CHAT_PRESETS.get(provider, EMBEDDING_PRESETS.get(provider, {})).get(
        "api_key_env", "OPENAI_COMPAT_API_KEY"
    )


def resolve_api_key(provider: str) -> str:
    """按优先级解析该厂商的 Key：会话临时 > .env/环境变量。
    Key 一律不落盘（多用户部署下防止互相用对方 Key 计费）；长期使用请配置到 .env。"""
    return (
        _session_key_overrides.get(provider)
        or (os.environ.get(api_key_env_name(provider)) or "").strip()
    )


def api_key_source(provider: str) -> str:
    """返回当前生效 Key 的来源：session（会话临时）/ env（.env 全局）/ 空"""
    if _session_key_overrides.get(provider):
        return "session"
    if (os.environ.get(api_key_env_name(provider)) or "").strip():
        return "env"
    return ""


def set_session_api_key(provider: str, key: str) -> None:
    """页面临时输入的 Key 放入进程内覆盖表（仅本次运行有效，不写盘、不污染环境变量）"""
    key = (key or "").strip()
    if key:
        _session_key_overrides[provider] = key


# =========================================================
# 模型构建
# =========================================================
def build_chat_model(cfg: dict, api_key: str | None = None):
    """按配置构建聊天模型：本地 ChatOllama / API ChatOpenAI（OpenAI 兼容）。

    api_key：调用方显式传入的 API Key（多用户场景下各自填自己的 key）。
    传入时优先使用；未传入则回退到进程环境变量（.env / 页面临时输入）。
    """
    chat = cfg.get("chat", {})
    if chat.get("source") == "api":
        provider = chat.get("provider", "自定义")
        preset = CHAT_PRESETS[provider]
        base_url = (chat.get("base_url") or "").strip() or preset["base_url"]
        model = (chat.get("model") or "").strip() or preset["default_model"]
        if not base_url:
            raise RuntimeError("Base URL 不能为空，请填写 API 服务地址")
        # 优先用调用方传入的 key（多用户各自填自己的 key），否则读环境变量
        key = (api_key or "").strip() or resolve_api_key(provider)
        if not key:
            raise RuntimeError(
                f"未配置 API Key：请填写你的 API Key，或在 .env 中设置 {preset['api_key_env']}"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=key,
            temperature=0,
            request_timeout=MODEL_TIMEOUT,
            max_retries=2,
        )
    # 本地 Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=(chat.get("model") or LOCAL_CHAT_MODEL).strip(),
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={"timeout": MODEL_TIMEOUT},
    )


def build_embeddings(cfg: dict):
    """按配置构建嵌入模型：本地 OllamaEmbeddings / API OpenAIEmbeddings"""
    emb = cfg.get("embedding", {})
    if emb.get("source") == "api":
        provider = emb.get("provider", "自定义")
        preset = EMBEDDING_PRESETS[provider]
        base_url = (emb.get("base_url") or "").strip() or preset["base_url"]
        model = (emb.get("model") or "").strip() or preset["default_model"]
        if not base_url:
            raise RuntimeError("Base URL 不能为空，请填写 API 服务地址")
        key = resolve_api_key(provider)
        if not key:
            raise RuntimeError(
                f"未配置 API Key：请在 .env 中设置 {preset['api_key_env']}，"
                "或在页面「模型设置」中临时输入（不落盘）"
            )
        from langchain_openai import OpenAIEmbeddings

        # chunk_size=16：国内兼容端点单请求批量上限普遍较小，取保守值
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=key,
            request_timeout=MODEL_TIMEOUT,
            max_retries=2,
            chunk_size=16,
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=(emb.get("model") or LOCAL_EMBEDDING_MODEL).strip(),
        base_url=OLLAMA_BASE_URL,
        client_kwargs={"timeout": MODEL_TIMEOUT},
    )


def build_runtime_models(cfg: dict, chat_api_key: str | None = None) -> tuple:
    """构建聊天/嵌入模型，API 配置无效时自动回退本地模型（保证系统可用）。
    chat_api_key：聊天模型的 API Key（多用户各自填自己的 key，传入时优先）。
    返回 (chat_model, embeddings, effective_cfg, warning)"""
    warning = ""
    effective = {
        "chat": dict(cfg.get("chat", {})),
        "embedding": dict(cfg.get("embedding", {})),
    }
    try:
        chat_model = build_chat_model(cfg, api_key=chat_api_key)
    except Exception as e:
        warning = f"聊天模型 API 配置无效，已回退本地 {LOCAL_CHAT_MODEL}：{e}"
        # 关键：清空模型名，否则本地分支会拿云端模型名（如 deepseek-chat）请求 Ollama，必然失败
        effective["chat"]["source"] = "local"
        effective["chat"]["model"] = ""
        chat_model = build_chat_model(effective)
    try:
        embeddings = build_embeddings(cfg)
    except Exception as e:
        warning = (warning + "\n" if warning else "") + (
            f"嵌入模型 API 配置无效，已回退本地 {LOCAL_EMBEDDING_MODEL}：{e}"
        )
        effective["embedding"]["source"] = "local"
        effective["embedding"]["model"] = ""
        embeddings = build_embeddings(effective)
    return chat_model, embeddings, effective, warning


# =========================================================
# 嵌入向量空间检查
# =========================================================
def _norm_local_model(name: str) -> str:
    """本地 Ollama 模型名去掉 :latest 标签（bge-m3:latest 与 bge-m3 是同一模型）"""
    name = (name or "").strip()
    return name[: -len(":latest")] if name.endswith(":latest") else name


def _canonical_embed_space(source: str, provider: str, model: str) -> str:
    """计算嵌入模型的向量空间标识（同空间 = 语义空间一致，可直接混用）"""
    if source == "local":
        return _norm_local_model(model) or LOCAL_EMBEDDING_MODEL
    if provider.startswith("SiliconFlow") and model == "BAAI/bge-m3":
        return LOCAL_EMBEDDING_MODEL  # 与本地 bge-m3 同模型同空间
    return f"api:{model}"


def embedding_space(cfg: dict) -> str:
    emb = cfg.get("embedding", {})
    return _canonical_embed_space(
        emb.get("source", "local"), emb.get("provider", ""), emb.get("model", "")
    )


def kb_compatibility(cfg: dict | None = None) -> tuple[bool, str, str]:
    """检查当前生效嵌入模型与知识库建库空间是否一致。
    cfg 建议传入运行时实际生效配置（API 回退本地后与 config.yaml 不同）；
    知识库空间始终实时读取 config.yaml（重建完成后无需重启即解除阻断）。
    返回 (ok, 当前空间, 知识库空间)"""
    current = embedding_space(cfg or load_model_config())
    kb_space = load_model_config().get("kb_embedding_space", LOCAL_EMBEDDING_MODEL)
    return current == kb_space, current, kb_space


# =========================================================
# 连接测试（独立实例，不影响运行时缓存）
# =========================================================
def _friendly_api_error(e: Exception) -> str:
    s = str(e)
    low = s.lower()
    if "401" in s or "unauthorized" in low or ("invalid" in low and "key" in low):
        return "认证失败（401）：API Key 无效或未填写"
    if "404" in s or ("model" in low and "not" in low and "found" in low):
        return "模型不存在（404）：请检查模型名是否正确"
    if "timeout" in low or "timed out" in low:
        return "请求超时：请检查 Base URL 是否可达、网络是否正常"
    if "connection" in low or "connect" in low:
        return "连接失败：无法访问 Base URL，请检查地址与网络"
    return s[:200]


def test_chat_connection(cfg: dict) -> tuple[bool, str]:
    """对聊天模型发最小请求验证连通性，返回 (是否成功, 说明)"""
    try:
        reply = build_chat_model(cfg).invoke("请只回复 OK")
        content = reply.content if hasattr(reply, "content") else str(reply)
        return True, f"连通正常，模型回复：{str(content)[:40]}"
    except Exception as e:
        return False, _friendly_api_error(e)


def test_embedding_connection(cfg: dict) -> tuple[bool, str]:
    """对嵌入模型发最小请求验证连通性，返回维度信息便于核对兼容性"""
    try:
        vec = build_embeddings(cfg).embed_query("连接测试")
        return True, f"连通正常，向量维度：{len(vec)}"
    except Exception as e:
        return False, _friendly_api_error(e)


# =========================================================
# 展示辅助
# =========================================================
def describe_source(section: dict, presets: dict, local_label: str) -> str:
    """人类可读的模型来源描述，用于侧边栏展示"""
    if section.get("source") == "api":
        return f"API · {section.get('provider', '')} / {section.get('model', '')}"
    return f"本地 · {section.get('model') or local_label}"
