"""模型列表 API - 含模型到 Grok 内部参数的映射"""

import time
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException

from app.models.openai_models import ModelList, Model

router = APIRouter()


# ── 模型映射表 ──────────────────────────────────────────────
# model_id → (grok_model, model_mode, display_name, description)

MODEL_REGISTRY = {
    # ── Grok 3 系列 ──
    "grok-3": (
        "grok-3", "MODEL_MODE_GROK_3",
        "Grok 3", "Standard Grok 3 model",
    ),
    "grok-3-mini": (
        "grok-3", "MODEL_MODE_GROK_3_MINI_THINKING",
        "Grok 3 Mini", "Grok 3 with mini thinking",
    ),
    "grok-3-thinking": (
        "grok-3", "MODEL_MODE_GROK_3_THINKING",
        "Grok 3 Thinking", "Grok 3 with full thinking",
    ),
    # ── Grok 4 系列 ──
    "grok-4": (
        "grok-4", "MODEL_MODE_GROK_4",
        "Grok 4", "Standard Grok 4 model",
    ),
    "grok-4-mini": (
        "grok-4-mini", "MODEL_MODE_GROK_4_MINI_THINKING",
        "Grok 4 Mini", "Fast Grok 4 Mini Thinking model",
    ),
    "grok-4-thinking": (
        "grok-4", "MODEL_MODE_GROK_4_THINKING",
        "Grok 4 Thinking", "Grok 4 with full thinking",
    ),
    "grok-4-heavy": (
        "grok-4", "MODEL_MODE_HEAVY",
        "Grok 4 Heavy", "Most powerful Grok 4 model (requires Super Token)",
    ),
    # ── Grok 4.1 系列 ──
    "grok-4.1-mini": (
        "grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_MINI_THINKING",
        "Grok 4.1 Mini", "Grok 4.1 mini thinking model",
    ),
    "grok-4.1-fast": (
        "grok-4-1-thinking-1129", "MODEL_MODE_FAST",
        "Grok 4.1 Fast", "Fast version of Grok 4.1",
    ),
    "grok-4.1-expert": (
        "grok-4-1-thinking-1129", "MODEL_MODE_EXPERT",
        "Grok 4.1 Expert", "Expert mode with enhanced reasoning",
    ),
    "grok-4.1-thinking": (
        "grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_THINKING",
        "Grok 4.1 Thinking", "Grok 4.1 with advanced thinking",
    ),
}

# ── 别名：兼容直接传内部模型名 ──
MODEL_ALIASES = {
    "grok-4-1-thinking-1129": "grok-4.1-thinking",
    "grok-4-mini-thinking-tahoe": "grok-4-mini",
}


def resolve_model(model_id: str) -> Tuple[str, str, str]:
    """将用户传入的 model_id 解析为 (grok_model, model_mode, resolved_model_id)

    支持：
    - 标准名称：grok-4.1-fast → 直接查表
    - 别名：grok-4-1-thinking-1129 → 转为 grok-4.1-thinking 再查表
    - 未知模型：原样透传，modelMode 用 MODEL_MODE_AUTO
    """
    # 标准名称
    if model_id in MODEL_REGISTRY:
        grok_model, model_mode, _, _ = MODEL_REGISTRY[model_id]
        return grok_model, model_mode, model_id

    # 别名
    alias = MODEL_ALIASES.get(model_id)
    if alias and alias in MODEL_REGISTRY:
        grok_model, model_mode, _, _ = MODEL_REGISTRY[alias]
        return grok_model, model_mode, alias

    # 未知模型：原样透传
    return model_id, "MODEL_MODE_AUTO", model_id


# ── API 路由 ──────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """列出所有可用模型"""
    created = int(time.time())
    models = [
        Model(id=model_id, created=created, owned_by="xai")
        for model_id in MODEL_REGISTRY
    ]
    return ModelList(data=models)


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """获取特定模型信息"""
    if model_id not in MODEL_REGISTRY and model_id not in MODEL_ALIASES:
        raise HTTPException(status_code=404, detail="Model not found")

    return Model(
        id=model_id,
        created=int(time.time()),
        owned_by="xai"
    )
