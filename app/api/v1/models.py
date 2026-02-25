"""模型列表 API - 含模型到 Grok 内部参数的映射"""

import time
from typing import Tuple
from fastapi import APIRouter, HTTPException

from app.models.openai_models import ModelList, Model

router = APIRouter()


# ── 模型映射表 ──────────────────────────────────────────────
# model_id → (grok_model, model_mode, display_name, description)

MODEL_REGISTRY = {
    "grok-3": (
        "grok-3",
        "MODEL_MODE_GROK_3",
        "Grok 3",
        "Standard Grok 3 model",
    ),
    "grok-3-mini": (
        "grok-3",
        "MODEL_MODE_GROK_3_MINI_THINKING",
        "Grok 3 Mini",
        "Grok 3 with mini thinking",
    ),
    "grok-4.1-thinking": (
        "grok-4-1-thinking-1129",
        "MODEL_MODE_GROK_4_1_THINKING",
        "Grok",
        "Grok",
    ),
    "grok-4.2-fast": (
        "grok-420",
        "MODEL_MODE_FAST",
        "Grok",
        "Grok",
    ),
    "grok-4.2": (
        "grok-420",
        "MODEL_MODE_GROK_420",
        "Grok 4.2",
        "Standard Grok 4.2 model",
    ),
    "grok-expert": (
        "grok-420",
        "MODEL_MODE_EXPERT",
        "Grok 4.2 Thinking",
        "Grok 4.2 Thinking",
    ),
}

# ── 别名：兼容直接传内部模型名 ──
MODEL_ALIASES = {
    "grok-420": "grok-4.2",
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

    return Model(id=model_id, created=int(time.time()), owned_by="xai")
