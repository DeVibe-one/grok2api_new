"""
Grok2API (2api)

- 兼容 OpenAI 的 `/v1/chat/completions` 接口
- 通过 Grok `conversationId` + `parentResponseId` 实现真实多轮对话上下文
- Token (SSO/Cookie) 轮询管理
- 简洁的管理后台 `/admin`
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.api.v1.models import router as models_router
from app.api.v1.images import router as images_router
from app.core.config import settings, runtime_config
from app.core.logger import logger
from app.core.storage import storage_manager
from app.services.conversation_manager import conversation_manager
from app.services.token_manager import token_manager
from app.services.request_stats import request_stats
from app.services.request_logger import request_logger
from app.services.api_keys import api_key_manager


try:
    if sys.platform != "win32":
        import uvloop

        uvloop.install()
        logger.info("[Grok2API] uvloop 已启用")
    else:
        logger.info("[Grok2API] Windows: 使用默认 asyncio 事件循环")
except Exception:
    logger.info("[Grok2API] uvloop 未安装，使用默认 asyncio 事件循环")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Grok2API] 正在启动...")

    await storage_manager.init()
    logger.info("[Grok2API] 存储管理器已初始化")

    await runtime_config.init()
    logger.info("[Grok2API] 运行时配置已初始化")

    await token_manager.init()
    logger.info("[Grok2API] Token管理器已初始化")

    await conversation_manager.init()
    logger.info("[Grok2API] 会话管理器已初始化")

    await request_stats.init()
    logger.info("[Grok2API] 请求统计已初始化")

    await request_logger.init()
    logger.info("[Grok2API] 请求日志已初始化")

    await api_key_manager.init()
    logger.info("[Grok2API] API Key管理器已初始化")

    logger.info("[Grok2API] 启动完成")
    yield

    logger.info("[Grok2API] 正在关闭...")
    await request_stats.save()
    await request_logger.save()
    await conversation_manager.shutdown()
    await token_manager.shutdown()
    await storage_manager.close()
    logger.info("[Grok2API] 关闭完成")


app = FastAPI(
    title=settings.app_name,
    description="Grok 转 OpenAI 兼容 API 代理（支持真实对话上下文）",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(images_router)  # /images/xxx
app.include_router(admin_router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "admin": "/admin",
        "login": "/admin/login",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

