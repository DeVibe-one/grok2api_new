"""Admin web UI + APIs - 完整管理功能"""

from __future__ import annotations

from dataclasses import asdict
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Header, status, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.config import settings, runtime_config
from app.services.token_manager import token_manager
from app.services.request_stats import request_stats
from app.services.request_logger import request_logger
from app.services.api_keys import api_key_manager
from app.services.image_cache import image_cache

router = APIRouter()

# 常量
TEMPLATE_DIR = Path(__file__).parent.parent / "template"
SESSION_EXPIRE_HOURS = 24

# 会话存储
_sessions: Dict[str, datetime] = {}


# === 请求/响应模型 ===

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str


class TokenCreateRequest(BaseModel):
    token: str = Field(..., min_length=1)
    name: str = ""
    enabled: bool = True


class TokenBatchCreateRequest(BaseModel):
    tokens: List[str] = Field(..., min_length=1)
    name: str = ""
    enabled: bool = True


class TokenUpdateRequest(BaseModel):
    token: str = Field(..., min_length=1)
    name: Optional[str] = None
    enabled: Optional[bool] = None


class TokenDeleteRequest(BaseModel):
    token: str = Field(..., min_length=1)


class TokenTestRequest(BaseModel):
    token: str = Field(..., min_length=1)


class TokenClearCooldownRequest(BaseModel):
    token: str = Field(..., min_length=1)


class ConversationDeleteRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)


class ApiKeyCreateRequest(BaseModel):
    name: str = ""
    count: int = 1


class ApiKeyDeleteRequest(BaseModel):
    keys: List[str]


class ApiKeyUpdateRequest(BaseModel):
    key: str
    name: Optional[str] = None
    enabled: Optional[bool] = None


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


# === 鉴权 ===

def verify_admin_session(authorization: Optional[str] = Header(None)) -> bool:
    """验证管理员会话（Bearer Token）"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "未授权访问", "code": "UNAUTHORIZED"}
        )

    token = authorization[7:]

    if token not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "会话无效", "code": "SESSION_INVALID"}
        )

    if datetime.now() > _sessions[token]:
        del _sessions[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "会话已过期", "code": "SESSION_EXPIRED"}
        )

    return True


# === 页面路由 ===

@router.get("/admin/login", response_class=HTMLResponse)
async def login_page():
    """登录页面"""
    login_html = TEMPLATE_DIR / "login.html"
    if login_html.exists():
        return HTMLResponse(login_html.read_text(encoding="utf-8"))
    return HTMLResponse(_get_fallback_login_html())


@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理页面"""
    admin_html = TEMPLATE_DIR / "admin.html"
    if admin_html.exists():
        return HTMLResponse(admin_html.read_text(encoding="utf-8"))
    return HTMLResponse(_get_fallback_admin_html())


# === 认证 API ===

@router.post("/admin/api/login", response_model=LoginResponse)
async def admin_login(request: LoginRequest) -> LoginResponse:
    """管理员登录"""
    expected_user = settings.admin_username
    expected_pass = settings.admin_password

    if request.username != expected_user or request.password != expected_pass:
        return LoginResponse(success=False, message="用户名或密码错误")

    session_token = secrets.token_urlsafe(32)
    _sessions[session_token] = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)

    return LoginResponse(success=True, token=session_token, message="登录成功")


@router.post("/admin/api/logout")
async def admin_logout(
    _: bool = Depends(verify_admin_session),
    authorization: Optional[str] = Header(None)
):
    """管理员登出"""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        if token in _sessions:
            del _sessions[token]
            return {"success": True, "message": "登出成功"}
    return {"success": False, "message": "无效的会话"}


# === Token 管理 API ===

@router.get("/admin/api/tokens")
async def admin_list_tokens(_: bool = Depends(verify_admin_session)):
    """获取 Token 列表"""
    now = time.time()
    tokens = []
    for t in token_manager.list_tokens():
        token_dict = asdict(t)
        # 添加冷却状态
        token_dict["in_cooldown"] = t.cooldown_until > now
        token_dict["cooldown_remaining"] = max(0, int(t.cooldown_until - now))
        tokens.append(token_dict)
    return {"tokens": tokens, "stats": token_manager.get_stats()}


@router.post("/admin/api/tokens")
async def admin_create_token(
    payload: TokenCreateRequest,
    _: bool = Depends(verify_admin_session)
):
    """添加 Token"""
    token = token_manager._normalize_token(payload.token)
    if not token:
        raise HTTPException(status_code=400, detail="token is required")

    if token in token_manager.tokens:
        raise HTTPException(status_code=409, detail="token already exists")

    await token_manager.add_token(token, payload.name)
    if payload.enabled is False:
        await token_manager.update_token(token, enabled=False)

    return {"ok": True}


@router.post("/admin/api/tokens/batch")
async def admin_batch_create_tokens(
    payload: TokenBatchCreateRequest,
    _: bool = Depends(verify_admin_session)
):
    """批量添加 Token（自动去重）"""
    result = await token_manager.add_tokens_batch(
        payload.tokens, payload.name, payload.enabled
    )
    return {"ok": True, **result}


@router.patch("/admin/api/tokens")
async def admin_update_token(
    payload: TokenUpdateRequest,
    _: bool = Depends(verify_admin_session)
):
    """更新 Token"""
    ok = await token_manager.update_token(
        payload.token, name=payload.name, enabled=payload.enabled
    )
    if not ok:
        raise HTTPException(status_code=404, detail="token not found")
    return {"ok": True}


@router.delete("/admin/api/tokens")
async def admin_delete_token(
    payload: TokenDeleteRequest,
    _: bool = Depends(verify_admin_session)
):
    """删除 Token"""
    ok = await token_manager.delete_token(payload.token)
    if not ok:
        raise HTTPException(status_code=404, detail="token not found")
    return {"ok": True}


@router.post("/admin/api/tokens/test")
async def admin_test_token(
    payload: TokenTestRequest,
    _: bool = Depends(verify_admin_session)
):
    """测试 Token 可用性"""
    result = await token_manager.test_token(payload.token)
    return result


@router.post("/admin/api/tokens/clear-cooldown")
async def admin_clear_token_cooldown(
    payload: TokenClearCooldownRequest,
    _: bool = Depends(verify_admin_session)
):
    """清除 Token 冷却"""
    await token_manager.clear_cooldown(payload.token)
    return {"ok": True}


@router.post("/admin/api/tokens/refresh-all")
async def admin_refresh_all_tokens(
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_session)
):
    """刷新所有 Token（后台任务）"""
    progress = token_manager.get_refresh_progress()
    if progress["in_progress"]:
        return {"success": False, "error": "刷新任务正在进行中", "progress": progress}

    # 在后台执行刷新
    background_tasks.add_task(token_manager.refresh_all_tokens)
    return {"success": True, "message": "刷新任务已启动"}


@router.get("/admin/api/tokens/refresh-progress")
async def admin_get_refresh_progress(_: bool = Depends(verify_admin_session)):
    """获取刷新进度"""
    return token_manager.get_refresh_progress()


# === 会话管理 API ===

@router.get("/admin/api/conversations")
async def admin_list_conversations(_: bool = Depends(verify_admin_session)):
    """获取会话列表"""
    try:
        from app.services.conversation_manager import conversation_manager

        conversations = []
        now = time.time()

        for conv_id, context in conversation_manager.conversations.items():
            # 计算剩余存活时间
            age = now - context.updated_at
            ttl_remaining = max(0, int(settings.conversation_ttl - age))

            conversations.append({
                "conversation_id": conv_id,
                "grok_conversation_id": context.conversation_id,
                "token": context.token[:12] + "..." if len(context.token) > 12 else context.token,
                "message_count": context.message_count,
                "created_at": context.created_at,
                "last_active": context.updated_at,
                "ttl_remaining": ttl_remaining
            })

        # 使用 conversation_manager 的统计
        manager_stats = conversation_manager.get_stats()
        stats = {
            "total_conversations": len(conversations),
            "total_messages": sum(c["message_count"] for c in conversations),
            "tokens_with_conversations": manager_stats["tokens_with_conversations"],
            "ttl_seconds": settings.conversation_ttl,
            "last_cleanup_time": manager_stats["last_cleanup_time"],
            "total_cleaned": manager_stats["total_cleaned"],
            "auto_cleanup_enabled": manager_stats["auto_cleanup_enabled"]
        }

        conversations.sort(key=lambda x: x.get("last_active", 0), reverse=True)

        return {"conversations": conversations, "stats": stats}
    except Exception as e:
        return {"conversations": [], "stats": {}, "error": str(e)}


@router.delete("/admin/api/conversations")
async def admin_delete_conversation(
    payload: ConversationDeleteRequest,
    _: bool = Depends(verify_admin_session)
):
    """删除单个会话"""
    try:
        from app.services.conversation_manager import conversation_manager

        conv_id = payload.conversation_id
        if conv_id not in conversation_manager.conversations:
            raise HTTPException(status_code=404, detail="conversation not found")

        await conversation_manager.delete_conversation(conv_id)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/api/conversations/clear")
async def admin_clear_conversations(_: bool = Depends(verify_admin_session)):
    """清空所有会话"""
    try:
        from app.services.conversation_manager import conversation_manager

        conversation_manager.conversations.clear()
        conversation_manager.token_conversations.clear()
        await conversation_manager._save_async()

        return {"ok": True, "message": "所有会话已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 请求统计 API ===

@router.get("/admin/api/stats")
async def admin_get_stats(_: bool = Depends(verify_admin_session)):
    """获取统计摘要"""
    return {
        "summary": request_stats.get_summary(),
        "token_stats": token_manager.get_stats(),
        "key_stats": api_key_manager.get_stats()
    }


@router.get("/admin/api/stats/hourly")
async def admin_get_hourly_stats(
    hours: int = 24,
    _: bool = Depends(verify_admin_session)
):
    """获取小时统计"""
    return {"data": request_stats.get_hourly_stats(hours)}


@router.get("/admin/api/stats/daily")
async def admin_get_daily_stats(
    days: int = 7,
    _: bool = Depends(verify_admin_session)
):
    """获取日统计"""
    return {"data": request_stats.get_daily_stats(days)}


# === 日志审计 API ===

@router.get("/admin/api/logs")
async def admin_get_logs(
    limit: int = 100,
    offset: int = 0,
    _: bool = Depends(verify_admin_session)
):
    """获取请求日志"""
    return {
        "logs": request_logger.get_logs(limit, offset),
        "total": request_logger.get_total()
    }


@router.post("/admin/api/logs/clear")
async def admin_clear_logs(_: bool = Depends(verify_admin_session)):
    """清空日志"""
    await request_logger.clear()
    return {"ok": True, "message": "日志已清空"}


# === API Key 管理 API ===

@router.get("/admin/api/keys")
async def admin_list_keys(_: bool = Depends(verify_admin_session)):
    """获取 API Key 列表"""
    keys = [asdict(k) for k in api_key_manager.list_keys()]
    return {"keys": keys, "stats": api_key_manager.get_stats()}


@router.post("/admin/api/keys")
async def admin_create_key(
    payload: ApiKeyCreateRequest,
    _: bool = Depends(verify_admin_session)
):
    """创建 API Key"""
    if payload.count > 1:
        keys = await api_key_manager.create_keys_batch(payload.count, payload.name)
        return {"ok": True, "keys": [asdict(k) for k in keys]}
    else:
        key = await api_key_manager.create_key(payload.name)
        return {"ok": True, "key": asdict(key)}


@router.patch("/admin/api/keys")
async def admin_update_key(
    payload: ApiKeyUpdateRequest,
    _: bool = Depends(verify_admin_session)
):
    """更新 API Key"""
    ok = await api_key_manager.update_key(
        payload.key, name=payload.name, enabled=payload.enabled
    )
    if not ok:
        raise HTTPException(status_code=404, detail="key not found")
    return {"ok": True}


@router.delete("/admin/api/keys")
async def admin_delete_keys(
    payload: ApiKeyDeleteRequest,
    _: bool = Depends(verify_admin_session)
):
    """删除 API Key"""
    deleted = await api_key_manager.delete_keys_batch(payload.keys)
    return {"ok": True, "deleted": deleted}


# === 系统配置 API ===

@router.get("/admin/api/config")
async def admin_get_config(_: bool = Depends(verify_admin_session)):
    """获取系统配置"""
    return {
        "schema": runtime_config.get_schema(),
        "groups": runtime_config.get_groups(),
        "values": runtime_config.get_all()
    }


@router.post("/admin/api/config")
async def admin_update_config(
    payload: ConfigUpdateRequest,
    _: bool = Depends(verify_admin_session)
):
    """更新系统配置"""
    results = await runtime_config.set_batch(payload.config)
    return {"ok": True, "results": results}


@router.post("/admin/api/config/reset")
async def admin_reset_config(
    key: str,
    _: bool = Depends(verify_admin_session)
):
    """重置配置为默认值"""
    ok = await runtime_config.reset(key)
    return {"ok": ok}


# === 图片缓存 API ===

@router.get("/admin/api/images")
async def admin_list_images(_: bool = Depends(verify_admin_session)):
    """获取缓存图片列表"""
    images = image_cache.list_cached_images()
    stats = image_cache.get_cache_stats()
    return {
        "ok": True,
        "images": images,
        "stats": stats
    }


@router.delete("/admin/api/images")
async def admin_delete_image(
    filename: str,
    _: bool = Depends(verify_admin_session)
):
    """删除指定缓存图片"""
    success = await image_cache.delete_cached_image(filename)
    return {"ok": success}


@router.post("/admin/api/images/clear")
async def admin_clear_images(_: bool = Depends(verify_admin_session)):
    """清空所有缓存图片"""
    count = await image_cache.clear_all_cache()
    return {"ok": True, "deleted": count}


# === 后备 HTML ===

def _get_fallback_login_html() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>登录 - Grok2API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #f5f5f5; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); width: 100%; max-width: 360px; }
        h1 { font-size: 1.5rem; text-align: center; margin-bottom: 1.5rem; }
        .field { margin-bottom: 1rem; }
        label { display: block; font-size: 0.875rem; margin-bottom: 0.5rem; }
        input { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; }
        button { width: 100%; padding: 0.75rem; background: #111; color: white; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; }
        button:hover { background: #333; }
        .error { color: #e53e3e; font-size: 0.875rem; margin-top: 1rem; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Grok2API 登录</h1>
        <form id="form">
            <div class="field"><label>账户</label><input type="text" id="username" required></div>
            <div class="field"><label>密码</label><input type="password" id="password" required></div>
            <button type="submit">登录</button>
            <div id="error" class="error"></div>
        </form>
    </div>
    <script>
        document.getElementById('form').onsubmit = async (e) => {
            e.preventDefault();
            const r = await fetch('/admin/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: document.getElementById('username').value, password: document.getElementById('password').value })
            });
            const d = await r.json();
            if (d.success) { localStorage.setItem('adminToken', d.token); location.href = '/admin'; }
            else { document.getElementById('error').textContent = d.message; }
        };
    </script>
</body>
</html>"""


def _get_fallback_admin_html() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>管理后台 - Grok2API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #f5f5f5; min-height: 100vh; }
        .header { background: white; border-bottom: 1px solid #eee; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.25rem; }
        .btn { padding: 0.5rem 1rem; background: #111; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #333; }
        .btn-outline { background: white; color: #111; border: 1px solid #ddd; }
        .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; padding: 1rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Grok2API 管理后台</h1>
        <button class="btn btn-outline" onclick="logout()">退出</button>
    </div>
    <div class="container">
        <div class="card">
            <p>请刷新页面或检查模板文件。</p>
        </div>
    </div>
    <script>
        const token = localStorage.getItem('adminToken');
        if (!token) location.href = '/admin/login';
        function logout() { localStorage.removeItem('adminToken'); location.href = '/admin/login'; }
    </script>
</body>
</html>"""
