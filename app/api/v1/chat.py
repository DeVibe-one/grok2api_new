"""聊天 API 路由"""

import time
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.models.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ResponseRequest,
    ResponseResponse,
)
from app.services.grok_client import GrokClient
from app.services.api_keys import api_key_manager
from app.services.request_stats import request_stats
from app.services.request_logger import request_logger
from app.core.logger import logger

router = APIRouter()


def _extract_api_key(raw_request: Request) -> Optional[str]:
    """从请求头中提取 API Key"""
    auth = raw_request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def _get_client_ip(raw_request: Request) -> str:
    """获取客户端 IP"""
    forwarded = raw_request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return raw_request.client.host if raw_request.client else "unknown"


async def _verify_api_key(raw_request: Request) -> Optional[str]:
    """验证 API Key，返回 key 字符串（如果有 key 配置的话）

    逻辑：
    - 如果没有配置任何 API Key，则跳过验证（开放模式）
    - 如果配置了 API Key，则必须提供有效的 key
    """
    api_key = _extract_api_key(raw_request)

    # 如果没有配置任何 key，开放模式
    if not api_key_manager.keys:
        return api_key

    # 有配置 key，必须验证
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "缺少认证令牌，请在请求头中提供 Authorization: Bearer <API_KEY>",
                    "type": "authentication_error",
                    "code": "missing_token",
                }
            },
        )

    key_info = api_key_manager.validate_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "无效的 API Key",
                    "type": "authentication_error",
                    "code": "invalid_token",
                }
            },
        )

    # 记录使用
    await api_key_manager.record_usage(api_key)
    return api_key


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request, raw_response: Response
):
    """
    聊天补全接口 - OpenAI 兼容

    支持特性：
    - 真实的多轮对话上下文
    - 流式和非流式响应
    - 通过 conversation_id 参数继续对话
    """
    start_time = time.time()
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    api_key = None
    client_ip = _get_client_ip(raw_request)

    try:
        # 验证 API Key
        api_key = await _verify_api_key(raw_request)

        conv_id = (request.conversation_id or "").strip() or None

        logger.info(
            f"[ChatAPI] 收到请求: model={request.model}, stream={request.stream}, conv_id={conv_id}, ip={client_ip}"
        )

        # 调用 Grok 客户端
        result, openai_conv_id, _, _ = await GrokClient.chat(
            messages=[msg.model_dump() for msg in request.messages],
            model=request.model,
            stream=request.stream,
            conversation_id=conv_id,
            thinking=request.thinking,
        )

        if request.stream:
            # 流式响应
            async def stream_wrapper():
                try:
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    created = int(time.time())

                    # 发送会话 ID（第一个块）
                    first_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(role="assistant"),
                                finish_reason=None,
                            )
                        ],
                        conversation_id=openai_conv_id,
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"

                    # 流式内容
                    async for text in result:
                        chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=text),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    # 结束块
                    end_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {end_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    # 流式成功 - 记录统计和日志
                    duration_ms = int((time.time() - start_time) * 1000)
                    await request_stats.record(request.model, success=True)
                    await request_logger.log(
                        model=request.model,
                        token="",
                        api_key=api_key,
                        success=True,
                        error=None,
                        duration_ms=duration_ms,
                        ip=client_ip,
                        stream=True,
                    )

                except Exception as e:
                    logger.error(f"[ChatAPI] 流式处理错误: {e}")

                    # 流式失败 - 记录统计和日志
                    duration_ms = int((time.time() - start_time) * 1000)
                    await request_stats.record(request.model, success=False)
                    await request_logger.log(
                        model=request.model,
                        token="",
                        api_key=api_key,
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                        ip=client_ip,
                        stream=True,
                    )

                    error_chunk = ChatCompletionChunk(
                        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content=f"Error: {str(e)}"
                                ),
                                finish_reason="error",
                            )
                        ],
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

            response_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            }
            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers=response_headers,
            )
        else:
            # 非流式响应
            duration_ms = int((time.time() - start_time) * 1000)

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatCompletionResponseMessage(
                            role="assistant", content=result
                        ),
                        finish_reason="stop",
                    )
                ],
                conversation_id=openai_conv_id,
            )

            # 记录统计和日志
            await request_stats.record(request.model, success=True)
            await request_logger.log(
                model=request.model,
                token="",
                api_key=api_key,
                success=True,
                error=None,
                duration_ms=duration_ms,
                ip=client_ip,
                stream=False,
            )

            raw_response.headers["X-Request-ID"] = request_id
            logger.info(
                f"[ChatAPI] 响应成功: conv_id={openai_conv_id}, duration={duration_ms}ms"
            )
            return response

    except HTTPException as e:
        # 认证错误直接抛出，不记录为请求失败
        e.headers = {**(e.headers or {}), "X-Request-ID": request_id}
        raise e
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[ChatAPI] 请求失败: {e}")

        # 记录失败统计和日志
        await request_stats.record(request.model, success=False)
        await request_logger.log(
            model=request.model,
            token="",
            api_key=api_key,
            success=False,
            error=str(e),
            duration_ms=duration_ms,
            ip=client_ip,
            stream=request.stream,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_server_error",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id},
        )


@router.post("/responses")
async def create_response(
    request: ResponseRequest, raw_request: Request, raw_response: Response
):
    """
    继续对话接口 - 真实上下文，不拼接历史

    特点：
    - 客户端只需发送 conversation_id + 当前新消息
    - 完全依赖 Grok 的会话缓存维护上下文
    - 更高效，符合真实多轮对话设计
    """
    start_time = time.time()
    request_id = f"req-{uuid.uuid4().hex[:16]}"
    api_key = None
    client_ip = _get_client_ip(raw_request)

    try:
        # 验证 API Key
        api_key = await _verify_api_key(raw_request)

        logger.info(
            f"[ResponseAPI] 收到请求: conv_id={request.conversation_id}, stream={request.stream}, ip={client_ip}"
        )

        conv_id = (request.conversation_id or "").strip()
        if not conv_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "缺少 conversation_id，请在请求体中提供",
                        "type": "invalid_request_error",
                        "code": "missing_conversation_id",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )

        # 构造消息格式（只包含当前新消息）
        messages = [{"role": "user", "content": request.message}]

        # 调用 Grok 客户端（强制使用 conversation_id）
        result, openai_conv_id, _, _ = await GrokClient.chat(
            messages=messages,
            model=request.model,
            stream=request.stream,
            conversation_id=conv_id,
        )

        if request.stream:
            # 流式响应
            async def stream_wrapper():
                try:
                    chunk_id = f"resp-{uuid.uuid4().hex[:24]}"
                    created = int(time.time())

                    # 发送会话 ID（第一个块）
                    first_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(role="assistant"),
                                finish_reason=None,
                            )
                        ],
                        conversation_id=openai_conv_id,
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"

                    # 流式内容
                    async for text in result:
                        chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=text),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    # 结束块
                    end_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {end_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    # 记录统计
                    duration_ms = int((time.time() - start_time) * 1000)
                    await request_stats.record(request.model, success=True)
                    await request_logger.log(
                        model=request.model,
                        token="",
                        api_key=api_key,
                        success=True,
                        error=None,
                        duration_ms=duration_ms,
                        ip=client_ip,
                        stream=True,
                    )

                except Exception as e:
                    logger.error(f"[ResponseAPI] 流式处理错误: {e}")
                    duration_ms = int((time.time() - start_time) * 1000)
                    await request_stats.record(request.model, success=False)
                    await request_logger.log(
                        model=request.model,
                        token="",
                        api_key=api_key,
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                        ip=client_ip,
                        stream=True,
                    )

                    error_chunk = ChatCompletionChunk(
                        id=f"resp-{uuid.uuid4().hex[:24]}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content=f"Error: {str(e)}"
                                ),
                                finish_reason="error",
                            )
                        ],
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

            response_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            }
            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers=response_headers,
            )
        else:
            # 非流式响应
            duration_ms = int((time.time() - start_time) * 1000)

            response = ResponseResponse(
                id=f"resp-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request.model,
                conversation_id=openai_conv_id,
                message=result,
            )

            # 记录统计
            await request_stats.record(request.model, success=True)
            await request_logger.log(
                model=request.model,
                token="",
                api_key=api_key,
                success=True,
                error=None,
                duration_ms=duration_ms,
                ip=client_ip,
                stream=False,
            )

            raw_response.headers["X-Request-ID"] = request_id
            logger.info(
                f"[ResponseAPI] 响应成功: conv_id={openai_conv_id}, duration={duration_ms}ms"
            )
            return response

    except HTTPException as e:
        e.headers = {**(e.headers or {}), "X-Request-ID": request_id}
        raise e
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[ResponseAPI] 请求失败: {e}")

        await request_stats.record(request.model, success=False)
        await request_logger.log(
            model=request.model,
            token="",
            api_key=api_key,
            success=False,
            error=str(e),
            duration_ms=duration_ms,
            ip=client_ip,
            stream=request.stream,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_server_error",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id},
        )
