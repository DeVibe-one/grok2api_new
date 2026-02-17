"""Grok API 客户端 - 支持真实上下文的多轮对话"""

import re
import uuid
import orjson
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator, Set
from curl_cffi.requests import AsyncSession

from app.core.logger import logger
from app.core.config import settings
from app.services.token_manager import token_manager
from app.services.conversation_manager import conversation_manager
from app.services.image_upload import ImageUploadManager
from app.services.image_cache import image_cache
from app.services.headers import get_dynamic_headers
from app.api.v1.models import resolve_model

# 最大重试次数（使用不同的 Token）
MAX_RETRY_TOKENS = 3

# 需要过滤的 Grok 内部 XML 标签
FILTER_TAGS = ["xaiartifact", "xai:tool_usage_card", "grok:render"]


class GrokAPIError(Exception):
    """Grok API 错误"""
    def __init__(self, status_code: int, message: str, has_quota: bool = True):
        self.status_code = status_code
        self.message = message
        self.has_quota = has_quota
        super().__init__(f"Grok API 错误 ({status_code}): {message}")


class GrokClient:
    """Grok API 客户端"""

    # API 端点
    NEW_CONVERSATION_URL = f"{settings.grok_api_endpoint}/conversations/new"
    CONTINUE_CONVERSATION_URL = f"{settings.grok_api_endpoint}/conversations/{{conversation_id}}/responses"
    SHARE_CONVERSATION_URL = f"{settings.grok_api_endpoint}/conversations/{{conversation_id}}/share"
    CLONE_SHARE_LINK_URL = f"{settings.grok_api_endpoint}/share_links/{{share_link_id}}/clone"

    @staticmethod
    async def chat(
        messages: List[Dict[str, Any]],
        model: str = "grok-4.1-thinking",
        stream: bool = False,
        conversation_id: Optional[str] = None,
        thinking: Optional[bool] = None,
        **kwargs
    ) -> Tuple[Any, Optional[str], Optional[str], Optional[str]]:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式响应
            conversation_id: OpenAI 格式的会话 ID（用于继续对话）
            **kwargs: 其他参数

        Returns:
            (响应数据, OpenAI会话ID, Grok会话ID, Grok响应ID)
        """
        # 检查是否是继续对话
        context = None
        if conversation_id:
            context = await conversation_manager.get_conversation(conversation_id)

        # 如果没有提供 conversation_id，尝试通过消息历史自动识别
        if not context and len(messages) > 1:
            auto_conv_id = await conversation_manager.find_conversation_by_history(messages)
            if auto_conv_id:
                context = await conversation_manager.get_conversation(auto_conv_id)
                conversation_id = auto_conv_id
                logger.info(f"[GrokClient] 自动识别到会话: {conversation_id}")

        # 提取消息内容和图片
        message_text, image_urls = GrokClient._extract_message_content(messages, is_continue=bool(context))

        # 用于跟踪已尝试的 Token
        used_tokens: Set[str] = set()
        last_error = None

        # 重试循环：最多尝试 MAX_RETRY_TOKENS 个不同的 Token
        for attempt in range(MAX_RETRY_TOKENS):
            # 获取 Token（排除已使用的）
            token = await token_manager.get_token(exclude=used_tokens)
            if not token:
                if used_tokens:
                    raise Exception(f"已尝试 {len(used_tokens)} 个 Token 均失败，没有更多可用的 Token")
                raise Exception("没有可用的 Token")

            used_tokens.add(token)
            logger.info(f"[GrokClient] 尝试第 {attempt + 1}/{MAX_RETRY_TOKENS} 个 Token")

            try:
                result = await GrokClient._do_chat_request(
                    token=token,
                    message_text=message_text,
                    image_urls=image_urls,
                    model=model,
                    stream=stream,
                    conversation_id=conversation_id,
                    context=context,
                    messages=messages,
                    thinking=thinking
                )
                # 成功，记录并返回
                await token_manager.record_success(token)
                return result

            except GrokAPIError as e:
                last_error = e
                logger.warning(f"[GrokClient] Token 请求失败 (尝试 {attempt + 1}): {e.message}")

                # 根据状态码记录失败
                if e.status_code == 429:
                    await token_manager.record_failure(token, "429", has_quota=e.has_quota)
                elif e.status_code == 401:
                    await token_manager.record_failure(token, "auth")
                else:
                    await token_manager.record_failure(token, "normal")

                # 继续尝试下一个 Token
                continue

            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"[GrokClient] Token 请求失败 (尝试 {attempt + 1}): {error_str}")

                # 记录普通错误
                await token_manager.record_failure(token, "normal")

                # 继续尝试下一个 Token
                continue

        # 所有重试都失败
        raise Exception(f"已尝试 {MAX_RETRY_TOKENS} 个 Token 均失败: {last_error}")

    @staticmethod
    async def _do_chat_request(
        token: str,
        message_text: str,
        image_urls: List[str],
        model: str,
        stream: bool,
        conversation_id: Optional[str],
        context: Optional[Any],
        messages: List[Dict[str, Any]] = None,
        thinking: Optional[bool] = None
    ) -> Tuple[Any, Optional[str], Optional[str], Optional[str]]:
        """执行实际的聊天请求（内部方法）"""
        # 解析模型名称 → (grok内部名, modelMode, 规范名)
        grok_model, model_mode, resolved_model = resolve_model(model)
        if resolved_model != model:
            logger.info(f"[GrokClient] 模型映射: {model} -> {grok_model} (mode={model_mode})")

        # 自动检测是否显示思考过程
        show_thinking = thinking
        if show_thinking is None:
            show_thinking = "THINKING" in model_mode

        # 上传图片
        file_ids = []
        if image_urls:
            logger.info(f"[GrokClient] 检测到 {len(image_urls)} 张图片，开始上传...")
            for img_url in image_urls:
                file_id, file_uri = await ImageUploadManager.upload(img_url, token)
                if file_id:
                    file_ids.append(file_id)
            logger.info(f"[GrokClient] 图片上传完成，成功 {len(file_ids)}/{len(image_urls)}")

        # 构建请求
        if context:
            # 继续对话 - 检查是否需要跨账号克隆
            if token != context.token:
                if context.share_link_id:
                    logger.info(f"[GrokClient] Token 不同，克隆会话: shareLinkId={context.share_link_id}")
                    new_conv_id, new_resp_id = await GrokClient._clone_conversation(token, context.share_link_id)
                    if new_conv_id and new_resp_id:
                        # 更新 context 为克隆后的会话
                        context.conversation_id = new_conv_id
                        context.last_response_id = new_resp_id
                        context.token = token
                        logger.info(f"[GrokClient] 会话已克隆到新账号: {new_conv_id}")
                    else:
                        logger.warning("[GrokClient] 克隆失败，降级为新对话")
                        # 重新提取完整消息（包含所有历史）
                        message_text, _ = GrokClient._extract_message_content(messages, is_continue=False)
                        context = None
                else:
                    logger.warning("[GrokClient] Token 不同但无 share_link_id，降级为新对话")
                    message_text, _ = GrokClient._extract_message_content(messages, is_continue=False)
                    context = None

        if context:
            # 继续对话 - message_text 已经是最后一条新消息
            url = GrokClient.CONTINUE_CONVERSATION_URL.format(
                conversation_id=context.conversation_id
            )
            payload = GrokClient._build_continue_payload(
                message_text,
                grok_model,
                model_mode,
                context.last_response_id,
                file_ids
            )
            logger.info(f"[GrokClient] 继续对话: {conversation_id} -> {context.conversation_id}, 只发送新消息")

            # 重要：继续对话时必须使用流式响应，因为非流式不返回 AI 回复
            force_stream = True
        else:
            # 新对话 - message_text 包含所有初始消息
            url = GrokClient.NEW_CONVERSATION_URL
            payload = GrokClient._build_new_payload(message_text, grok_model, model_mode, file_ids)
            logger.info(f"[GrokClient] 创建新对话")
            force_stream = False

        # 构建请求头
        pathname = url.split("/rest/app-chat")[-1] if "/rest/app-chat" in url else "/rest/app-chat/conversations/new"
        headers = GrokClient._build_headers(token, pathname)

        # 发送请求
        session = AsyncSession(impersonate="chrome120")
        proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

        response = await session.post(
            url,
            headers=headers,
            data=orjson.dumps(payload),
            timeout=settings.request_timeout,
            stream=True,
            proxies=proxies
        )

        if response.status_code != 200:
            error_text = await response.atext()
            await session.close()
            logger.error(f"[GrokClient] 请求失败: {response.status_code} - {error_text[:200]}")

            # 检测是否有额度（429 时解析响应）
            has_quota = True
            if response.status_code == 429:
                try:
                    # 尝试解析响应判断额度
                    error_lower = error_text.lower()
                    if "quota" in error_lower or "limit" in error_lower or "exceeded" in error_lower:
                        # 检查是否明确说无额度
                        if "no quota" in error_lower or "quota exceeded" in error_lower or "0 remaining" in error_lower:
                            has_quota = False
                except Exception:
                    pass

            raise GrokAPIError(response.status_code, error_text[:200], has_quota)

        # 处理响应
        if stream:
            # 用户请求流式响应
            result = await GrokClient._process_stream(response, session, token, conversation_id, context, messages, show_thinking)
            return result, conversation_id, None, None
        elif force_stream:
            # 继续对话时强制使用流式，然后转换为非流式
            logger.info(f"[GrokClient] 继续对话强制使用流式响应")
            content, grok_resp_id = await GrokClient._collect_stream_to_text(response, session, token, show_thinking)

            # 分享会话（用于下次跨账号继续）
            grok_conv_id = context.conversation_id
            share_link_id = None
            if grok_resp_id:
                share_link_id = await GrokClient._share_conversation(token, grok_conv_id, grok_resp_id)

            # 更新会话
            if grok_resp_id:
                await conversation_manager.update_conversation(
                    conversation_id, grok_resp_id,
                    share_link_id=share_link_id,
                    grok_conversation_id=grok_conv_id,
                    token=token
                )

            return content, conversation_id, grok_conv_id, grok_resp_id
        else:
            # 非流式响应（仅新对话）
            result, grok_conv_id, grok_resp_id = await GrokClient._process_normal(
                response, session, token, is_continue=False
            )

            # 分享会话（用于下次跨账号继续）
            share_link_id = ""
            if grok_conv_id and grok_resp_id:
                share_link_id = await GrokClient._share_conversation(token, grok_conv_id, grok_resp_id) or ""

            # 创建新会话
            openai_conv_id = await conversation_manager.create_conversation(
                token, grok_conv_id, grok_resp_id, messages, share_link_id=share_link_id
            )

            return result, openai_conv_id, grok_conv_id, grok_resp_id

    @staticmethod
    def _extract_message_content(messages: List[Dict[str, Any]], is_continue: bool = False) -> Tuple[str, List[str]]:
        """提取消息文本和图片 - 真实上下文，不拼接历史

        Args:
            messages: 消息列表
            is_continue: 是否是继续对话

        Returns:
            (文本内容, 图片URL列表)
        """
        images = []

        if not messages:
            return "", images

        if is_continue:
            # 继续对话：只发送最后一条新消息，完全依赖 Grok 的 conversationId 维护上下文
            last_msg = messages[-1]
            content = last_msg.get("content", "")

            # 处理多模态内容
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        if img_data := item.get("image_url"):
                            if url := img_data.get("url"):
                                images.append(url)
                content = "".join(text_parts)

            return content, images
        else:
            # 首次对话：拼接所有消息
            parts = []

            # 判断是否有多轮对话（user/assistant 消息超过1条时加角色标记）
            user_assistant_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
            has_multi_turn = user_assistant_count > 1

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # 处理多模态内容
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            if img_data := item.get("image_url"):
                                if url := img_data.get("url"):
                                    images.append(url)
                    content = "".join(text_parts)

                if content:
                    if has_multi_turn:
                        # 多轮对话加角色标记，让 AI 理解对话结构
                        role_label = {"system": "[System]", "user": "[User]", "assistant": "[Assistant]"}.get(role, "[User]")
                        parts.append(f"{role_label}\n{content}")
                    else:
                        parts.append(content)

            return "\n\n".join(parts), images

    @staticmethod
    def _build_new_payload(message: str, grok_model: str, model_mode: str, file_ids: List[str] = None) -> Dict:
        """构建新对话的请求载荷"""
        return {
            "temporary": True,
            "modelName": grok_model,
            "message": message,
            "fileAttachments": file_ids or [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": False,
            "disableMemory": False,
            "forceSideBySide": False,
            "modelMode": model_mode,
            "isAsyncChat": False
        }

    @staticmethod
    def _build_continue_payload(message: str, grok_model: str, model_mode: str, parent_response_id: str, file_ids: List[str] = None) -> Dict:
        """构建继续对话的请求载荷"""
        payload = GrokClient._build_new_payload(message, grok_model, model_mode, file_ids)
        payload["parentResponseId"] = parent_response_id
        return payload

    @staticmethod
    def _build_headers(token: str, pathname: str = "/rest/app-chat/conversations/new") -> Dict[str, str]:
        """构建请求头"""
        # 获取动态请求头
        headers = get_dynamic_headers(pathname)

        # 添加 Cookie（确保 Token 包含 sso= 前缀）
        if not token.startswith("sso="):
            token = f"sso={token}"
        headers["Cookie"] = token

        return headers

    @staticmethod
    async def _share_conversation(token: str, conversation_id: str, response_id: str) -> Optional[str]:
        """分享会话，获取 shareLinkId（用于跨账号克隆）

        Args:
            token: 当前账号的 SSO token
            conversation_id: Grok 会话 ID
            response_id: 最后一条响应 ID

        Returns:
            shareLinkId 或 None
        """
        if not conversation_id or not response_id:
            return None

        url = GrokClient.SHARE_CONVERSATION_URL.format(conversation_id=conversation_id)
        pathname = f"/rest/app-chat/conversations/{conversation_id}/share"
        headers = GrokClient._build_headers(token, pathname)
        payload = {"responseId": response_id, "allowIndexing": True}

        proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

        try:
            async with AsyncSession(impersonate="chrome120") as session:
                response = await session.post(
                    url,
                    headers=headers,
                    data=orjson.dumps(payload),
                    timeout=30,
                    proxies=proxies
                )

                if response.status_code == 200:
                    data = orjson.loads(response.content)
                    share_link_id = data.get("shareLinkId")
                    logger.info(f"[GrokClient] 会话已分享: conv={conversation_id}, shareLinkId={share_link_id}")
                    return share_link_id
                else:
                    error_text = response.text
                    logger.warning(f"[GrokClient] 分享会话失败: {response.status_code} - {error_text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"[GrokClient] 分享会话异常: {e}")
            return None

    @staticmethod
    async def _clone_conversation(token: str, share_link_id: str) -> Tuple[Optional[str], Optional[str]]:
        """克隆分享的会话到当前账号

        Args:
            token: 新账号的 SSO token
            share_link_id: 分享链接 ID

        Returns:
            (新会话ID, 最后一条助手响应ID) 或 (None, None)
        """
        if not share_link_id:
            return None, None

        url = GrokClient.CLONE_SHARE_LINK_URL.format(share_link_id=share_link_id)
        pathname = f"/rest/app-chat/share_links/{share_link_id}/clone"
        headers = GrokClient._build_headers(token, pathname)

        proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

        try:
            async with AsyncSession(impersonate="chrome120") as session:
                response = await session.post(
                    url,
                    headers=headers,
                    data=orjson.dumps({}),
                    timeout=30,
                    proxies=proxies
                )

                if response.status_code == 200:
                    data = orjson.loads(response.content)

                    new_conv_id = data.get("conversation", {}).get("conversationId")
                    if not new_conv_id:
                        logger.warning("[GrokClient] 克隆响应中缺少 conversationId")
                        return None, None

                    # 找到最后一条助手响应的 responseId 作为 parentResponseId
                    responses = data.get("responses", [])
                    last_resp_id = None
                    for resp in reversed(responses):
                        if resp.get("sender") == "assistant":
                            last_resp_id = resp.get("responseId")
                            break

                    # 如果没有助手响应，用最后一条响应
                    if not last_resp_id and responses:
                        last_resp_id = responses[-1].get("responseId")

                    logger.info(f"[GrokClient] 会话已克隆: newConv={new_conv_id}, lastRespId={last_resp_id}")
                    return new_conv_id, last_resp_id
                else:
                    error_text = response.text
                    logger.warning(f"[GrokClient] 克隆会话失败: {response.status_code} - {error_text[:200]}")
                    return None, None
        except Exception as e:
            logger.error(f"[GrokClient] 克隆会话异常: {e}")
            return None, None

    @staticmethod
    def _filter_tags_regex(content: str) -> str:
        """过滤内容中的 Grok XML 标签（用于非流式响应）"""
        if not content:
            return content
        for tag in FILTER_TAGS:
            pattern = rf"<{re.escape(tag)}[^>]*>.*?</{re.escape(tag)}>|<{re.escape(tag)}[^>]*/>"
            content = re.sub(pattern, "", content, flags=re.DOTALL)
        return content

    @staticmethod
    async def _process_normal(response, session, token: str, is_continue: bool = False) -> Tuple[str, str, str]:
        """处理非流式响应

        Args:
            is_continue: 是否是继续对话（继续对话时响应格式不同）
        """
        try:
            content = ""
            grok_conversation_id = None
            grok_response_id = None
            generated_images = []  # 收集生成的图片

            # 读取完整响应
            full_text = await response.atext()
            logger.debug(f"[GrokClient] 完整响应前500字符: {full_text[:500]}")

            # 按行分割处理
            lines = full_text.strip().split('\n')
            logger.debug(f"[GrokClient] 响应行数: {len(lines)}, 是否继续对话: {is_continue}")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = orjson.loads(line)

                    # 提取会话 ID - 从 result.conversation
                    if result := data.get("result"):
                        if conversation := result.get("conversation"):
                            if conv_id := conversation.get("conversationId"):
                                grok_conversation_id = conv_id
                                logger.info(f"[GrokClient] 提取到会话ID: {conv_id}")

                        # 直接从 result.modelResponse 提取图片（备用路径）
                        if model_resp_direct := result.get("modelResponse"):
                            if resp_id := model_resp_direct.get("responseId"):
                                grok_response_id = resp_id
                            if images := model_resp_direct.get("generatedImageUrls"):
                                if images:
                                    generated_images.extend(images)
                                    logger.info(f"[GrokClient] 从 result.modelResponse 检测到 {len(images)} 张图片")
                            if msg := model_resp_direct.get("message"):
                                content = msg

                        # 从 result.token 提取文本片段
                        if token_text := result.get("token"):
                            if isinstance(token_text, str) and token_text:
                                content += token_text

                        # 提取响应数据 - 从 result.response（主要路径）
                        if response_data := result.get("response"):
                            if isinstance(response_data, dict):
                                # 提取响应 ID
                                if resp_id := response_data.get("responseId"):
                                    grok_response_id = resp_id

                                # 检查 modelResponse（独立检查）
                                if model_resp := response_data.get("modelResponse"):
                                    if resp_id := model_resp.get("responseId"):
                                        grok_response_id = resp_id
                                        logger.info(f"[GrokClient] 提取到响应ID: {resp_id}")

                                    if msg := model_resp.get("message"):
                                        content = msg
                                        logger.debug(f"[GrokClient] 提取到内容: {msg[:100] if msg else ''}")

                                    # 提取生成的图片
                                    if images := model_resp.get("generatedImageUrls"):
                                        if images:
                                            generated_images.extend(images)
                                            logger.info(f"[GrokClient] 从 response.modelResponse 检测到 {len(images)} 张图片")

                                # 从 token 累积内容
                                if token_text := response_data.get("token"):
                                    if isinstance(token_text, str) and token_text:
                                        content += token_text

                        # 继续对话时，可能只返回 userResponse
                        if is_continue and result.get("userResponse"):
                            user_resp = result["userResponse"]
                            if resp_id := user_resp.get("responseId"):
                                grok_response_id = resp_id
                                logger.info(f"[GrokClient] 继续对话，提取到用户响应ID: {resp_id}")

                except Exception as e:
                    logger.debug(f"[GrokClient] 解析行失败: {e}, 数据: {line[:100]}")
                    continue

            # 处理生成的图片 - 去重
            if generated_images:
                generated_images = list(dict.fromkeys(generated_images))
                content = await GrokClient._append_images(content, generated_images, token)

            # 过滤 Grok 内部 XML 标签
            content = GrokClient._filter_tags_regex(content)

            # 如果是继续对话且没有内容，说明需要等待 AI 回复
            if is_continue and not content:
                logger.warning(f"[GrokClient] 继续对话返回空内容，可能需要使用流式响应或轮询")

            logger.info(f"[GrokClient] 解析完成: conv_id={grok_conversation_id}, resp_id={grok_response_id}, content_len={len(content)}, images={len(generated_images)}")
            return content, grok_conversation_id, grok_response_id

        finally:
            await session.close()

    @staticmethod
    async def _process_stream(
        response,
        session,
        token: str,
        conversation_id: Optional[str],
        context: Optional[Any],
        messages: List[Dict[str, Any]] = None,
        show_thinking: bool = False
    ) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        async def stream_generator():
            try:
                grok_conversation_id = context.conversation_id if context else None
                grok_response_id = None
                openai_conv_id = conversation_id
                generated_images = []  # 收集生成的图片
                is_image_mode = False  # 是否进入图片生成模式

                # 思考状态
                think_opened = False
                first_think_token = True  # 跳过第一条固定的思考开头

                # 标签过滤状态
                in_filter_tag = False
                tag_buffer = ""

                def filter_token(token_text):
                    """逐字符过滤 XML 标签（支持跨 token 的标签）"""
                    nonlocal in_filter_tag, tag_buffer
                    if not FILTER_TAGS or not token_text:
                        return token_text

                    result = []
                    i = 0
                    while i < len(token_text):
                        char = token_text[i]

                        if in_filter_tag:
                            tag_buffer += char
                            if char == ">":
                                # 检测自闭合标签或结束标签
                                if "/>" in tag_buffer:
                                    in_filter_tag = False
                                    tag_buffer = ""
                                else:
                                    for tag in FILTER_TAGS:
                                        if f"</{tag}>" in tag_buffer:
                                            in_filter_tag = False
                                            tag_buffer = ""
                                            break
                            i += 1
                            continue

                        if char == "<":
                            remaining = token_text[i:]
                            tag_started = False
                            for tag in FILTER_TAGS:
                                if remaining.startswith(f"<{tag}"):
                                    tag_started = True
                                    break
                                # 部分匹配（标签可能跨 token 分割）
                                if len(remaining) < len(tag) + 2:
                                    prefix = f"<{tag}"
                                    if prefix.startswith(remaining):
                                        tag_started = True
                                        break

                            if tag_started:
                                in_filter_tag = True
                                tag_buffer = char
                                i += 1
                                continue

                        result.append(char)
                        i += 1

                    return "".join(result)

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = orjson.loads(line)

                        # 提取会话信息
                        if result := data.get("result"):
                            # 图片生成进度
                            if img := result.get("streamingImageGenerationResponse"):
                                is_image_mode = True
                                if show_thinking:
                                    if not think_opened:
                                        yield "<think>\n"
                                        think_opened = True
                                    idx = img.get("imageIndex", 0) + 1
                                    progress = img.get("progress", 0)
                                    yield f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                                continue

                            # 检测图片生成模式 - 在 result.response.imageAttachmentInfo
                            if response_data := result.get("response"):
                                if isinstance(response_data, dict):
                                    if response_data.get("imageAttachmentInfo"):
                                        is_image_mode = True
                                        logger.info("[GrokClient] 进入图片生成模式")

                            # 提取会话 ID
                            if conversation := result.get("conversation"):
                                if conv_id := conversation.get("conversationId"):
                                    grok_conversation_id = conv_id

                            # 直接从 result.modelResponse 提取图片和响应ID（备用路径）
                            if model_resp_direct := result.get("modelResponse"):
                                if resp_id := model_resp_direct.get("responseId"):
                                    grok_response_id = resp_id
                                if images := model_resp_direct.get("generatedImageUrls"):
                                    if images:
                                        generated_images.extend(images)
                                        logger.info(f"[GrokClient] 流式从 result.modelResponse 检测到 {len(images)} 张图片")
                                # modelResponse 到达表示生成结束，关闭 think 标签
                                if think_opened and show_thinking:
                                    if msg := model_resp_direct.get("message"):
                                        yield msg + "\n"
                                    yield "</think>\n"
                                    think_opened = False

                            # 从 result.response 提取（主要路径）
                            response_data = result.get("response")
                            if isinstance(response_data, dict):
                                # 提取响应 ID
                                if resp_id := response_data.get("responseId"):
                                    grok_response_id = resp_id

                                # 检查 modelResponse（独立检查）
                                if model_resp := response_data.get("modelResponse"):
                                    if resp_id := model_resp.get("responseId"):
                                        grok_response_id = resp_id
                                    # 提取生成的图片
                                    if images := model_resp.get("generatedImageUrls"):
                                        if images:
                                            generated_images.extend(images)
                                            logger.info(f"[GrokClient] 流式从 response.modelResponse 检测到 {len(images)} 张图片: {images}")
                                    # modelResponse 到达，关闭 think 标签
                                    if think_opened and show_thinking:
                                        if msg := model_resp.get("message"):
                                            yield msg + "\n"
                                        yield "</think>\n"
                                        think_opened = False

                                # 提取 token（文本片段）- 带思考检测
                                # token 可能在 response_data.token 或 result.token
                                # isThinking 也可能在不同层级
                                if not is_image_mode:
                                    token_text = response_data.get("token")
                                    is_thinking = response_data.get("isThinking", False)

                                    # 备用：token 在 result 顶层（继续对话时常见）
                                    if token_text is None:
                                        token_text = result.get("token")
                                        is_thinking = result.get("isThinking", is_thinking)

                                    if token_text and isinstance(token_text, str):
                                        if show_thinking:
                                            if is_thinking:
                                                # 跳过固定的思考开头
                                                if first_think_token:
                                                    first_think_token = False
                                                    if "Thinking about" in token_text:
                                                        continue
                                                # 思考中 → 包裹在 <think> 标签内（同时过滤 XML）
                                                if not think_opened:
                                                    yield "<think>\n"
                                                    think_opened = True
                                                yield filter_token(token_text)
                                            else:
                                                # 非思考 → 关闭 think 标签并过滤 XML
                                                if think_opened:
                                                    yield "\n</think>\n"
                                                    think_opened = False
                                                yield filter_token(token_text)
                                        else:
                                            # 不显示思考 → 跳过思考 token，过滤正常 token
                                            if not is_thinking:
                                                yield filter_token(token_text)

                            # 从 result.token 直接提取（无 response 对象时的备用路径）
                            elif not is_image_mode:
                                if token_text := result.get("token"):
                                    if isinstance(token_text, str):
                                        is_thinking = result.get("isThinking", False)
                                        if show_thinking:
                                            if is_thinking:
                                                if first_think_token:
                                                    first_think_token = False
                                                    if "Thinking about" in token_text:
                                                        continue
                                                if not think_opened:
                                                    yield "<think>\n"
                                                    think_opened = True
                                                yield filter_token(token_text)
                                            else:
                                                if think_opened:
                                                    yield "\n</think>\n"
                                                    think_opened = False
                                                yield filter_token(token_text)
                                        else:
                                            if not is_thinking:
                                                yield filter_token(token_text)

                    except Exception as e:
                        logger.debug(f"[GrokClient] 流式解析失败: {e}")
                        continue

                # 流结束时如果 think 标签未关闭，关闭它
                if think_opened:
                    yield "\n</think>\n"

                # 流式结束后处理图片 - 去重
                logger.info(f"[GrokClient] 流式结束，收集到 {len(generated_images)} 张图片, is_image_mode={is_image_mode}")
                if generated_images:
                    generated_images = list(dict.fromkeys(generated_images))
                    image_content = await GrokClient._append_images("", generated_images, token)
                    if image_content:
                        yield image_content

                # 流式结束后更新会话并分享
                if grok_conversation_id and grok_response_id:
                    # 分享会话（用于下次跨账号继续）
                    share_link_id = await GrokClient._share_conversation(token, grok_conversation_id, grok_response_id)

                    if context:
                        # 更新现有会话
                        await conversation_manager.update_conversation(
                            openai_conv_id, grok_response_id, messages,
                            share_link_id=share_link_id,
                            grok_conversation_id=grok_conversation_id,
                            token=token
                        )
                    else:
                        # 创建新会话
                        await conversation_manager.create_conversation(
                            token, grok_conversation_id, grok_response_id, messages,
                            share_link_id=share_link_id or ""
                        )

            finally:
                await session.close()

        return stream_generator()

    @staticmethod
    async def _collect_stream_to_text(response, session, auth_token: str = "", show_thinking: bool = False):
        """收集流式响应为完整文本"""
        try:
            content = ""
            thinking_content = ""
            grok_response_id = None
            generated_images = []
            is_image_mode = False
            is_in_thinking = False

            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = orjson.loads(line)

                    if result := data.get("result"):
                        # 图片生成进度
                        if img := result.get("streamingImageGenerationResponse"):
                            is_image_mode = True
                            if show_thinking:
                                idx = img.get("imageIndex", 0) + 1
                                progress = img.get("progress", 0)
                                thinking_content += f"正在生成第{idx}张图片中，当前进度{progress}%\n"
                            continue

                        # 从 result.response 提取（主要路径）
                        response_data = result.get("response")
                        if isinstance(response_data, dict):
                            # 检测图片生成模式
                            if response_data.get("imageAttachmentInfo"):
                                is_image_mode = True

                            # 提取响应 ID
                            if resp_id := response_data.get("responseId"):
                                grok_response_id = resp_id

                            # 检查 modelResponse（独立检查）
                            if model_resp := response_data.get("modelResponse"):
                                if resp_id := model_resp.get("responseId"):
                                    grok_response_id = resp_id
                                if msg := model_resp.get("message"):
                                    content = msg
                                if images := model_resp.get("generatedImageUrls"):
                                    if images:
                                        generated_images.extend(images)
                                # modelResponse 到达，如果有图片生成进度，追加 message
                                if show_thinking and thinking_content and is_image_mode:
                                    if msg := model_resp.get("message"):
                                        thinking_content += msg + "\n"

                            # 从 token 提取文本（非图片模式）
                            if not is_image_mode:
                                is_thinking = response_data.get("isThinking", False)

                                if token_text := response_data.get("token"):
                                    if isinstance(token_text, str):
                                        if is_thinking:
                                            if show_thinking:
                                                thinking_content += token_text
                                        else:
                                            is_in_thinking = False
                                            content += token_text

                        # 直接从 result.modelResponse 提取（备用路径）
                        if model_resp_direct := result.get("modelResponse"):
                            if resp_id := model_resp_direct.get("responseId"):
                                grok_response_id = resp_id
                            if images := model_resp_direct.get("generatedImageUrls"):
                                if images:
                                    generated_images.extend(images)
                            if msg := model_resp_direct.get("message"):
                                content = msg

                        # 从 result.token 提取文本（非图片模式，备用路径）
                        if not is_image_mode:
                            if token_text := result.get("token"):
                                if isinstance(token_text, str):
                                    is_thinking = result.get("isThinking", False)
                                    if is_thinking:
                                        if show_thinking:
                                            thinking_content += token_text
                                    else:
                                        content += token_text

                except Exception as e:
                    logger.debug(f"[GrokClient] 流式解析失败: {e}")
                    continue

            # 过滤 XML 标签
            content = GrokClient._filter_tags_regex(content)

            # 如果有思考内容，过滤标签并添加 <think> 标签
            if show_thinking and thinking_content:
                thinking_content = GrokClient._filter_tags_regex(thinking_content)
                content = f"<think>\n{thinking_content}\n</think>\n{content}"

            # 处理图片 - 去重
            if generated_images:
                generated_images = list(dict.fromkeys(generated_images))
                content = await GrokClient._append_images(content, generated_images, auth_token)

            logger.info(f"[GrokClient] 流式收集完成: resp_id={grok_response_id}, content_len={len(content)}, images={len(generated_images)}")
            return content, grok_response_id

        finally:
            await session.close()

    @staticmethod
    async def _append_images(content: str, images: list, auth_token: str) -> str:
        """追加生成的图片到内容（Markdown 格式）"""
        base_url = settings.base_url.rstrip("/") if settings.base_url else ""

        for img in images:
            try:
                # 下载并缓存图片
                cache_path = await image_cache.download(f"/{img}", auth_token)
                if cache_path:
                    # 转换路径格式：users/xxx/image.jpg -> users-xxx-image.jpg
                    img_path = img.replace('/', '-')
                    img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                    content += f"\n\n![Generated Image]({img_url})"
                    logger.info(f"[GrokClient] 图片已缓存: {img_url}")
                else:
                    # 下载失败，使用原始 URL
                    content += f"\n\n![Generated Image](https://assets.grok.com/{img})"
                    logger.warning(f"[GrokClient] 图片缓存失败，使用原始URL: {img}")
            except Exception as e:
                logger.warning(f"[GrokClient] 处理图片失败: {e}")
                content += f"\n\n![Generated Image](https://assets.grok.com/{img})"

        return content
