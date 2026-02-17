"""会话管理器 - 管理多轮对话上下文，支持自动回收"""

import asyncio
import time
import uuid
import hashlib
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from app.core.logger import logger
from app.core.storage import storage_manager
from app.core.config import settings


# 自动清理间隔（秒）- 每10分钟检查一次
CLEANUP_INTERVAL = 600


@dataclass
class ConversationContext:
    """会话上下文"""
    conversation_id: str  # Grok 会话 ID
    last_response_id: str  # 最后一条响应 ID
    created_at: float
    updated_at: float
    message_count: int
    token: str  # 使用的 token
    history_hash: str = ""  # 消息历史的哈希值，用于自动识别会话
    share_link_id: str = ""  # 分享链接 ID，用于跨账号继续对话


class ConversationManager:
    """会话管理器 - 支持真实的多轮对话，自动回收过期会话"""

    def __init__(self):
        # 内存存储：{openai_conversation_id: ConversationContext}
        self.conversations: Dict[str, ConversationContext] = {}
        # Token 到会话的映射：{token: [conversation_ids]}
        self.token_conversations: Dict[str, List[str]] = {}
        # 历史哈希到会话 ID 的映射：{history_hash: openai_conversation_id}
        self.hash_to_conversation: Dict[str, str] = {}
        self.initialized = False
        # 自动清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        # 清理统计
        self._last_cleanup_time: float = 0
        self._total_cleaned: int = 0

    @staticmethod
    def compute_history_hash(messages: List[Dict[str, Any]], exclude_last_user: bool = False) -> str:
        """计算消息历史的哈希值 - 基于 system + 所有 user 消息

        存储时：hash(system + 所有 user) → 作为下次查找目标
        查找时：hash(system + 除最后一条外的所有 user) → 匹配已存储的哈希

        Args:
            messages: 完整消息列表
            exclude_last_user: 是否排除最后一条 user 消息（查找时为 True）
        """
        if not messages:
            return ""

        # 收集 system 和 user 消息
        system_parts = []
        user_parts = []
        has_assistant = False

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    content = "".join(text_parts)
                system_parts.append(f"system:{content}")

            elif role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    content = "".join(text_parts)
                user_parts.append(f"user:{content}")

            elif role == "assistant":
                has_assistant = True

        # 查找模式：排除最后一条 user（仅当存在 assistant 消息时，表示这是续接对话）
        if exclude_last_user and has_assistant and user_parts:
            user_parts = user_parts[:-1]

        key_parts = system_parts + user_parts
        if not key_parts:
            return ""

        hash_input = "\n".join(key_parts).encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()[:16]

    async def init(self):
        """初始化"""
        if self.initialized:
            return

        # 从存储加载会话数据
        data = await storage_manager.load_json("conversations.json", {})

        # 恢复会话
        for conv_id, conv_data in data.get("conversations", {}).items():
            # 兼容旧数据
            if "history_hash" not in conv_data:
                conv_data["history_hash"] = ""
            if "share_link_id" not in conv_data:
                conv_data["share_link_id"] = ""
            context = ConversationContext(**conv_data)
            self.conversations[conv_id] = context
            # 恢复哈希映射
            if context.history_hash:
                self.hash_to_conversation[context.history_hash] = conv_id

        # 恢复 token 映射
        self.token_conversations = data.get("token_conversations", {})

        # 清理过期会话
        await self._cleanup_expired()

        # 启动自动清理任务
        self._start_cleanup_task()

        logger.info(f"[ConversationManager] 已加载 {len(self.conversations)} 个会话，自动清理已启动")
        self.initialized = True

    async def find_conversation_by_history(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """通过消息历史自动查找会话 ID

        查找逻辑：排除最后一条 user 消息后计算哈希，匹配已存储的完整哈希
        这样 "上次存储的 hash(sys+u1+u2)" == "这次查找的 hash(sys+u1+u2)" (排除了新的 u3)

        Args:
            messages: 完整的消息历史

        Returns:
            找到的会话 ID，如果没有找到返回 None
        """
        if not messages:
            return None

        # 查找模式：排除最后一条 user 消息
        history_hash = self.compute_history_hash(messages, exclude_last_user=True)
        if not history_hash:
            return None

        # 查找匹配的会话
        conv_id = self.hash_to_conversation.get(history_hash)
        if conv_id:
            # 检查会话是否仍然有效
            context = await self.get_conversation(conv_id)
            if context:
                logger.info(f"[ConversationManager] 自动识别会话: {conv_id}, hash={history_hash}")
                return conv_id
            else:
                # 会话已过期，清理哈希映射
                del self.hash_to_conversation[history_hash]

        return None

    async def create_conversation(self, token: str, grok_conversation_id: str,
                                  grok_response_id: str, messages: List[Dict[str, Any]] = None,
                                  share_link_id: str = "") -> str:
        """创建新会话

        Args:
            token: 使用的 SSO token
            grok_conversation_id: Grok 返回的会话 ID
            grok_response_id: Grok 返回的响应 ID
            messages: 消息历史（用于计算哈希）

        Returns:
            OpenAI 格式的会话 ID
        """
        # 生成 OpenAI 格式的会话 ID
        openai_conv_id = f"conv-{uuid.uuid4().hex[:24]}"

        # 计算历史哈希（只基于 system 和 user 消息）
        history_hash = ""
        if messages:
            history_hash = self.compute_history_hash(messages)

        # 创建会话上下文
        context = ConversationContext(
            conversation_id=grok_conversation_id,
            last_response_id=grok_response_id,
            created_at=time.time(),
            updated_at=time.time(),
            message_count=1,
            token=token,
            history_hash=history_hash,
            share_link_id=share_link_id
        )

        # 保存到内存
        self.conversations[openai_conv_id] = context

        # 保存哈希映射
        if history_hash:
            self.hash_to_conversation[history_hash] = openai_conv_id

        # 更新 token 映射
        if token not in self.token_conversations:
            self.token_conversations[token] = []
        self.token_conversations[token].append(openai_conv_id)

        # 限制每个 token 的会话数量
        await self._limit_token_conversations(token)

        logger.info(f"[ConversationManager] 创建会话: {openai_conv_id} -> {grok_conversation_id}, hash={history_hash}")

        # 异步保存
        await self._save_async()

        return openai_conv_id

    async def get_conversation(self, openai_conv_id: str) -> Optional[ConversationContext]:
        """获取会话上下文"""
        context = self.conversations.get(openai_conv_id)

        if context:
            # 检查是否过期
            if time.time() - context.updated_at > settings.conversation_ttl:
                logger.info(f"[ConversationManager] 会话已过期: {openai_conv_id}")
                await self.delete_conversation(openai_conv_id)
                return None

        return context

    async def update_conversation(self, openai_conv_id: str, grok_response_id: str,
                                   messages: List[Dict[str, Any]] = None,
                                   share_link_id: str = None,
                                   grok_conversation_id: str = None,
                                   token: str = None):
        """更新会话（添加新的响应 ID，并刷新哈希）"""
        context = self.conversations.get(openai_conv_id)
        if not context:
            logger.warning(f"[ConversationManager] 会话不存在: {openai_conv_id}")
            return

        context.last_response_id = grok_response_id
        context.updated_at = time.time()
        context.message_count += 1

        if share_link_id is not None:
            context.share_link_id = share_link_id
        if grok_conversation_id is not None:
            context.conversation_id = grok_conversation_id
        if token is not None:
            context.token = token

        # 刷新哈希：加入新的 user 消息后重新计算
        if messages:
            new_hash = self.compute_history_hash(messages)
            if new_hash and new_hash != context.history_hash:
                # 移除旧哈希映射
                if context.history_hash and context.history_hash in self.hash_to_conversation:
                    del self.hash_to_conversation[context.history_hash]
                # 存储新哈希映射
                context.history_hash = new_hash
                self.hash_to_conversation[new_hash] = openai_conv_id
                logger.debug(f"[ConversationManager] 哈希已更新: {openai_conv_id}, newHash={new_hash}")

        logger.debug(f"[ConversationManager] 更新会话: {openai_conv_id}, 消息数: {context.message_count}")

        # 异步保存
        await self._save_async()

    async def delete_conversation(self, openai_conv_id: str):
        """删除会话"""
        context = self.conversations.pop(openai_conv_id, None)
        if context:
            # 从哈希映射中移除
            if context.history_hash and context.history_hash in self.hash_to_conversation:
                del self.hash_to_conversation[context.history_hash]

            # 从 token 映射中移除
            if context.token in self.token_conversations:
                try:
                    self.token_conversations[context.token].remove(openai_conv_id)
                except ValueError:
                    pass

            logger.info(f"[ConversationManager] 删除会话: {openai_conv_id}")
            await self._save_async()

    async def _limit_token_conversations(self, token: str):
        """限制每个 token 的会话数量"""
        conv_ids = self.token_conversations.get(token, [])

        if len(conv_ids) > settings.max_conversations_per_token:
            # 删除最旧的会话
            to_delete = len(conv_ids) - settings.max_conversations_per_token
            for conv_id in conv_ids[:to_delete]:
                context = self.conversations.get(conv_id)
                if context:
                    del self.conversations[conv_id]
                    logger.info(f"[ConversationManager] 清理旧会话: {conv_id}")

            # 更新映射
            self.token_conversations[token] = conv_ids[to_delete:]

    async def _cleanup_expired(self):
        """清理过期会话"""
        now = time.time()
        expired = []

        for conv_id, context in self.conversations.items():
            if now - context.updated_at > settings.conversation_ttl:
                expired.append(conv_id)

        for conv_id in expired:
            await self.delete_conversation(conv_id)

        if expired:
            self._total_cleaned += len(expired)
            logger.info(f"[ConversationManager] 清理了 {len(expired)} 个过期会话")

        self._last_cleanup_time = now
        return len(expired)

    def _start_cleanup_task(self):
        """启动自动清理任务"""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(CLEANUP_INTERVAL)
                    cleaned = await self._cleanup_expired()
                    if cleaned > 0:
                        await self._save_async()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[ConversationManager] 自动清理出错: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"[ConversationManager] 自动清理任务已启动，间隔 {CLEANUP_INTERVAL} 秒")

    def _stop_cleanup_task(self):
        """停止自动清理任务"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("[ConversationManager] 自动清理任务已停止")

    async def _save_async(self):
        """异步保存会话数据"""
        try:
            data = {
                "conversations": {
                    conv_id: asdict(context)
                    for conv_id, context in self.conversations.items()
                },
                "token_conversations": self.token_conversations
            }
            await storage_manager.save_json("conversations.json", data)
        except Exception as e:
            logger.error(f"[ConversationManager] 保存失败: {e}")

    async def shutdown(self):
        """关闭时保存数据"""
        self._stop_cleanup_task()
        await self._save_async()
        logger.info("[ConversationManager] 会话管理器已关闭")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_conversations": len(self.conversations),
            "tokens_with_conversations": len(self.token_conversations),
            "avg_messages_per_conversation": sum(c.message_count for c in self.conversations.values()) / len(self.conversations) if self.conversations else 0,
            "ttl_seconds": settings.conversation_ttl,
            "last_cleanup_time": self._last_cleanup_time,
            "total_cleaned": self._total_cleaned,
            "auto_cleanup_enabled": self._cleanup_task is not None
        }


# 全局会话管理器实例
conversation_manager = ConversationManager()
