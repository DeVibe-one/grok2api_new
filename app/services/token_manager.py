"""Token 管理器 - 支持轮询和智能冷却"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from app.core.logger import logger
from app.core.storage import storage_manager


@dataclass
class TokenInfo:
    """Token 信息"""
    token: str
    name: str
    enabled: bool
    created_at: float
    last_used: float
    request_count: int
    failure_count: int
    # 冷却相关
    cooldown_until: float = 0  # 冷却结束时间戳
    cooldown_reason: str = ""  # 冷却原因
    consecutive_failures: int = 0  # 连续失败次数
    # 额度相关
    remaining_queries: int = -1  # 剩余Chat查询次数（-1表示未知）
    last_check: float = 0  # 最后检查时间
    last_failure_reason: str = ""  # 最后失败原因
    # 账号类型
    account_type: str = "unknown"  # unknown / free / super


# 冷却时间配置（秒）
COOLDOWN_NORMAL_ERROR = 3600  # 普通错误连续失败5次：1小时
COOLDOWN_429_WITH_QUOTA = 18000  # 429限流+有额度：5小时
COOLDOWN_429_NO_QUOTA = 36000  # 429限流+无额度：10小时
COOLDOWN_CONSECUTIVE_FAILURES = 5  # 连续失败N次后触发冷却


class TokenManager:
    """Token 管理器"""

    def __init__(self):
        self.tokens: Dict[str, TokenInfo] = {}
        self.initialized = False
        # 轮询索引
        self._round_robin_index = 0
        # 刷新进度
        self._refresh_in_progress = False
        self._refresh_total = 0
        self._refresh_completed = 0
        self._refresh_results: List[dict] = []
        # 后台额度检查：正在检查的 token 集合（防重入）
        self._pending_quota_checks: Set[str] = set()

    async def init(self):
        """初始化"""
        if self.initialized:
            return

        # 从存储加载 token
        data = await storage_manager.load_json("tokens.json", {})

        for token_str, token_data in data.items():
            # 兼容旧数据
            if "cooldown_until" not in token_data:
                token_data["cooldown_until"] = 0
            if "cooldown_reason" not in token_data:
                token_data["cooldown_reason"] = ""
            if "consecutive_failures" not in token_data:
                token_data["consecutive_failures"] = 0
            if "remaining_queries" not in token_data:
                token_data["remaining_queries"] = -1
            if "last_check" not in token_data:
                token_data["last_check"] = 0
            if "last_failure_reason" not in token_data:
                token_data["last_failure_reason"] = ""
            if "account_type" not in token_data:
                token_data["account_type"] = "unknown"
            # 移除已废弃的字段
            token_data.pop("remaining_image_queries", None)

            self.tokens[token_str] = TokenInfo(**token_data)

        logger.info(f"[TokenManager] 已加载 {len(self.tokens)} 个 Token")
        self.initialized = True

    @staticmethod
    def _normalize_token(token: str) -> str:
        """标准化 Token：去除空白和 sso= 前缀"""
        token = (token or "").strip()
        # 去除 sso= 前缀，统一存储裸 token
        if token.startswith("sso="):
            token = token[4:]
        return token.strip()

    async def add_token(self, token: str, name: str = ""):
        """添加 Token"""
        token = self._normalize_token(token)
        if not token:
            logger.warning("[TokenManager] Token 为空，已忽略")
            return

        if token in self.tokens:
            logger.warning(f"[TokenManager] Token 已存在: {name}")
            return

        self.tokens[token] = TokenInfo(
            token=token,
            name=name or f"token-{len(self.tokens) + 1}",
            enabled=True,
            created_at=time.time(),
            last_used=0,
            request_count=0,
            failure_count=0
        )

        logger.info(f"[TokenManager] 添加 Token: {name}")
        await self._save()

    async def add_tokens_batch(self, tokens: List[str], name: str = "", enabled: bool = True) -> dict:
        """批量添加 Token（去重 + 单次保存）

        Args:
            tokens: Token 字符串列表
            name: 名称前缀（批量时自动编号）
            enabled: 是否启用

        Returns:
            {"added": int, "duplicates": int, "empty": int}
        """
        now = time.time()
        added = 0
        duplicates = 0
        empty = 0
        seen_in_batch: Set[str] = set()  # 批次内去重

        for raw_token in tokens:
            normalized = self._normalize_token(raw_token)
            if not normalized:
                empty += 1
                continue

            # 批次内去重 + 已有去重
            if normalized in seen_in_batch or normalized in self.tokens:
                duplicates += 1
                continue

            seen_in_batch.add(normalized)

            token_name = name
            if not token_name:
                token_name = f"token-{len(self.tokens) + 1}"
            elif len(tokens) > 1:
                token_name = f"{name}-{added + 1}"

            self.tokens[normalized] = TokenInfo(
                token=normalized,
                name=token_name,
                enabled=enabled,
                created_at=now,
                last_used=0,
                request_count=0,
                failure_count=0
            )
            added += 1

        if added > 0:
            await self._save()
            logger.info(f"[TokenManager] 批量添加 {added} 个 Token（重复 {duplicates}，空 {empty}）")

        return {"added": added, "duplicates": duplicates, "empty": empty}

    def list_tokens(self) -> List[TokenInfo]:
        """列出所有 Token（按创建时间排序）"""
        return sorted(self.tokens.values(), key=lambda t: t.created_at)

    async def update_token(self, token: str, *, name: Optional[str] = None, enabled: Optional[bool] = None) -> bool:
        """更新 Token 元数据（名称/启用状态）"""
        token = (token or "").strip()
        if not token:
            return False

        info = self.tokens.get(token)
        if not info:
            return False

        if name is not None:
            info.name = name
        if enabled is not None:
            info.enabled = enabled

        await self._save()
        return True

    async def delete_token(self, token: str) -> bool:
        """删除 Token"""
        token = (token or "").strip()
        if not token:
            return False

        removed = self.tokens.pop(token, None)
        if not removed:
            return False

        logger.info(f"[TokenManager] 已删除 Token: {removed.name}")
        await self._save()
        return True

    def is_in_cooldown(self, token: str) -> bool:
        """检查 Token 是否在冷却中"""
        info = self.tokens.get(token)
        if not info:
            return False
        return info.cooldown_until > time.time()

    def get_cooldown_remaining(self, token: str) -> int:
        """获取冷却剩余时间（秒）"""
        info = self.tokens.get(token)
        if not info:
            return 0
        remaining = info.cooldown_until - time.time()
        return max(0, int(remaining))

    async def get_token(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """获取可用的 Token（轮询策略，跳过冷却中的和排除的）

        Args:
            exclude: 需要排除的 Token 集合（用于重试时避免重复使用同一 Token）
        """
        now = time.time()
        exclude = exclude or set()

        # 获取所有可用的 Token（启用、未冷却、未排除）
        available_tokens = [
            info for info in self.tokens.values()
            if info.enabled and info.cooldown_until <= now and info.token not in exclude
        ]

        if not available_tokens:
            # 检查是否所有 Token 都在冷却或被排除
            all_enabled = [info for info in self.tokens.values() if info.enabled and info.token not in exclude]
            if all_enabled:
                # 找出最快解除冷却的 Token
                soonest = min(all_enabled, key=lambda t: t.cooldown_until)
                wait_time = int(soonest.cooldown_until - now)
                logger.warning(f"[TokenManager] 所有 Token 都在冷却中，最快 {wait_time}秒 后解除")
            elif exclude:
                logger.warning(f"[TokenManager] 没有更多可用的 Token（已排除 {len(exclude)} 个）")
            else:
                logger.error("[TokenManager] 没有可用的 Token")
            return None

        # 按创建时间排序，保证轮询顺序稳定
        sorted_tokens = sorted(available_tokens, key=lambda t: t.created_at)

        # 轮询选择
        self._round_robin_index = self._round_robin_index % len(sorted_tokens)
        token_info = sorted_tokens[self._round_robin_index]
        self._round_robin_index += 1

        # 更新使用信息
        token_info.last_used = now
        token_info.request_count += 1

        logger.info(f"[TokenManager] 轮询选择 Token: {token_info.name}")
        return token_info.token

    async def record_success(self, token: str):
        """记录成功"""
        if token in self.tokens:
            info = self.tokens[token]
            info.consecutive_failures = 0  # 重置连续失败计数
            # 后台异步刷新额度（不阻塞响应）
            asyncio.create_task(self._refresh_token_quota_bg(token))

    async def record_failure(self, token: str, error_type: str = "normal", has_quota: bool = True):
        """记录失败并可能触发冷却

        Args:
            token: Token 字符串
            error_type: 错误类型 - "normal" | "429" | "auth"
            has_quota: 是否还有额度（仅 429 时有效）
        """
        if token not in self.tokens:
            return

        info = self.tokens[token]
        info.failure_count += 1
        info.consecutive_failures += 1
        info.last_failure_reason = error_type

        now = time.time()

        # 根据错误类型设置冷却
        if error_type == "429":
            if has_quota:
                # 429 + 有额度：5小时
                info.cooldown_until = now + COOLDOWN_429_WITH_QUOTA
                info.cooldown_reason = "429限流（有额度）"
                logger.warning(f"[TokenManager] Token {info.name} 触发 429，冷却5小时")
            else:
                # 429 + 无额度：10小时
                info.cooldown_until = now + COOLDOWN_429_NO_QUOTA
                info.cooldown_reason = "429限流（无额度）"
                info.remaining_queries = 0
                logger.warning(f"[TokenManager] Token {info.name} 触发 429（无额度），冷却10小时")
        elif error_type == "auth":
            # 认证失败，禁用 Token
            info.enabled = False
            info.cooldown_reason = "认证失败"
            logger.warning(f"[TokenManager] Token {info.name} 认证失败，已禁用")
        elif info.consecutive_failures >= COOLDOWN_CONSECUTIVE_FAILURES:
            # 连续失败5次：1小时
            info.cooldown_until = now + COOLDOWN_NORMAL_ERROR
            info.cooldown_reason = f"连续失败{info.consecutive_failures}次"
            logger.warning(f"[TokenManager] Token {info.name} 连续失败{info.consecutive_failures}次，冷却1小时")

        await self._save()

    async def clear_cooldown(self, token: str):
        """清除冷却"""
        if token in self.tokens:
            self.tokens[token].cooldown_until = 0
            self.tokens[token].cooldown_reason = ""
            self.tokens[token].consecutive_failures = 0
            await self._save()

    async def _refresh_token_quota_bg(self, token: str):
        """后台刷新单个 Token 的额度（不阻塞主线程）"""
        # 防重入：同一 token 不并发检查
        if token in self._pending_quota_checks:
            return

        info = self.tokens.get(token)
        if not info or not info.enabled:
            return

        self._pending_quota_checks.add(token)
        try:
            result = await self._check_rate_limits(token)
            if result["success"]:
                new_remaining = result.get("remaining_queries", -1)
                old_remaining = info.remaining_queries
                info.remaining_queries = new_remaining
                info.last_check = time.time()
                if old_remaining != new_remaining:
                    logger.info(f"[TokenManager] 后台额度更新: {info.name} {old_remaining} -> {new_remaining}")
                    await self._save()
        except Exception as e:
            logger.debug(f"[TokenManager] 后台额度检查失败: {e}")
        finally:
            self._pending_quota_checks.discard(token)

    async def update_remaining_queries(self, token: str, remaining: int):
        """更新剩余查询次数"""
        if token in self.tokens:
            self.tokens[token].remaining_queries = remaining
            self.tokens[token].last_check = time.time()

    async def test_token(self, token: str) -> dict:
        """测试 Token 可用性并获取剩余额度"""
        if token not in self.tokens:
            return {"success": False, "error": "Token 不存在"}

        info = self.tokens[token]

        # 调用 Grok rate-limits API 检测
        try:
            result = await self._check_rate_limits(token)
            account_type = await self._check_subscription(token)
            info.account_type = account_type

            if result["success"]:
                # 更新额度信息
                info.remaining_queries = result.get("remaining_queries", -1)
                info.last_check = time.time()
                await self._save()

            return {
                "success": result["success"],
                "token": token[:12] + "...",
                "name": info.name,
                "enabled": info.enabled,
                "in_cooldown": self.is_in_cooldown(token),
                "cooldown_remaining": self.get_cooldown_remaining(token),
                "remaining_queries": result.get("remaining_queries", info.remaining_queries),
                "account_type": account_type,
                "error": result.get("error")
            }
        except Exception as e:
            logger.error(f"[TokenManager] 测试 Token 失败: {e}")
            return {
                "success": False,
                "token": token[:12] + "...",
                "name": info.name,
                "error": str(e)
            }

    async def _check_rate_limits(self, token: str) -> dict:
        """调用 Grok API 检测额度"""
        from curl_cffi.requests import AsyncSession
        from app.core.config import settings
        from app.services.headers import get_dynamic_headers

        RATE_LIMIT_API = "https://grok.com/rest/rate-limits"

        try:
            headers = get_dynamic_headers("/rest/rate-limits")

            # 确保 Token 格式正确
            if not token.startswith("sso="):
                cookie = f"sso={token}"
            else:
                cookie = token
            headers["Cookie"] = cookie

            proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

            async with AsyncSession(impersonate="chrome120") as session:
                # 检测 Chat 额度（基础模型 grok-3）
                payload = {"requestKind": "DEFAULT", "modelName": "grok-3"}
                response = await session.post(
                    RATE_LIMIT_API,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    proxies=proxies
                )

                if response.status_code == 200:
                    data = response.json()
                    chat_remaining = data.get("remainingTokens", -1)
                    logger.info(f"[TokenManager] Chat 额度: {chat_remaining}")
                    return {
                        "success": True,
                        "remaining_queries": chat_remaining
                    }
                elif response.status_code == 401:
                    logger.warning(f"[TokenManager] Token 无效: 401")
                    return {"success": False, "error": "Token 无效或已过期", "remaining_queries": 0}
                elif response.status_code == 429:
                    logger.warning(f"[TokenManager] 请求限流: 429")
                    return {"success": False, "error": "请求过于频繁", "remaining_queries": -1}
                else:
                    logger.warning(f"[TokenManager] 检测失败: {response.status_code}")
                    return {"success": False, "error": f"HTTP {response.status_code}", "remaining_queries": -1}

        except Exception as e:
            logger.error(f"[TokenManager] 检测请求异常: {e}")
            return {"success": False, "error": str(e), "remaining_queries": -1}

    async def _check_subscription(self, token: str) -> str:
        """检测账号类型（普通/会员）"""
        from curl_cffi.requests import AsyncSession
        from app.core.config import settings
        from app.services.headers import get_dynamic_headers

        SUBSCRIPTION_API = "https://grok.com/rest/subscriptions"

        try:
            headers = get_dynamic_headers("/rest/subscriptions")

            if not token.startswith("sso="):
                cookie = f"sso={token}"
            else:
                cookie = token
            headers["Cookie"] = cookie

            proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

            async with AsyncSession(impersonate="chrome120") as session:
                response = await session.get(
                    SUBSCRIPTION_API,
                    headers=headers,
                    timeout=30,
                    proxies=proxies
                )

                if response.status_code == 200:
                    data = response.json()
                    subscriptions = data.get("subscriptions", [])
                    if subscriptions:
                        # 有订阅记录，检查是否有活跃订阅
                        for sub in subscriptions:
                            if sub.get("status") == "SUBSCRIPTION_STATUS_ACTIVE":
                                tier = sub.get("tier", "")
                                logger.info(f"[TokenManager] 账号类型: Super ({tier})")
                                return "super"
                        # 有订阅但都不活跃
                        return "free"
                    else:
                        logger.info(f"[TokenManager] 账号类型: Free")
                        return "free"
                else:
                    logger.warning(f"[TokenManager] 订阅检测失败: {response.status_code}")
                    return "unknown"

        except Exception as e:
            logger.error(f"[TokenManager] 订阅检测异常: {e}")
            return "unknown"

    def get_refresh_progress(self) -> dict:
        """获取刷新进度"""
        return {
            "in_progress": self._refresh_in_progress,
            "total": self._refresh_total,
            "completed": self._refresh_completed,
            "results": self._refresh_results[-10:]  # 最近10条结果
        }

    async def refresh_all_tokens(self) -> dict:
        """刷新所有 Token 的额度状态（并发）"""
        if self._refresh_in_progress:
            return {"success": False, "error": "刷新任务正在进行中"}

        self._refresh_in_progress = True
        self._refresh_total = len(self.tokens)
        self._refresh_completed = 0
        self._refresh_results = []

        try:
            # 先处理禁用的 token
            enabled_items = []
            for token_str, info in self.tokens.items():
                if not info.enabled:
                    self._refresh_completed += 1
                    self._refresh_results.append({
                        "token": token_str[:12] + "...",
                        "name": info.name,
                        "status": "跳过",
                        "reason": "已禁用"
                    })
                else:
                    enabled_items.append((token_str, info))

            # 并发检测所有启用的 token
            sem = asyncio.Semaphore(5)

            async def check_one(token_str: str, info):
                async with sem:
                    try:
                        result, account_type = await asyncio.gather(
                            self._check_rate_limits(token_str),
                            self._check_subscription(token_str)
                        )
                        remaining = result.get("remaining_queries", -1)
                        info.account_type = account_type

                        if result["success"]:
                            info.remaining_queries = remaining
                            info.last_check = time.time()

                        self._refresh_results.append({
                            "token": token_str[:12] + "...",
                            "name": info.name,
                            "status": "成功" if result["success"] else "失败",
                            "remaining": remaining,
                            "account_type": account_type,
                            "error": result.get("error")
                        })
                    except Exception as e:
                        self._refresh_results.append({
                            "token": token_str[:12] + "...",
                            "name": info.name,
                            "status": "错误",
                            "error": str(e)
                        })
                    self._refresh_completed += 1

            await asyncio.gather(*[check_one(t, i) for t, i in enabled_items])

            await self._save()
            return {
                "success": True,
                "total": self._refresh_total,
                "completed": self._refresh_completed,
                "results": self._refresh_results
            }

        finally:
            self._refresh_in_progress = False

    async def disable_token(self, token: str):
        """禁用 Token"""
        if token in self.tokens:
            self.tokens[token].enabled = False
            logger.warning(f"[TokenManager] Token 已禁用: {self.tokens[token].name}")
            await self._save()

    async def _save(self):
        """保存 Token 数据"""
        try:
            data = {
                token_str: asdict(info)
                for token_str, info in self.tokens.items()
            }
            await storage_manager.save_json("tokens.json", data)
        except Exception as e:
            logger.error(f"[TokenManager] 保存失败: {e}")

    async def shutdown(self):
        """关闭时保存"""
        await self._save()
        logger.info("[TokenManager] Token 管理器已关闭")

    def get_stats(self) -> dict:
        """获取统计信息"""
        now = time.time()
        in_cooldown = sum(1 for t in self.tokens.values() if t.cooldown_until > now)

        # 计算失效 token 数（已禁用的）
        expired = sum(1 for t in self.tokens.values() if not t.enabled)

        # 计算 Chat 剩余总量（所有已启用 token 的 remaining_queries 之和）
        chat_remaining = 0
        for t in self.tokens.values():
            if t.enabled and t.remaining_queries > 0:
                chat_remaining += t.remaining_queries

        return {
            "total_tokens": len(self.tokens),
            "enabled_tokens": sum(1 for t in self.tokens.values() if t.enabled),
            "expired_tokens": expired,
            "in_cooldown": in_cooldown,
            "chat_remaining": chat_remaining,
            "total_requests": sum(t.request_count for t in self.tokens.values()),
            "total_failures": sum(t.failure_count for t in self.tokens.values())
        }


# 全局 token 管理器实例
token_manager = TokenManager()
