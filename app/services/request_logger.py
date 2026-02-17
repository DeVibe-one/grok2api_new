"""请求日志服务 - 记录每个请求的详情"""

import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

from app.core.storage import storage_manager
from app.core.logger import logger


@dataclass
class RequestLog:
    """请求日志条目"""
    id: str
    timestamp: float
    model: str
    token_preview: str  # Token预览（前8后4）
    api_key_preview: str  # API Key预览
    status: str  # success / failed
    error: Optional[str]
    duration_ms: int  # 耗时（毫秒）
    ip: Optional[str]
    stream: bool


class RequestLogger:
    """请求日志管理器"""

    MAX_LOGS = 1000  # 最多保留1000条

    def __init__(self):
        self.logs: List[RequestLog] = []
        self.initialized = False
        self._counter = 0

    async def init(self):
        """初始化"""
        if self.initialized:
            return

        data = await storage_manager.load_json("request_logs.json", {"logs": []})

        for log_data in data.get("logs", []):
            try:
                self.logs.append(RequestLog(**log_data))
            except Exception:
                pass

        # 只保留最新的
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[-self.MAX_LOGS:]

        self.initialized = True
        logger.info(f"[RequestLogger] 已加载 {len(self.logs)} 条日志")

    async def log(
        self,
        model: str,
        token: str,
        api_key: Optional[str],
        success: bool,
        error: Optional[str],
        duration_ms: int,
        ip: Optional[str],
        stream: bool
    ):
        """记录请求"""
        self._counter += 1
        log_id = f"{int(time.time() * 1000)}-{self._counter}"

        # Token预览
        token_preview = ""
        if token:
            if len(token) > 12:
                token_preview = token[:8] + "..." + token[-4:]
            else:
                token_preview = token

        # API Key预览
        api_key_preview = ""
        if api_key:
            if len(api_key) > 12:
                api_key_preview = api_key[:8] + "..." + api_key[-4:]
            else:
                api_key_preview = api_key

        log_entry = RequestLog(
            id=log_id,
            timestamp=time.time(),
            model=model,
            token_preview=token_preview,
            api_key_preview=api_key_preview,
            status="success" if success else "failed",
            error=error,
            duration_ms=duration_ms,
            ip=ip,
            stream=stream
        )

        self.logs.append(log_entry)

        # 超过最大数量时删除旧的
        if len(self.logs) > self.MAX_LOGS:
            self.logs = self.logs[-self.MAX_LOGS:]

    async def save(self):
        """保存日志"""
        try:
            data = {"logs": [asdict(log) for log in self.logs]}
            await storage_manager.save_json("request_logs.json", data)
        except Exception as e:
            logger.error(f"[RequestLogger] 保存失败: {e}")

    def get_logs(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """获取日志（倒序）"""
        sorted_logs = sorted(self.logs, key=lambda x: x.timestamp, reverse=True)
        return [asdict(log) for log in sorted_logs[offset:offset + limit]]

    def get_total(self) -> int:
        """获取总数"""
        return len(self.logs)

    async def clear(self):
        """清空日志"""
        self.logs.clear()
        await self.save()
        logger.info("[RequestLogger] 日志已清空")


# 全局实例
request_logger = RequestLogger()
