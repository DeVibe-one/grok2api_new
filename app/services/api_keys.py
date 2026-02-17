"""API Key管理服务 - 多用户密钥管理"""

import secrets
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from app.core.storage import storage_manager
from app.core.logger import logger


@dataclass
class ApiKeyInfo:
    """API Key信息"""
    key: str
    name: str
    enabled: bool
    created_at: float
    last_used: float
    request_count: int


class ApiKeyManager:
    """API Key管理器"""

    def __init__(self):
        self.keys: Dict[str, ApiKeyInfo] = {}
        self.initialized = False

    async def init(self):
        """初始化"""
        if self.initialized:
            return

        data = await storage_manager.load_json("api_keys.json", {"keys": {}})

        for key, info in data.get("keys", {}).items():
            self.keys[key] = ApiKeyInfo(
                key=key,
                name=info.get("name", ""),
                enabled=info.get("enabled", True),
                created_at=info.get("created_at", time.time()),
                last_used=info.get("last_used", 0),
                request_count=info.get("request_count", 0)
            )

        self.initialized = True
        logger.info(f"[ApiKeyManager] 已加载 {len(self.keys)} 个API Key")

    async def _save(self):
        """保存数据"""
        try:
            data = {"keys": {k: asdict(v) for k, v in self.keys.items()}}
            await storage_manager.save_json("api_keys.json", data)
        except Exception as e:
            logger.error(f"[ApiKeyManager] 保存失败: {e}")

    def generate_key(self) -> str:
        """生成新的API Key"""
        return f"sk-{secrets.token_urlsafe(32)}"

    async def create_key(self, name: str = "") -> ApiKeyInfo:
        """创建新的API Key"""
        key = self.generate_key()
        info = ApiKeyInfo(
            key=key,
            name=name,
            enabled=True,
            created_at=time.time(),
            last_used=0,
            request_count=0
        )
        self.keys[key] = info
        await self._save()
        logger.info(f"[ApiKeyManager] 创建API Key: {key[:16]}...")
        return info

    async def create_keys_batch(self, count: int, prefix: str = "") -> List[ApiKeyInfo]:
        """批量创建API Key"""
        created = []
        for i in range(count):
            name = f"{prefix}{i + 1}" if prefix else ""
            info = await self.create_key(name)
            created.append(info)
        return created

    async def delete_key(self, key: str) -> bool:
        """删除API Key"""
        if key in self.keys:
            del self.keys[key]
            await self._save()
            logger.info(f"[ApiKeyManager] 删除API Key: {key[:16]}...")
            return True
        return False

    async def delete_keys_batch(self, keys: List[str]) -> int:
        """批量删除API Key"""
        deleted = 0
        for key in keys:
            if key in self.keys:
                del self.keys[key]
                deleted += 1
        if deleted > 0:
            await self._save()
            logger.info(f"[ApiKeyManager] 批量删除 {deleted} 个API Key")
        return deleted

    async def update_key(
        self,
        key: str,
        name: Optional[str] = None,
        enabled: Optional[bool] = None
    ) -> bool:
        """更新API Key"""
        if key not in self.keys:
            return False

        info = self.keys[key]
        if name is not None:
            info.name = name
        if enabled is not None:
            info.enabled = enabled

        await self._save()
        return True

    def validate_key(self, key: str) -> Optional[ApiKeyInfo]:
        """验证API Key"""
        if not key:
            return None

        info = self.keys.get(key)
        if info and info.enabled:
            return info
        return None

    async def record_usage(self, key: str):
        """记录使用"""
        if key in self.keys:
            self.keys[key].last_used = time.time()
            self.keys[key].request_count += 1

    def list_keys(self) -> List[ApiKeyInfo]:
        """列出所有API Key"""
        return list(self.keys.values())

    def get_stats(self) -> dict:
        """获取统计"""
        total = len(self.keys)
        enabled = sum(1 for k in self.keys.values() if k.enabled)
        total_requests = sum(k.request_count for k in self.keys.values())
        return {
            "total_keys": total,
            "enabled_keys": enabled,
            "disabled_keys": total - enabled,
            "total_requests": total_requests
        }


# 全局实例
api_key_manager = ApiKeyManager()
