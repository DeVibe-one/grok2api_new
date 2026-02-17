"""存储管理器"""

import json
import aiofiles
from pathlib import Path
from typing import Any, Optional, Dict
from app.core.logger import logger
from app.core.config import settings


class StorageManager:
    """存储管理器 - 支持 JSON 文件存储"""

    def __init__(self):
        self.storage_path = Path(settings.storage_path)
        self.initialized = False

    async def init(self):
        """初始化存储"""
        if self.initialized:
            return

        # 创建存储目录
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Storage] 存储目录: {self.storage_path}")

        self.initialized = True

    async def save_json(self, filename: str, data: Any):
        """保存 JSON 数据"""
        file_path = self.storage_path / filename
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"[Storage] 保存失败 {filename}: {e}")
            raise

    async def load_json(self, filename: str, default: Any = None) -> Any:
        """加载 JSON 数据"""
        file_path = self.storage_path / filename
        if not file_path.exists():
            return default

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"[Storage] 加载失败 {filename}: {e}")
            return default

    async def close(self):
        """关闭存储"""
        logger.info("[Storage] 存储管理器已关闭")


# 全局存储管理器实例
storage_manager = StorageManager()
