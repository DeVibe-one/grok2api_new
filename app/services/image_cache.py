"""图片缓存服务 - 下载和缓存 Grok 生成的图片"""

import asyncio
import base64
from pathlib import Path
from typing import Optional
from curl_cffi.requests import AsyncSession

from app.core.logger import logger
from app.core.config import settings
from app.services.headers import get_dynamic_headers


# MIME 类型映射
MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
}
DEFAULT_MIME = 'image/jpeg'
ASSETS_URL = "https://assets.grok.com"


class ImageCache:
    """图片缓存服务"""

    def __init__(self):
        self.cache_dir = Path("data/temp/image")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = 30.0
        self._cleanup_lock = asyncio.Lock()

    def _get_path(self, file_path: str) -> Path:
        """转换文件路径为缓存路径"""
        return self.cache_dir / file_path.lstrip('/').replace('/', '-')

    def _build_headers(self, file_path: str, auth_token: str) -> dict:
        """构建请求头"""
        headers = get_dynamic_headers(pathname=file_path)
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://grok.com/",
            "Cookie": auth_token if auth_token.startswith("sso=") else f"sso={auth_token}"
        })
        return headers

    async def download(self, file_path: str, auth_token: str) -> Optional[Path]:
        """下载并缓存图片"""
        cache_path = self._get_path(file_path)
        if cache_path.exists():
            logger.debug(f"[ImageCache] 文件已缓存: {cache_path.name}")
            return cache_path

        MAX_RETRY = 3
        for retry in range(MAX_RETRY):
            try:
                proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

                async with AsyncSession(impersonate="chrome120") as session:
                    url = f"{ASSETS_URL}{file_path}"
                    logger.debug(f"[ImageCache] 下载: {url}")

                    response = await session.get(
                        url,
                        headers=self._build_headers(file_path, auth_token),
                        proxies=proxies,
                        timeout=self.timeout,
                        allow_redirects=True
                    )

                    if response.status_code == 200:
                        await asyncio.to_thread(cache_path.write_bytes, response.content)
                        logger.info(f"[ImageCache] 缓存成功: {cache_path.name}")
                        # 异步清理
                        asyncio.create_task(self._cleanup())
                        return cache_path
                    elif response.status_code in [401, 403, 429]:
                        logger.warning(f"[ImageCache] 下载失败 {response.status_code}，重试 {retry + 1}/{MAX_RETRY}")
                        await asyncio.sleep(0.5 * (retry + 1))
                    else:
                        logger.error(f"[ImageCache] 下载失败: {response.status_code}")
                        return None

            except Exception as e:
                logger.error(f"[ImageCache] 下载异常: {e}")
                if retry < MAX_RETRY - 1:
                    await asyncio.sleep(0.5)

        return None

    def get_cached(self, file_path: str) -> Optional[Path]:
        """获取已缓存的文件"""
        path = self._get_path(file_path)
        return path if path.exists() else None

    @staticmethod
    def to_base64(image_path: Path) -> Optional[str]:
        """图片转 base64"""
        try:
            if not image_path.exists():
                return None
            data = base64.b64encode(image_path.read_bytes()).decode()
            mime = MIME_TYPES.get(image_path.suffix.lower(), DEFAULT_MIME)
            return f"data:{mime};base64,{data}"
        except Exception as e:
            logger.error(f"[ImageCache] 转换失败: {e}")
            return None

    async def download_base64(self, file_path: str, auth_token: str) -> Optional[str]:
        """下载并转为 base64"""
        cache_path = await self.download(file_path, auth_token)
        if not cache_path:
            return None
        return self.to_base64(cache_path)

    async def _cleanup(self):
        """清理超限缓存"""
        if self._cleanup_lock.locked():
            return

        async with self._cleanup_lock:
            try:
                max_mb = settings.max_image_cache_mb
                max_bytes = max_mb * 1024 * 1024

                files = [(f, (s := f.stat()).st_size, s.st_mtime)
                         for f in self.cache_dir.glob("*") if f.is_file()]
                total = sum(size for _, size, _ in files)

                if total <= max_bytes:
                    return

                logger.info(f"[ImageCache] 清理缓存 {total / 1024 / 1024:.1f}MB -> {max_mb}MB")

                for path, size, _ in sorted(files, key=lambda x: x[2]):
                    if total <= max_bytes:
                        break
                    await asyncio.to_thread(path.unlink)
                    total -= size

            except Exception as e:
                logger.error(f"[ImageCache] 清理失败: {e}")

    def list_cached_images(self) -> list:
        """列出所有缓存的图片"""
        try:
            files = []
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                        "url": f"/images/{file_path.name}"
                    })
            # 按修改时间降序排序
            files.sort(key=lambda x: x["modified"], reverse=True)
            return files
        except Exception as e:
            logger.error(f"[ImageCache] 列出缓存失败: {e}")
            return []

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        try:
            files = list(self.cache_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            return {
                "count": len(files),
                "total_size": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "cache_dir": str(self.cache_dir)
            }
        except Exception as e:
            logger.error(f"[ImageCache] 获取统计失败: {e}")
            return {"count": 0, "total_size": 0, "total_size_mb": 0, "cache_dir": str(self.cache_dir)}

    async def delete_cached_image(self, filename: str) -> bool:
        """删除指定的缓存图片"""
        try:
            file_path = self.cache_dir / filename
            if file_path.exists() and file_path.is_file():
                await asyncio.to_thread(file_path.unlink)
                logger.info(f"[ImageCache] 删除缓存: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"[ImageCache] 删除缓存失败: {e}")
            return False

    async def clear_all_cache(self) -> int:
        """清空所有缓存"""
        try:
            files = list(self.cache_dir.glob("*"))
            count = 0
            for file_path in files:
                if file_path.is_file():
                    await asyncio.to_thread(file_path.unlink)
                    count += 1
            logger.info(f"[ImageCache] 清空缓存: {count} 个文件")
            return count
        except Exception as e:
            logger.error(f"[ImageCache] 清空缓存失败: {e}")
            return 0


# 全局实例
image_cache = ImageCache()
