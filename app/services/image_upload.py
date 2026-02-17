"""图片上传管理器"""

import asyncio
import base64
import re
from typing import Tuple, Optional
from urllib.parse import urlparse
from curl_cffi.requests import AsyncSession

from app.core.logger import logger
from app.core.config import settings
from app.services.headers import get_dynamic_headers


# 常量
UPLOAD_API = "https://grok.com/rest/app-chat/upload-file"
TIMEOUT = 30
BROWSER = "chrome120"

# MIME类型
MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
}
DEFAULT_MIME = "image/jpeg"
DEFAULT_EXT = "jpg"


class ImageUploadManager:
    """图片上传管理器"""

    @staticmethod
    async def upload(image_input: str, auth_token: str) -> Tuple[str, str]:
        """上传图片（支持Base64或URL）

        Args:
            image_input: 图片输入（Base64字符串或URL）
            auth_token: 认证token

        Returns:
            (file_id, file_uri) 元组
        """
        try:
            # 判断类型并处理
            if ImageUploadManager._is_url(image_input):
                buffer, mime = await ImageUploadManager._download(image_input)
                filename, _ = ImageUploadManager._get_info("", mime)
            else:
                buffer = image_input.split(",")[1] if "data:image" in image_input else image_input
                filename, mime = ImageUploadManager._get_info(image_input)

            # 构建数据
            data = {
                "fileName": filename,
                "fileMimeType": mime,
                "content": buffer,
            }

            if not auth_token:
                logger.error("[Upload] 认证令牌缺失")
                return "", ""

            # 请求配置
            headers = get_dynamic_headers("/rest/app-chat/upload-file")

            # 添加 Cookie（确保 token 包含 sso= 前缀）
            if not auth_token.startswith("sso="):
                auth_token = f"sso={auth_token}"
            headers["Cookie"] = auth_token

            proxies = {"http": settings.proxy_url, "https": settings.proxy_url} if settings.proxy_url else None

            # 上传
            async with AsyncSession(impersonate=BROWSER) as session:
                response = await session.post(
                    UPLOAD_API,
                    headers=headers,
                    json=data,
                    timeout=TIMEOUT,
                    proxies=proxies,
                )

                if response.status_code == 200:
                    result = response.json()
                    file_id = result.get("fileMetadataId", "")
                    file_uri = result.get("fileUri", "")
                    logger.info(f"[Upload] 上传成功，ID: {file_id}")
                    return file_id, file_uri
                else:
                    logger.error(f"[Upload] 上传失败，状态码: {response.status_code}")
                    return "", ""

        except Exception as e:
            logger.error(f"[Upload] 上传异常: {e}")
            return "", ""

    @staticmethod
    def _is_url(input_str: str) -> bool:
        """检查是否为URL"""
        try:
            result = urlparse(input_str)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False

    @staticmethod
    async def _download(url: str) -> Tuple[str, str]:
        """下载图片并转Base64

        Returns:
            (base64_string, mime_type) 元组
        """
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=10)
                if response.status_code != 200:
                    logger.error(f"[Upload] 下载失败: {response.status_code}")
                    return "", ""

                content_type = response.headers.get('content-type', DEFAULT_MIME)
                if not content_type.startswith('image/'):
                    content_type = DEFAULT_MIME

                b64 = base64.b64encode(response.content).decode()
                logger.info(f"[Upload] 图片下载成功: {url[:50]}...")
                return b64, content_type
        except Exception as e:
            logger.error(f"[Upload] 下载异常: {e}")
            return "", ""

    @staticmethod
    def _get_info(image_data: str, mime_type: Optional[str] = None) -> Tuple[str, str]:
        """获取文件名和MIME类型

        Returns:
            (file_name, mime_type) 元组
        """
        # 已提供MIME类型
        if mime_type:
            ext = mime_type.split("/")[1] if "/" in mime_type else DEFAULT_EXT
            return f"image.{ext}", mime_type

        # 从Base64提取
        mime = DEFAULT_MIME
        ext = DEFAULT_EXT

        if "data:image" in image_data:
            if match := re.search(r"data:([a-zA-Z0-9]+/[a-zA-Z0-9-.+]+);base64,", image_data):
                mime = match.group(1)
                ext = mime.split("/")[1]

        return f"image.{ext}", mime
