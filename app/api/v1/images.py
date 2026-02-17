"""图片代理 API - 提供缓存的图片文件访问"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.logger import logger
from app.services.image_cache import image_cache


router = APIRouter()


@router.get("/images/{img_path:path}")
async def get_image(img_path: str):
    """获取缓存的图片

    Args:
        img_path: 文件路径（格式：users-xxx-generated-xxx-image.jpg）
    """
    try:
        # 转换路径（短横线→斜杠）
        original_path = "/" + img_path.replace('-', '/')

        # 获取缓存文件
        cache_path = image_cache.get_cached(original_path)

        if cache_path and cache_path.exists():
            # 根据扩展名判断 MIME 类型
            ext = cache_path.suffix.lower()
            media_types = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_types.get(ext, 'image/jpeg')

            logger.debug(f"[ImageAPI] 返回缓存: {cache_path.name}")
            return FileResponse(
                path=str(cache_path),
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Access-Control-Allow-Origin": "*"
                }
            )

        logger.warning(f"[ImageAPI] 未找到: {original_path}")
        raise HTTPException(status_code=404, detail="Image not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ImageAPI] 获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
