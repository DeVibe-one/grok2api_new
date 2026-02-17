"""请求统计服务 - 按小时/天统计请求趋势"""

import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from app.core.storage import storage_manager
from app.core.logger import logger


@dataclass
class HourlyStats:
    """小时统计"""
    hour: str  # 格式: "2024-01-15 14"
    total: int = 0
    success: int = 0
    failed: int = 0
    models: Dict[str, int] = field(default_factory=dict)


@dataclass
class DailyStats:
    """日统计"""
    date: str  # 格式: "2024-01-15"
    total: int = 0
    success: int = 0
    failed: int = 0
    models: Dict[str, int] = field(default_factory=dict)


class RequestStats:
    """请求统计管理器"""

    def __init__(self):
        self.hourly_stats: Dict[str, HourlyStats] = {}
        self.daily_stats: Dict[str, DailyStats] = {}
        self.initialized = False
        self._dirty = False

    async def init(self):
        """初始化"""
        if self.initialized:
            return

        data = await storage_manager.load_json("request_stats.json", {})

        # 恢复小时统计
        for hour, stats in data.get("hourly", {}).items():
            self.hourly_stats[hour] = HourlyStats(
                hour=hour,
                total=stats.get("total", 0),
                success=stats.get("success", 0),
                failed=stats.get("failed", 0),
                models=stats.get("models", {})
            )

        # 恢复日统计
        for date, stats in data.get("daily", {}).items():
            self.daily_stats[date] = DailyStats(
                date=date,
                total=stats.get("total", 0),
                success=stats.get("success", 0),
                failed=stats.get("failed", 0),
                models=stats.get("models", {})
            )

        # 清理30天前的数据
        await self._cleanup_old_data()

        self.initialized = True
        logger.info(f"[RequestStats] 已加载 {len(self.hourly_stats)} 小时统计, {len(self.daily_stats)} 日统计")

    async def record(self, model: str, success: bool):
        """记录一次请求"""
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d %H")
        date_key = now.strftime("%Y-%m-%d")

        # 更新小时统计
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = HourlyStats(hour=hour_key)

        hourly = self.hourly_stats[hour_key]
        hourly.total += 1
        if success:
            hourly.success += 1
        else:
            hourly.failed += 1
        hourly.models[model] = hourly.models.get(model, 0) + 1

        # 更新日统计
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = DailyStats(date=date_key)

        daily = self.daily_stats[date_key]
        daily.total += 1
        if success:
            daily.success += 1
        else:
            daily.failed += 1
        daily.models[model] = daily.models.get(model, 0) + 1

        self._dirty = True

    async def save(self):
        """保存统计数据"""
        if not self._dirty:
            return

        try:
            data = {
                "hourly": {k: asdict(v) for k, v in self.hourly_stats.items()},
                "daily": {k: asdict(v) for k, v in self.daily_stats.items()}
            }
            await storage_manager.save_json("request_stats.json", data)
            self._dirty = False
        except Exception as e:
            logger.error(f"[RequestStats] 保存失败: {e}")

    async def _cleanup_old_data(self):
        """清理30天前的数据"""
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        cutoff_hour = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H")

        # 清理旧的小时数据（保留7天）
        old_hours = [k for k in self.hourly_stats if k < cutoff_hour]
        for k in old_hours:
            del self.hourly_stats[k]

        # 清理旧的日数据（保留30天）
        old_days = [k for k in self.daily_stats if k < cutoff_date]
        for k in old_days:
            del self.daily_stats[k]

        if old_hours or old_days:
            self._dirty = True
            logger.info(f"[RequestStats] 清理了 {len(old_hours)} 小时 + {len(old_days)} 日 旧数据")

    def get_hourly_stats(self, hours: int = 24) -> List[dict]:
        """获取最近N小时的统计"""
        now = datetime.now()
        result = []

        for i in range(hours - 1, -1, -1):
            hour = (now - timedelta(hours=i)).strftime("%Y-%m-%d %H")
            if hour in self.hourly_stats:
                result.append(asdict(self.hourly_stats[hour]))
            else:
                result.append({
                    "hour": hour,
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "models": {}
                })

        return result

    def get_daily_stats(self, days: int = 7) -> List[dict]:
        """获取最近N天的统计"""
        now = datetime.now()
        result = []

        for i in range(days - 1, -1, -1):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            if date in self.daily_stats:
                result.append(asdict(self.daily_stats[date]))
            else:
                result.append({
                    "date": date,
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "models": {}
                })

        return result

    def get_summary(self) -> dict:
        """获取统计摘要"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_stats = self.daily_stats.get(today, DailyStats(date=today))

        # 计算总体统计
        total_requests = sum(d.total for d in self.daily_stats.values())
        total_success = sum(d.success for d in self.daily_stats.values())

        # 模型分布
        model_distribution = defaultdict(int)
        for daily in self.daily_stats.values():
            for model, count in daily.models.items():
                model_distribution[model] += count

        return {
            "today": {
                "total": today_stats.total,
                "success": today_stats.success,
                "failed": today_stats.failed,
                "success_rate": round(today_stats.success / today_stats.total * 100, 1) if today_stats.total > 0 else 0
            },
            "all_time": {
                "total": total_requests,
                "success": total_success,
                "failed": total_requests - total_success,
                "success_rate": round(total_success / total_requests * 100, 1) if total_requests > 0 else 0
            },
            "model_distribution": dict(model_distribution)
        }


# 全局实例
request_stats = RequestStats()
