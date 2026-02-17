"""配置管理 - 从本地 config.json 读取，支持运行时修改"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


# 配置文件路径
CONFIG_DIR = Path("data")
CONFIG_FILE = CONFIG_DIR / "config.json"

# 默认配置
DEFAULTS = {
    "app_name": "Grok2API",
    "app_version": "2.0.0",
    "debug": False,
    "log_level": "INFO",

    "api_key": None,

    "admin_username": "admin",
    "admin_password": "admin",

    "grok_base_url": "https://grok.com",
    "grok_api_endpoint": "https://grok.com/rest/app-chat",

    "proxy_url": None,
    "base_url": "",

    "request_timeout": 120,
    "stream_timeout": 600,

    "storage_type": "json",
    "storage_path": "data",

    "conversation_ttl": 72000,
    "max_conversations_per_token": 100,

    "token_file": "data/tokens.json",

    "max_log_entries": 1000,
    "max_image_cache_mb": 500,
}


class Settings:
    """应用配置 - 从 data/config.json 加载"""

    def __init__(self):
        # 先设置默认值
        for key, value in DEFAULTS.items():
            setattr(self, key, value)

        # 从文件加载
        self._load()

    def _load(self):
        """从 config.json 加载配置"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if key in DEFAULTS:
                        setattr(self, key, value)
            except Exception:
                pass  # 文件损坏时用默认值
        else:
            # 首次运行，生成默认配置文件
            self._save()

    def _save(self):
        """保存当前配置到文件"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {key: getattr(self, key) for key in DEFAULTS}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 全局配置实例
settings = Settings()


class RuntimeConfig:
    """运行时配置管理器 - 支持动态修改并持久化到 config.json"""

    # 可以在运行时修改的配置项（分组）
    EDITABLE_KEYS = {
        # 后台账号配置
        "admin_username": {"type": "string", "label": "管理员账号", "desc": "后台登录用户名", "group": "auth"},
        "admin_password": {"type": "password", "label": "管理员密码", "desc": "后台登录密码", "group": "auth"},

        # 网络配置
        "proxy_url": {"type": "string", "label": "代理地址", "desc": "HTTP 代理服务器地址，如 http://127.0.0.1:7890", "group": "network"},
        "request_timeout": {"type": "int", "label": "请求超时", "desc": "普通请求超时时间（秒）", "group": "network"},
        "stream_timeout": {"type": "int", "label": "流式超时", "desc": "流式请求总超时时间（秒）", "group": "network"},

        # 图片配置
        "base_url": {"type": "string", "label": "图片服务地址", "desc": "图片缓存服务的外部访问地址，如 http://your-server:8000，留空则使用相对路径", "group": "image"},

        # 会话配置
        "conversation_ttl": {"type": "int", "label": "会话保留时间", "desc": "会话数据保留时间（秒），默认72000（20小时）", "group": "conversation"},
        "max_conversations_per_token": {"type": "int", "label": "最大会话数", "desc": "每个 Token 最多保留的会话数", "group": "conversation"},

        # 系统配置
        "log_level": {"type": "select", "label": "日志级别", "desc": "日志输出级别", "options": ["DEBUG", "INFO", "WARNING", "ERROR"], "group": "system"},
        "debug": {"type": "bool", "label": "调试模式", "desc": "启用调试模式", "group": "system"},
        "max_log_entries": {"type": "int", "label": "日志上限", "desc": "最多保留的请求日志条数，超出自动清理旧日志", "group": "system"},
        "max_image_cache_mb": {"type": "int", "label": "图片缓存上限(MB)", "desc": "图片缓存最大占用空间，超出自动清理最旧的图片", "group": "image"},
    }

    # 配置分组
    GROUPS = {
        "auth": {"label": "后台认证", "order": 1},
        "network": {"label": "网络设置", "order": 2},
        "image": {"label": "图片设置", "order": 3},
        "conversation": {"label": "会话管理", "order": 4},
        "system": {"label": "系统设置", "order": 5},
    }

    def __init__(self):
        self.initialized = False

    async def init(self):
        """初始化"""
        if self.initialized:
            return
        self.initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return getattr(settings, key, default)

    async def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        if key not in self.EDITABLE_KEYS:
            return False

        # 类型转换
        key_info = self.EDITABLE_KEYS[key]
        try:
            if key_info["type"] == "int":
                value = int(value)
            elif key_info["type"] == "bool":
                value = bool(value)
            elif key_info["type"] in ("string", "password"):
                value = str(value) if value else None
        except (ValueError, TypeError):
            return False

        from app.core.logger import logger

        setattr(settings, key, value)
        settings._save()

        # 密码不记录明文
        log_value = "******" if key_info["type"] == "password" else value
        logger.info(f"[RuntimeConfig] 配置已更新: {key} = {log_value}")
        return True

    async def set_batch(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """批量设置配置"""
        results = {}
        for key, value in updates.items():
            results[key] = await self.set(key, value)
        return results

    def get_all(self) -> Dict[str, Any]:
        """获取所有可编辑配置的当前值"""
        result = {}
        for key in self.EDITABLE_KEYS:
            result[key] = self.get(key)
        return result

    def get_schema(self) -> Dict[str, dict]:
        """获取配置项的元数据（包含分组信息）"""
        schema = {}
        for key, info in self.EDITABLE_KEYS.items():
            value = self.get(key)
            # 密码类型不返回明文，但标识是否已设置
            if info["type"] == "password":
                value = "******" if value else ""
            schema[key] = {
                **info,
                "value": value
            }
        return schema

    def get_groups(self) -> Dict[str, dict]:
        """获取配置分组信息"""
        return self.GROUPS

    async def reset(self, key: str) -> bool:
        """重置配置为默认值"""
        if key in DEFAULTS and key in self.EDITABLE_KEYS:
            setattr(settings, key, DEFAULTS[key])
            settings._save()
            return True
        return False


# 全局运行时配置实例
runtime_config = RuntimeConfig()
