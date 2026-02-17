# Grok2API

Grok 逆向代理，将 Grok 网页端 API 转换为 OpenAI 兼容格式。支持多账号轮询、真实多轮对话上下文、跨账号会话续接，附带可视化管理后台。

## 功能特性

- **OpenAI 兼容接口** — 直接对接 ChatGPT-Next-Web、LobeChat 等前端
- **真实多轮对话** — 通过 Grok 原生 `conversationId` + `responseId` 维持上下文，非重发历史消息
- **多账号轮询** — 支持多个 SSO Token 轮询调用，自动负载均衡
- **智能冷却** — 429 限流、认证失败等自动冷却，到期自动恢复
- **跨账号续接** — 通过 Share + Clone 机制实现不同账号无缝续接同一会话
- **账号类型检测** — 自动识别 Free / Super 会员账号
- **流式响应** — 支持 SSE 流式输出
- **思考过程** — 支持 Thinking 模型的推理过程展示（`<think>` 标签）
- **图片支持** — 支持图片上传和图片生成结果缓存
- **管理后台** — Web 可视化管理 Token、会话、统计、日志、API Key、系统配置

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repo-url> && cd 2api

# 安装依赖
pip install -r requirements.txt

# 启动
python main.py
```

或使用脚本：

```bash
# Linux / macOS
chmod +x install.sh start.sh
./install.sh && ./start.sh

# Windows
install.bat
start.bat
```

启动后访问：
- API 地址：`http://localhost:8000`
- 管理后台：`http://localhost:8000/admin`（默认账密 `admin` / `admin`）

### 添加 Token

1. 登录 [grok.com](https://grok.com)，从浏览器 Cookie 中提取 `sso` 值
2. 进入管理后台 → Token 管理 → 添加（支持批量添加）

### 对接客户端

在任意 OpenAI 兼容客户端中配置：

```
API Base URL: http://localhost:8000/v1
API Key: （如未设置 API_KEY 则留空）
Model: grok-4.1-thinking
```

## 配置

首次启动自动生成 `data/` 和 `logs/` 目录及所有必要文件，配置文件为 `data/config.json`：

```json
{
  "admin_username": "admin",
  "admin_password": "admin",
  "api_key": null,
  "proxy_url": null,
  "base_url": "",
  "request_timeout": 120,
  "stream_timeout": 600,
  "conversation_ttl": 600,
  "max_conversations_per_token": 100,
  "log_level": "INFO",
  "debug": false
}
```

| 字段 | 说明 |
|------|------|
| `admin_username` / `admin_password` | 管理后台登录账密 |
| `api_key` | API 访问密钥，`null` 则不鉴权 |
| `proxy_url` | HTTP 代理地址，如 `http://127.0.0.1:7890` |
| `base_url` | 图片缓存外部访问地址，留空用相对路径 |
| `request_timeout` / `stream_timeout` | 请求 / 流式超时（秒） |
| `conversation_ttl` | 会话存活时间（秒） |
| `max_conversations_per_token` | 每个 Token 最大会话数 |
| `log_level` | 日志级别：DEBUG / INFO / WARNING / ERROR |

以上配置也可在管理后台「系统配置」中热修改，修改后自动写回 `config.json`。

## 支持模型

| 模型 ID | 说明 |
|---------|------|
| `grok-3` | Grok 3 标准 |
| `grok-3-mini` | Grok 3 Mini Thinking |
| `grok-3-thinking` | Grok 3 完整思考 |
| `grok-4` | Grok 4 标准 |
| `grok-4-mini` | Grok 4 Mini Thinking |
| `grok-4-thinking` | Grok 4 完整思考 |
| `grok-4-heavy` | Grok 4 Heavy（需 Super 账号） |
| `grok-4.1-mini` | Grok 4.1 Mini Thinking |
| `grok-4.1-fast` | Grok 4.1 快速 |
| `grok-4.1-expert` | Grok 4.1 专家推理 |
| `grok-4.1-thinking` | Grok 4.1 完整思考 |

未识别的模型名会原样透传给 Grok。

## API 接口

### 对话补全

```http
POST /v1/chat/completions
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "model": "grok-4.1-thinking",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

支持 `conversation_id` 参数或 `X-Conversation-ID` 请求头续接多轮对话。

### 模型列表

```http
GET /v1/models
```

### 健康检查

```http
GET /health
```

## 管理后台

后台路径：`/admin`

| 功能 | 说明 |
|------|------|
| Token 管理 | 增删改查、批量导入、额度检测、账号类型识别、冷却管理 |
| 会话管理 | 查看活跃会话、手动清理 |
| 请求统计 | 24h 趋势、7 天统计、模型分布 |
| 请求日志 | 全量请求审计日志 |
| API Key | 创建 / 批量创建 / 启停管理 |
| 系统配置 | 热修改运行时参数 |
| 图片缓存 | 查看和清理缓存的图片 |

## 项目结构

```
├── main.py                          # 入口
├── requirements.txt                 # 依赖
├── app/
│   ├── api/
│   │   ├── admin.py                 # 管理后台 API
│   │   └── v1/
│   │       ├── chat.py              # /v1/chat/completions
│   │       ├── models.py            # /v1/models + 模型映射
│   │       └── images.py            # 图片代理
│   ├── core/
│   │   ├── config.py                # 配置管理
│   │   ├── logger.py                # 日志
│   │   └── storage.py               # JSON 存储
│   ├── models/
│   │   └── openai_models.py         # OpenAI 格式数据模型
│   ├── services/
│   │   ├── grok_client.py           # Grok API 客户端
│   │   ├── token_manager.py         # Token 轮询 + 冷却
│   │   ├── conversation_manager.py  # 会话上下文管理
│   │   ├── api_keys.py              # API Key 管理
│   │   ├── headers.py               # 请求头生成
│   │   ├── image_cache.py           # 图片缓存
│   │   ├── image_upload.py          # 图片上传
│   │   ├── request_stats.py         # 请求统计
│   │   └── request_logger.py        # 请求日志
│   └── template/                    # 前端模板
└── data/                            # 持久化数据
```

## 依赖

- Python 3.8+
- FastAPI + Uvicorn
- curl-cffi（TLS 指纹模拟）
- orjson / aiofiles / pydantic
