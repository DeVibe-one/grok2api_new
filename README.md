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
- **可追踪响应** — 每次请求返回 `X-Request-ID`，便于日志排障与问题定位
- **思考过程** — 支持 Thinking 模型的推理过程展示（`<think>` 标签）
- **搜索过程** — 实时展示 Grok 搜索查询和结果数量
- **图片支持** — 支持图片上传和图片生成结果缓存
- **自动清理** — 日志文件、请求日志、图片缓存均支持自动清理，后台可配置上限
- **管理后台** — Web 可视化管理 Token、会话、统计、日志、API Key、系统配置

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/Tomiya233/grok2api_new.git && cd grok2api_new

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

### Docker 部署

使用 Docker Compose（推荐）：

```bash
# 构建并后台启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止并删除容器
docker compose down
```

使用 Docker 命令：

```bash
# 构建镜像
docker build -t grok2api:latest .

# 运行容器（挂载数据和日志目录）
docker run -d \
  --name grok2api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  grok2api:latest
```

说明：
- 首次启动会自动初始化 `data/` 和 `logs/`。
- 已在 `docker-compose.yml` 中默认映射 `./data` 与 `./logs`，用于持久化。

首次启动自动生成 `data/` 和 `logs/` 目录及所有必要文件，无需手动创建。

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
API Key: sk-test（默认，可在后台管理）
Model: grok-4.2
```

## 上下文实现

与大多数逆向代理每次重发完整消息历史不同，本项目通过 Grok 原生的会话机制实现真实上下文：

### 核心原理

Grok 的对话基于两个 ID：
- **conversationId** — 标识一个对话
- **responseId** — 标识对话中的每条消息，新消息通过 `parentResponseId` 关联到上一条

本项目在服务端缓存这两个 ID，后续对话只发送新消息，Grok 服务端自动维持完整上下文。

### 对话流程

**首次对话：**
```
客户端 → POST /v1/chat/completions（完整消息）
       → Grok API: POST /conversations/new
       → 返回 conversationId + responseId
       → 缓存上下文，计算消息哈希
       → 分享会话（获取 share_link_id，为跨账号做准备）
       → 返回响应 + conversation_id
```

**续接对话（同 Token）：**
```
客户端 → POST /v1/chat/completions（新消息 + conversation_id）
       → 从缓存取出 conversationId + responseId
       → Grok API: POST /conversations/{id}/responses（只发新消息）
       → 更新缓存
       → 返回响应
```

**续接对话（不同 Token，轮询切换时）：**
```
客户端 → 发送新消息
       → 轮询到不同 Token
       → 通过 share_link_id 克隆会话到新账号
       → Grok API: POST /share_links/{id}/clone
       → 获取新的 conversationId + responseId
       → 继续对话，上下文无损
```

### 自动识别

客户端不传 `conversation_id` 也能续接对话。系统基于消息哈希自动匹配：

1. **存储时**：对 `system + 所有 user 消息` 计算 SHA-256 哈希
2. **查找时**：对 `system + 除最后一条外的 user 消息` 计算哈希
3. 新一轮对话的「历史部分」哈希 = 上一轮的「完整」哈希 → 匹配成功

这意味着即使客户端（如 ChatGPT-Next-Web）每次重发全部历史，本项目也能识别出是同一个对话，只将最后一条新消息发给 Grok。

## 配置

配置文件为 `data/config.json`，首次启动自动生成，也可在管理后台「系统配置」中热修改：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `admin_username` | `admin` | 管理后台用户名 |
| `admin_password` | `admin` | 管理后台密码 |
| `proxy_url` | 空 | HTTP 代理地址 |
| `base_url` | 空 | 图片缓存外部访问地址 |
| `request_timeout` | `120` | 请求超时（秒） |
| `stream_timeout` | `600` | 流式超时（秒） |
| `conversation_ttl` | `72000` | 会话存活时间（秒，默认 20 小时） |
| `max_conversations_per_token` | `100` | 每个 Token 最大会话数 |
| `max_log_entries` | `1000` | 请求日志最大条数 |
| `max_image_cache_mb` | `500` | 图片缓存上限（MB） |
| `max_log_file_mb` | `10` | 日志文件上限（MB） |
| `log_level` | `INFO` | 日志级别 |
| `show_thinking` | `true` | 显示思考过程（`<think>` 标签） |
| `show_search` | `true` | 显示搜索查询和结果数量 |

## 支持模型

| 模型 ID | 说明 |
|---------|------|
| `grok-3` | Grok 3 标准 |
| `grok-3-mini` | Grok 3 Mini Thinking |
| `grok-4.2` | Grok 4.2 标准 |
| `grok-4.2-fast` | Grok 4.2 快速 |
| `grok-expert` | Grok 专家推理（深度思考） |

未识别的模型名会原样透传给 Grok。

## API 接口

### 对话补全

```http
POST /v1/chat/completions
Authorization: Bearer sk-test
Content-Type: application/json

{
  "model": "grok-4.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

支持 `conversation_id` 参数续接多轮对话。

响应头：
- `X-Request-ID`：请求追踪 ID

在流式（SSE）且首次新建会话时，首个 chunk 也会携带稳定 `conversation_id`，可立即用于下一轮续接。

### 继续对话

```http
POST /v1/responses
Authorization: Bearer sk-test
Content-Type: application/json

{
  "conversation_id": "conv-xxx",
  "model": "grok-4.2",
  "message": "继续讲下一步",
  "stream": false
}
```

说明：
- `conversation_id` 必须通过请求体传入。
- 缺少 `conversation_id` 会返回 `400`，错误码 `missing_conversation_id`，并附带 `request_id` 便于排查。

### 模型列表

```http
GET /v1/models
```

### 健康检查

```http
GET /health
```

### 错误追踪

错误响应会尽量包含：
- `detail.error.code`：错误码
- `detail.error.message`：错误信息
- `detail.error.request_id`：请求追踪 ID（服务端异常或关键校验失败时）

同时响应头会返回 `X-Request-ID`，用于串联客户端日志与服务端日志。

## 管理后台

后台路径：`/admin`

| 功能 | 说明 |
|------|------|
| Token 管理 | 增删改查、批量导入、额度检测、账号类型识别（Free/Super）、冷却管理 |
| 会话管理 | 查看活跃会话、手动清理 |
| 请求统计 | 24h 趋势、7 天统计、模型分布 |
| 请求日志 | 全量请求审计日志（自动清理） |
| API Key | 创建 / 批量创建 / 启停管理 / 调用统计 |
| 系统配置 | 热修改运行时参数（分组卡片式界面） |
| 图片缓存 | 查看和清理缓存的图片（自动清理） |

## Changelog

### 2026-02-21

- **接口加固**：增强 OpenAI 请求模型校验（角色约束、空内容拦截、关键字段最小长度约束）。
- **会话一致性**：修复会话清理时哈希索引残留问题，新增 `clear_all()` 统一清理入口。
- **可观测性**：新增 `X-Request-ID` 响应头，错误响应附带 `request_id` 便于排障。
- **体验增强**：`/v1/responses` 统一使用请求体 `conversation_id` 续接会话，缺失时返回 `missing_conversation_id`。
- **流式续接优化**：新会话流式首包返回稳定 `conversation_id`，可立即用于下一轮续接。
- **性能优化**：批量创建 API Key 改为单次落盘；请求日志分页读取改为切片策略。
- **管理台优化**：后台与登录页错误提示解析增强，避免出现 `[object Object]`。
- **死代码清理**：删除无引用文件 `app/services/grok_client_patch.py`，并移除未使用方法。
- **部署能力**：新增 `Dockerfile`、`docker-compose.yml` 与 `.dockerignore`，支持容器化部署。

## 项目结构

```
├── main.py                          # 入口
├── requirements.txt                 # 依赖
├── Dockerfile                       # Docker 镜像构建
├── docker-compose.yml               # Docker Compose 部署
├── .dockerignore                    # Docker 构建忽略
├── app/
│   ├── api/
│   │   ├── admin.py                 # 管理后台 API
│   │   └── v1/
│   │       ├── chat.py              # /v1/chat/completions
│   │       ├── models.py            # /v1/models + 模型映射
│   │       └── images.py            # 图片代理
│   ├── core/
│   │   ├── config.py                # 配置管理
│   │   ├── logger.py                # 日志（自动轮转）
│   │   └── storage.py               # JSON 存储
│   ├── models/
│   │   └── openai_models.py         # OpenAI 格式数据模型
│   ├── services/
│   │   ├── grok_client.py           # Grok API 客户端
│   │   ├── token_manager.py         # Token 轮询 + 冷却
│   │   ├── conversation_manager.py  # 会话上下文管理
│   │   ├── api_keys.py              # API Key 管理
│   │   ├── headers.py               # 请求头生成
│   │   ├── image_cache.py           # 图片缓存（自动清理）
│   │   ├── image_upload.py          # 图片上传
│   │   ├── request_stats.py         # 请求统计
│   │   └── request_logger.py        # 请求日志（自动清理）
│   └── template/                    # 前端模板
├── data/                            # 持久化数据（自动生成）
└── logs/                            # 日志文件（自动生成）
```
