"""OpenAI 数据模型"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

    # 扩展字段 - 用于会话管理
    conversation_id: Optional[str] = None  # OpenAI 格式的会话 ID
    thinking: Optional[bool] = None  # 是否显示思考过程（None=自动检测）


class ChatCompletionResponseMessage(BaseModel):
    """响应消息"""
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    """响应选项"""
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """聊天补全响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

    # 扩展字段
    conversation_id: Optional[str] = None  # 返回会话 ID 供后续使用


class ChatCompletionChunkDelta(BaseModel):
    """流式响应增量"""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """流式响应选项"""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """流式响应块"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

    # 扩展字段
    conversation_id: Optional[str] = None


class Model(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "xai"


class ModelList(BaseModel):
    """模型列表"""
    object: str = "list"
    data: List[Model]


class ResponseRequest(BaseModel):
    """继续对话请求 - 真实上下文，不拼接历史"""
    conversation_id: str  # 必需：会话 ID
    message: Union[str, List[Dict[str, Any]]]  # 当前新消息
    model: Optional[str] = "grok-2-1212"
    stream: Optional[bool] = False


class ResponseResponse(BaseModel):
    """继续对话响应"""
    id: str
    object: str = "response"
    created: int
    model: str
    conversation_id: str
    message: str  # AI 回复内容
