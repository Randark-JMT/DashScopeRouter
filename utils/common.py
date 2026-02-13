"""
共用工具函数 - 鉴权、MIME 检测、错误构造等
"""

import logging
import mimetypes
import os
from typing import Optional

from fastapi import HTTPException, UploadFile

logger = logging.getLogger("dashscope-router")

# ---------------------------------------------------------------------------
# DashScope 基础配置
# ---------------------------------------------------------------------------
DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ---------------------------------------------------------------------------
# MIME 类型映射（根据文件扩展名补全）
# ---------------------------------------------------------------------------
MIME_FALLBACK: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".amr": "audio/amr",
    ".pcm": "audio/pcm",
}


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------
def extract_api_key(authorization: Optional[str]) -> str:
    """从 Authorization header 提取 API Key，回退到环境变量。"""
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:].strip()
        return authorization.strip()
    if DEFAULT_API_KEY:
        return DEFAULT_API_KEY
    raise HTTPException(
        status_code=401,
        detail=build_openai_error(
            "未提供 API Key。请通过 Authorization: Bearer <key> 或环境变量 DASHSCOPE_API_KEY 设置。",
            code="missing_api_key",
        ),
    )


# ---------------------------------------------------------------------------
# MIME 检测
# ---------------------------------------------------------------------------
def resolve_mime_type(upload: UploadFile) -> str:
    """尝试从 UploadFile 获取 MIME 类型。"""
    if upload.content_type and upload.content_type != "application/octet-stream":
        return upload.content_type
    if upload.filename:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext in MIME_FALLBACK:
            return MIME_FALLBACK[ext]
        guessed = mimetypes.guess_type(upload.filename)[0]
        if guessed:
            return guessed
    return "audio/mpeg"


# ---------------------------------------------------------------------------
# 错误响应构造
# ---------------------------------------------------------------------------
def build_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> dict:
    """构造 OpenAI 风格的错误响应体。"""
    err: dict = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    if code:
        err["error"]["code"] = code
    return err
