"""
DashScope Router - OpenAI 兼容的音频识别 API 中转服务
将 OpenAI /v1/audio/transcriptions 请求转发到 DashScope MultiModalConversation API
"""

import base64
import json
import logging
import mimetypes
import os
import time
import uuid
from typing import Optional

import dashscope
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dashscope-router")

# ---------------------------------------------------------------------------
# DashScope 基础配置
# ---------------------------------------------------------------------------
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
dashscope.base_http_api_url = DASHSCOPE_BASE_URL

# 默认 API Key（可被请求中的 Authorization header 覆盖）
DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 默认模型
DEFAULT_MODEL = os.getenv("DEFAULT_ASR_MODEL", "qwen3-asr-flash")

# ---------------------------------------------------------------------------
# 支持的模型列表
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = [
    "qwen3-asr-flash",
    "qwen3-asr-flash-2025-09-08",
    "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash-filetrans-2025-11-17",
]

# ---------------------------------------------------------------------------
# MIME 类型映射（根据文件扩展名补全）
# ---------------------------------------------------------------------------
MIME_FALLBACK = {
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
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DashScope Router",
    description="OpenAI 兼容的音频识别 API 中转服务",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _extract_api_key(authorization: Optional[str]) -> str:
    """从 Authorization header 提取 API Key，回退到环境变量。"""
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:].strip()
        return authorization.strip()
    if DEFAULT_API_KEY:
        return DEFAULT_API_KEY
    raise HTTPException(
        status_code=401,
        detail={
            "error": {
                "message": "未提供 API Key。请通过 Authorization: Bearer <key> 或环境变量 DASHSCOPE_API_KEY 设置。",
                "type": "invalid_request_error",
                "code": "missing_api_key",
            }
        },
    )


def _resolve_mime_type(upload: UploadFile) -> str:
    """尝试从 UploadFile 获取 MIME 类型。"""
    if upload.content_type and upload.content_type != "application/octet-stream":
        return upload.content_type
    # 按文件名猜测
    if upload.filename:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext in MIME_FALLBACK:
            return MIME_FALLBACK[ext]
        guessed = mimetypes.guess_type(upload.filename)[0]
        if guessed:
            return guessed
    return "audio/mpeg"  # 兜底


def _build_openai_error(message: str, error_type: str = "invalid_request_error", code: Optional[str] = None) -> dict:
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


def _format_verbose_json(text: str, model: str, duration: Optional[float] = None) -> dict:
    """构造 verbose_json 格式的响应。"""
    return {
        "task": "transcribe",
        "language": "unknown",
        "duration": duration or 0.0,
        "text": text,
        "segments": [],
        "words": [],
    }


def _format_srt(text: str) -> str:
    """简易 SRT 格式（单段）。"""
    return f"1\n00:00:00,000 --> 99:59:59,999\n{text}\n"


def _format_vtt(text: str) -> str:
    """简易 WebVTT 格式（单段）。"""
    return f"WEBVTT\n\n00:00:00.000 --> 99:59:59.999\n{text}\n"


# ---------------------------------------------------------------------------
# 路由：GET /v1/models - 列出可用模型
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """返回支持的模型列表（OpenAI 兼容格式）。"""
    models = []
    for m in SUPPORTED_MODELS:
        models.append(
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "dashscope",
            }
        )
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# 路由：POST /v1/audio/transcriptions - 音频识别
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI 兼容的音频识别接口。

    将上传的音频文件转发到 DashScope qwen3‑asr 系列模型，并按照
    OpenAI 的响应格式返回结果。

    支持的 response_format: json | text | verbose_json | srt | vtt
    """

    # 1. 鉴权
    api_key = _extract_api_key(authorization)

    # 2. 读取音频文件并编码为 base64 data URI
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=_build_openai_error("上传的音频文件为空。"),
        )

    mime_type = _resolve_mime_type(file)
    base64_str = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:{mime_type};base64,{base64_str}"

    logger.info(
        "收到转写请求: model=%s, file=%s, size=%d bytes, mime=%s, language=%s, response_format=%s",
        model, file.filename, len(audio_bytes), mime_type, language, response_format,
    )

    # 3. 构造 DashScope 请求
    system_text = prompt or ""
    messages = [
        {"role": "system", "content": [{"text": system_text}]},
        {"role": "user", "content": [{"audio": data_uri}]},
    ]

    asr_options: dict = {"enable_itn": False}
    if language:
        asr_options["language"] = language

    # 4. 调用 DashScope
    try:
        start_ts = time.time()
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            asr_options=asr_options,
        )
        elapsed = time.time() - start_ts
    except Exception as e:
        logger.exception("DashScope 调用异常")
        raise HTTPException(
            status_code=502,
            detail=_build_openai_error(
                f"调用 DashScope 失败: {e}",
                error_type="upstream_error",
            ),
        )

    # 5. 解析 DashScope 响应
    if response.status_code != 200:
        error_msg = getattr(response, "message", None) or str(response)
        logger.error("DashScope 返回错误: status=%s, message=%s", response.status_code, error_msg)
        raise HTTPException(
            status_code=response.status_code if 400 <= response.status_code < 600 else 502,
            detail=_build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(response, "code", None),
            ),
        )

    # 提取识别文本
    try:
        text = response.output.choices[0].message.content[0]["text"]
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        logger.error("无法解析 DashScope 响应: %s | raw=%s", e, response)
        raise HTTPException(
            status_code=502,
            detail=_build_openai_error(
                f"无法解析上游响应: {e}",
                error_type="upstream_error",
            ),
        )

    logger.info("转写完成: %.2fs, text_length=%d", elapsed, len(text))

    # 6. 按 response_format 返回
    fmt = (response_format or "json").lower()

    if fmt == "text":
        return PlainTextResponse(content=text)
    elif fmt == "verbose_json":
        return JSONResponse(content=_format_verbose_json(text, model, elapsed))
    elif fmt == "srt":
        return PlainTextResponse(content=_format_srt(text), media_type="text/plain")
    elif fmt == "vtt":
        return PlainTextResponse(content=_format_vtt(text), media_type="text/plain")
    else:
        # 默认 json
        return JSONResponse(content={"text": text})


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
