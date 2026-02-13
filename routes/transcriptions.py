"""
路由：POST /v1/audio/transcriptions - 音频识别（OpenAI Whisper API 兼容）
"""

import base64
import logging
import os
import time
from typing import Optional

import dashscope
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from utils.common import build_openai_error, extract_api_key, resolve_mime_type

logger = logging.getLogger("dashscope-router")

router = APIRouter()

# ---------------------------------------------------------------------------
# 支持的模型
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("DEFAULT_ASR_MODEL", "qwen3-asr-flash")

SUPPORTED_MODELS = [
    "qwen3-asr-flash",
    "qwen3-asr-flash-2025-09-08",
    "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash-filetrans-2025-11-17",
]


# ---------------------------------------------------------------------------
# 响应格式辅助
# ---------------------------------------------------------------------------
def _format_verbose_json(text: str, model: str, duration: Optional[float] = None) -> dict:
    return {
        "task": "transcribe",
        "language": "unknown",
        "duration": duration or 0.0,
        "text": text,
        "segments": [],
        "words": [],
    }


def _format_srt(text: str) -> str:
    return f"1\n00:00:00,000 --> 99:59:59,999\n{text}\n"


def _format_vtt(text: str) -> str:
    return f"WEBVTT\n\n00:00:00.000 --> 99:59:59.999\n{text}\n"


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------
@router.post("/v1/audio/transcriptions")
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

    将上传的音频文件转发到 DashScope qwen3-asr 系列模型，
    并按照 OpenAI 的响应格式返回结果。

    支持的 response_format: json | text | verbose_json | srt | vtt
    """

    # 1. 鉴权
    api_key = extract_api_key(authorization)

    # 2. 读取音频并编码
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("上传的音频文件为空。"),
        )

    mime_type = resolve_mime_type(file)
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
            detail=build_openai_error(
                f"调用 DashScope 失败: {e}",
                error_type="upstream_error",
            ),
        )

    # 5. 解析响应
    if response.status_code != 200:
        error_msg = getattr(response, "message", None) or str(response)
        logger.error("DashScope 返回错误: status=%s, message=%s", response.status_code, error_msg)
        raise HTTPException(
            status_code=response.status_code if 400 <= response.status_code < 600 else 502,
            detail=build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(response, "code", None),
            ),
        )

    try:
        text = response.output.choices[0].message.content[0]["text"]
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        logger.error("无法解析 DashScope 响应: %s | raw=%s", e, response)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
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
        return JSONResponse(content={"text": text})
