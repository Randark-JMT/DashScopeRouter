"""
路由：POST /v1/audio/speech - 语音合成（OpenAI TTS API 兼容）
"""

import base64
import io
import logging
import os
import time
from typing import Optional

import dashscope
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response

from utils.common import build_openai_error, extract_api_key

logger = logging.getLogger("dashscope-router")

router = APIRouter()

# ---------------------------------------------------------------------------
# 支持的模型与音色
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("DEFAULT_TTS_MODEL", "qwen3-tts-flash")

SUPPORTED_MODELS = [
    "qwen3-tts-instruct-flash",
    "qwen3-tts-instruct-flash-2026-01-26",
    "qwen3-tts-vd-2026-01-26",
    "qwen3-tts-vc-2026-01-22",
    "qwen3-tts-flash",
    "qwen3-tts-flash-2025-11-27",
    "qwen3-tts-flash-2025-09-18",
    "qwen-tts",
    "qwen-tts-latest",
    "qwen-tts-2025-05-22",
    "qwen-tts-2025-04-10",
]

# OpenAI voice → DashScope voice 映射
# OpenAI 标准 voice: alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer
# DashScope voices: Cherry, Serena, Ethan, Chelsie, Momo, Vivian, Moon, Maia, Kai, Nofish, Bella, Jennifer
VOICE_MAP: dict[str, str] = {
    "alloy": "Chelsie",
    "ash": "Ethan",
    "ballad": "Serena",
    "coral": "Cherry",
    "echo": "Kai",
    "fable": "Maia",
    "onyx": "Nofish",
    "nova": "Vivian",
    "sage": "Moon",
    "shimmer": "Bella",
}

# DashScope 响应音频的 MIME 类型映射
RESPONSE_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _resolve_voice(voice: str) -> str:
    """将 OpenAI voice 名映射到 DashScope voice，未匹配则透传。"""
    return VOICE_MAP.get(voice.lower(), voice)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------
@router.post("/v1/audio/speech")
async def audio_speech(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI 兼容的语音合成接口。

    请求体 JSON:
      - model (str, required): 模型名
      - input (str, required): 要合成的文本
      - voice (str, required): 音色名称
      - response_format (str, optional): 音频格式，默认 mp3
      - speed (float, optional): 语速（暂不支持，忽略）
    """

    # 1. 鉴权
    api_key = extract_api_key(authorization)

    # 2. 解析请求体
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("请求体必须为有效的 JSON。"),
        )

    model = body.get("model", DEFAULT_MODEL)
    text = body.get("input")
    voice = body.get("voice")
    response_format = body.get("response_format", "mp3").lower()

    if not text:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少必填参数: input"),
        )
    if not voice:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少必填参数: voice"),
        )

    ds_voice = _resolve_voice(voice)

    logger.info(
        "收到 TTS 请求: model=%s, voice=%s→%s, text_length=%d, format=%s",
        model, voice, ds_voice, len(text), response_format,
    )

    # 3. 调用 DashScope
    try:
        start_ts = time.time()
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            text=text,
            voice=ds_voice,
            stream=False,
        )
        elapsed = time.time() - start_ts
    except Exception as e:
        logger.exception("DashScope TTS 调用异常")
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"调用 DashScope 失败: {e}",
                error_type="upstream_error",
            ),
        )

    # 4. 解析响应
    if response.status_code != 200:
        error_msg = getattr(response, "message", None) or str(response)
        logger.error("DashScope TTS 返回错误: status=%s, message=%s", response.status_code, error_msg)
        raise HTTPException(
            status_code=response.status_code if 400 <= response.status_code < 600 else 502,
            detail=build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(response, "code", None),
            ),
        )

    # 提取音频数据
    try:
        audio_content = response.output.choices[0].message.content[0]
        # DashScope TTS 返回的音频可能在 "audio" 字段中（base64 或 URL）
        if "audio" in audio_content:
            audio_data_uri = audio_content["audio"]
            # 解析 data URI: data:<mime>;base64,<data>
            if audio_data_uri.startswith("data:"):
                b64_part = audio_data_uri.split(",", 1)[1]
                audio_bytes = base64.b64decode(b64_part)
            else:
                # 可能是原始 base64
                audio_bytes = base64.b64decode(audio_data_uri)
        else:
            raise ValueError(f"响应中未找到音频数据: {audio_content}")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
        logger.error("无法解析 DashScope TTS 响应: %s | raw=%s", e, response)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"无法解析上游 TTS 响应: {e}",
                error_type="upstream_error",
            ),
        )

    mime = RESPONSE_MIME.get(response_format, "audio/mpeg")
    logger.info("TTS 合成完成: %.2fs, audio_size=%d bytes", elapsed, len(audio_bytes))

    return Response(
        content=audio_bytes,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{response_format}"',
        },
    )
