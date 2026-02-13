"""
路由：POST /v1/audio/speech - 语音合成（OpenAI TTS API 兼容）
"""

import base64
import gc
import io
import logging
import os
import time
from typing import Optional

import httpx

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


# OpenAI 模型名 → DashScope 模型名映射
MODEL_MAP: dict[str, str] = {
    "tts-1": "qwen3-tts-flash",
    "tts-1-hd": "qwen3-tts-flash",
}

# 默认音色（当请求中未提供 voice 时使用）
DEFAULT_VOICE = "Chelsie"


def _resolve_voice(voice: str) -> str:
    """将 OpenAI voice 名映射到 DashScope voice，未匹配则透传。"""
    return VOICE_MAP.get(voice.lower(), voice)


def _resolve_model(model: str) -> str:
    """将 OpenAI 模型名映射到 DashScope 模型名，未匹配则透传。"""
    return MODEL_MAP.get(model.lower(), model)


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

    # 2. 解析请求体（兼容 JSON 和 form-data）
    content_type = request.headers.get("content-type", "")
    logger.debug("TTS 请求 Content-Type: %s", content_type)

    body: dict = {}
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            logger.error("JSON 解析失败: %s", e)
            raise HTTPException(
                status_code=400,
                detail=build_openai_error("请求体必须为有效的 JSON。"),
            )
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        body = dict(form)
        logger.debug("从 form-data 解析请求体: keys=%s", list(body.keys()))
    else:
        # 尝试按 JSON 解析
        try:
            body = await request.json()
        except Exception:
            raw = await request.body()
            logger.error("无法解析请求体, Content-Type=%s, raw=%s", content_type, raw[:500])
            raise HTTPException(
                status_code=400,
                detail=build_openai_error(f"不支持的 Content-Type: {content_type}。请使用 application/json。"),
            )

    logger.debug("TTS 请求体: %s", {k: (v[:50] if isinstance(v, str) and len(v) > 50 else v) for k, v in body.items()})

    model = body.get("model", DEFAULT_MODEL)
    text = body.get("input")
    voice = body.get("voice", "") or ""
    response_format = str(body.get("response_format", "mp3")).lower()

    if not text:
        logger.warning("TTS 请求缺少 input 参数, body keys=%s", list(body.keys()))
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少必填参数: input"),
        )

    # voice 未提供时使用默认音色
    if not voice:
        voice = DEFAULT_VOICE
        logger.info("voice 未提供，使用默认音色: %s", DEFAULT_VOICE)

    ds_model = _resolve_model(model)
    ds_voice = _resolve_voice(voice)

    logger.info(
        "收到 TTS 请求: model=%s→%s, voice=%s→%s, text_length=%d, format=%s",
        model,
        ds_model,
        voice,
        ds_voice,
        len(text),
        response_format,
    )

    # 3. 调用 DashScope
    try:
        start_ts = time.time()
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=ds_model,
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
    # DashScope TTS 响应结构: output.audio.url / output.audio.data
    try:
        audio_obj = response.output.get("audio") if isinstance(response.output, dict) else getattr(response.output, "audio", None)

        if audio_obj is None:
            raise ValueError(f"响应中未找到 audio 字段: {response.output}")

        # audio_obj 可能是 dict 或对象
        if isinstance(audio_obj, dict):
            audio_url = audio_obj.get("url", "")
            audio_data = audio_obj.get("data", "")
        else:
            audio_url = getattr(audio_obj, "url", "")
            audio_data = getattr(audio_obj, "data", "")

        if audio_data:
            # 内联 base64 数据
            if audio_data.startswith("data:"):
                b64_part = audio_data.split(",", 1)[1]
                audio_bytes = base64.b64decode(b64_part)
            else:
                audio_bytes = base64.b64decode(audio_data)
            logger.info("从内联 data 获取音频, size=%d bytes", len(audio_bytes))
        elif audio_url:
            # 从 URL 下载音频
            logger.info("从 URL 下载音频: %s", audio_url[:120])
            async with httpx.AsyncClient(timeout=60) as client:
                dl_resp = await client.get(audio_url)
                dl_resp.raise_for_status()
                audio_bytes = dl_resp.content
            logger.info("音频下载完成, size=%d bytes", len(audio_bytes))
        else:
            raise ValueError(f"audio 字段中无 url 也无 data: {audio_obj}")

    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
        logger.error("无法解析 DashScope TTS 响应: %s | raw=%s", e, response)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"无法解析上游 TTS 响应: {e}",
                error_type="upstream_error",
            ),
        )
    except httpx.HTTPError as e:
        logger.error("下载音频失败: %s", e)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"下载音频失败: {e}",
                error_type="upstream_error",
            ),
        )

    # 释放 DashScope 响应对象，仅保留 audio_bytes
    del response
    gc.collect()

    mime = RESPONSE_MIME.get(response_format, "audio/mpeg")
    logger.info("TTS 合成完成: %.2fs, audio_size=%d bytes", elapsed, len(audio_bytes))

    return Response(
        content=audio_bytes,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{response_format}"',
        },
    )
