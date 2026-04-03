"""
路由：POST /v1/images/generations - 文本生成图像（OpenAI Images API 兼容）

调用策略：
  - 万相 wan* 系列模型 → ImageSynthesis.call()（仅支持异步，SDK 封装为同步）
  - qwen-image-max 系列  → MultiModalConversation.call()（仅支持同步）
  - qwen-image-plus / qwen-image → MultiModalConversation.call()（两种均可，此处统一用同步）
"""

import logging
import os
import time
from http import HTTPStatus
from typing import Optional
import base64
import json

import dashscope
from dashscope import ImageSynthesis, MultiModalConversation
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from utils.common import build_openai_error, extract_api_key

logger = logging.getLogger("dashscope-router")

router = APIRouter()

# ---------------------------------------------------------------------------
# 支持的模型
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "qwen-image-plus")
DEFAULT_EDIT_MODEL = os.getenv("DEFAULT_IMAGE_EDIT_MODEL", "qwen-image-edit-plus")

# 同步模型（通过 MultiModalConversation.call 调用）
# 注：使用前缀匹配，快照版本（如 qwen-image-max-2025-12-30）会自动匹配基础模型名
SYNC_MODELS = {
    "qwen-image-max",
    "qwen-image-plus",
    "qwen-image",
    "qwen-image-2.0",
    "qwen-image-2.0-pro",
}

# 异步模型（通过 ImageSynthesis.call 调用，SDK 封装为同步）
ASYNC_MODELS = {
    "qwen-image-plus",
    "qwen-image",
    "wan2.6-t2i",
    "wan2.5-t2i-preview",
    "wan2.2-t2i-plus",
    "wan2.2-t2i-flash",
    "wanx2.1-t2i-plus",
    "wanx2.1-t2i-turbo",
    "wanx2.0-t2i-turbo",
}

# 仅支持同步的模型（即只能用 MultiModalConversation.call）
SYNC_ONLY_MODELS = {
    "qwen-image-max",
    "qwen-image-2.0-pro",
}

# 仅支持异步的模型（即只能用 ImageSynthesis.call）
ASYNC_ONLY_MODELS = {
    "wan2.6-t2i",
    "wan2.5-t2i-preview",
    "wan2.2-t2i-plus",
    "wan2.2-t2i-flash",
    "wanx2.1-t2i-plus",
    "wanx2.1-t2i-turbo",
    "wanx2.0-t2i-turbo",
}

# 所有支持的模型列表（用于 /v1/models 展示）
IMAGE_EDIT_MODELS = {
    "qwen-image-2.0-pro",
    "qwen-image-2.0-pro-2026-03-03",
    "qwen-image-2.0",
    "qwen-image-2.0-2026-03-03",
    "qwen-image-edit-max",
    "qwen-image-edit-max-2026-01-16",
    "qwen-image-edit-plus",
    "qwen-image-edit-plus-2025-12-15",
    "qwen-image-edit-plus-2025-10-30",
    "qwen-image-edit",
}

SUPPORTED_MODELS = sorted(SYNC_MODELS | ASYNC_MODELS | IMAGE_EDIT_MODELS)

# OpenAI 模型名 → DashScope 模型名映射
MODEL_MAP: dict[str, str] = {
    "dall-e-2": "qwen-image-plus",
    "dall-e-3": "qwen-image-max",
    "gpt-image-1": "qwen-image-max",
}

# 默认尺寸
DEFAULT_SIZE = "1664*928"


def _resolve_model(model: str) -> str:
    """将 OpenAI 模型名映射到 DashScope 模型名，未匹配则透传。"""
    return MODEL_MAP.get(model.lower(), model)


def _normalize_size(size: str | None) -> str:
    """将 OpenAI 格式的 size（1024x1024）转换为 DashScope 格式（1024*1024）。"""
    if not size:
        return DEFAULT_SIZE
    return size.replace("x", "*").replace("X", "*")


def _model_in_set(model: str, model_set: set) -> bool:
    """前缀匹配：model 等于集合中某个基础名，或以「基础名-」开头（支持快照版本）。"""
    return any(model == base or model.startswith(base + "-") for base in model_set)


def _use_async_api(model: str) -> bool:
    """判断该模型是否应使用 ImageSynthesis（异步）API。"""
    if _model_in_set(model, ASYNC_ONLY_MODELS):
        return True
    # 对于同时支持两种方式的模型，优先使用同步 MultiModalConversation
    if _model_in_set(model, SYNC_ONLY_MODELS) or _model_in_set(model, SYNC_MODELS):
        return False
    # 未知模型默认使用异步 API
    return True


# ---------------------------------------------------------------------------
# POST /v1/images/generations
# ---------------------------------------------------------------------------
@router.post("/v1/images/generations")
async def images_generations(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI 兼容的文本生成图像接口。

    请求体 JSON:
        - model (str, optional): 模型名，默认 qwen-image-plus
        - prompt (str, required): 图像描述文本
        - n (int, optional): 生成图片数量，默认 1
        - size (str, optional): 图片尺寸，如 "1664x928"，默认 "1664*928"
        - quality (str, optional): 质量（暂透传为参考，部分模型不支持）
        - response_format (str, optional): "url" 或 "b64_json"，默认 "url"
        - negative_prompt (str, optional): 反向提示词（DashScope 扩展参数）
        - prompt_extend (bool, optional): 是否扩展提示词，默认 True
        - watermark (bool, optional): 是否添加水印，默认 False
    """

    # 1. 鉴权
    api_key = extract_api_key(authorization)

    # 2. 解析请求体
    content_type = request.headers.get("content-type", "")
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
    else:
        try:
            body = await request.json()
        except Exception:
            raw = await request.body()
            logger.error("无法解析请求体, Content-Type=%s, raw=%s", content_type, raw[:500])
            raise HTTPException(
                status_code=400,
                detail=build_openai_error(
                    f"不支持的 Content-Type: {content_type}。请使用 application/json。"
                ),
            )

    # 3. 提取参数
    model_raw = body.get("model", DEFAULT_MODEL)
    prompt = body.get("prompt")
    n = int(body.get("n", 1))
    size = _normalize_size(body.get("size"))
    response_format = str(body.get("response_format", "url")).lower()
    negative_prompt = body.get("negative_prompt", "")
    prompt_extend = body.get("prompt_extend", True)
    watermark = body.get("watermark", False)

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少必填参数: prompt"),
        )

    ds_model = _resolve_model(model_raw)

    # 仅当模型属于纯编辑模型（不在生成模型集合中）时才拒绝
    if _model_in_set(ds_model, IMAGE_EDIT_MODELS) and not _model_in_set(ds_model, SYNC_MODELS):
        raise HTTPException(
            status_code=400,
            detail=build_openai_error(
                f"模型 {ds_model} 为图像编辑模型，请使用 /v1/images/edits 接口。"
            ),
        )

    logger.info(
        "收到图像生成请求: model=%s→%s, n=%d, size=%s, prompt_length=%d",
        model_raw, ds_model, n, size, len(prompt),
    )

    # 4. 判断调用方式并执行
    use_async = _use_async_api(ds_model)

    try:
        start_ts = time.time()

        if use_async:
            # ---- 异步 API（ImageSynthesis.call，SDK 封装为同步等待） ----
            result = await _call_async_api(
                api_key=api_key,
                model=ds_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )
        else:
            # ---- 同步 API（MultiModalConversation.call） ----
            result = await _call_sync_api(
                api_key=api_key,
                model=ds_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                size=size,
                prompt_extend=prompt_extend,
                watermark=watermark,
            )

        elapsed = time.time() - start_ts

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("图像生成调用异常")
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"调用 DashScope 失败: {e}",
                error_type="upstream_error",
            ),
        )

    logger.info("图像生成完成: %.2fs, images=%d", elapsed, len(result))

    # 5. 构造 OpenAI 兼容响应
    created = int(time.time())
    data = []
    for item in result:
        entry: dict = {}
        if response_format == "b64_json":
            # 需要下载图片并转为 base64
            b64 = await _download_as_base64(item["url"])
            if b64:
                entry["b64_json"] = b64
            else:
                entry["url"] = item["url"]  # 降级为 url
        else:
            entry["url"] = item["url"]

        if item.get("revised_prompt"):
            entry["revised_prompt"] = item["revised_prompt"]
        data.append(entry)

    return JSONResponse(content={"created": created, "data": data})


# ---------------------------------------------------------------------------
# POST /v1/images/edits
# ---------------------------------------------------------------------------
@router.post("/v1/images/edits")
async def images_edits(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI 兼容的图像编辑接口。

    支持 application/json 或 multipart/form-data。
    - 必填: prompt
    - 输入图像: image/images（URL、data URI 或上传文件）
    - 模型: 默认 qwen-image-edit-plus，按前缀匹配
    """

    api_key = extract_api_key(authorization)

    content_type = request.headers.get("content-type", "")
    body: dict = {}
    form_data = None

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
        form_data = await request.form()
        body = dict(form_data)
    else:
        try:
            body = await request.json()
        except Exception:
            raw = await request.body()
            logger.error("无法解析请求体, Content-Type=%s, raw=%s", content_type, raw[:500])
            raise HTTPException(
                status_code=400,
                detail=build_openai_error(
                    f"不支持的 Content-Type: {content_type}。请使用 application/json 或 multipart/form-data。"
                ),
            )

    model_raw = body.get("model", DEFAULT_EDIT_MODEL)
    prompt = body.get("prompt")
    n = int(body.get("n", 1))
    size = _normalize_size(body.get("size"))
    response_format = str(body.get("response_format", "url")).lower()
    negative_prompt = body.get("negative_prompt", "")
    prompt_extend = _to_bool(body.get("prompt_extend", True), True)
    watermark = _to_bool(body.get("watermark", False), False)

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少必填参数: prompt"),
        )

    ds_model = _resolve_model(model_raw)
    if not _model_in_set(ds_model, IMAGE_EDIT_MODELS):
        raise HTTPException(
            status_code=400,
            detail=build_openai_error(
                f"模型 {ds_model} 不在图像编辑支持列表中。"
            ),
        )

    input_images = await _extract_input_images(body=body, form_data=form_data)
    if not input_images:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("缺少图像输入：请提供 image 或 images 参数（可多张）。"),
        )
    if len(input_images) > 3:
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("图像编辑最多支持 3 张输入图像。"),
        )
    if not (1 <= n <= 6):
        raise HTTPException(
            status_code=400,
            detail=build_openai_error("参数 n 必须在 1 到 6 之间。"),
        )

    logger.info(
        "收到图像编辑请求: model=%s→%s, inputs=%d, n=%d, size=%s, prompt_length=%d",
        model_raw,
        ds_model,
        len(input_images),
        n,
        size,
        len(prompt),
    )

    try:
        start_ts = time.time()
        result = await _call_edit_api(
            api_key=api_key,
            model=ds_model,
            prompt=prompt,
            input_images=input_images,
            negative_prompt=negative_prompt,
            n=n,
            size=size,
            prompt_extend=prompt_extend,
            watermark=watermark,
        )
        elapsed = time.time() - start_ts
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("图像编辑调用异常")
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"调用 DashScope 失败: {e}",
                error_type="upstream_error",
            ),
        )

    logger.info("图像编辑完成: %.2fs, images=%d", elapsed, len(result))

    created = int(time.time())
    data = []
    for item in result:
        entry: dict = {}
        if response_format == "b64_json":
            b64 = await _download_as_base64(item["url"])
            if b64:
                entry["b64_json"] = b64
            else:
                entry["url"] = item["url"]
        else:
            entry["url"] = item["url"]

        if item.get("revised_prompt"):
            entry["revised_prompt"] = item["revised_prompt"]
        data.append(entry)

    return JSONResponse(content={"created": created, "data": data})


# ---------------------------------------------------------------------------
# 内部：异步 API 调用（ImageSynthesis.call）
# ---------------------------------------------------------------------------
async def _call_async_api(
    *,
    api_key: str,
    model: str,
    prompt: str,
    negative_prompt: str,
    n: int,
    size: str,
    prompt_extend: bool,
    watermark: bool,
) -> list[dict]:
    """
    通过 ImageSynthesis.call 调用（SDK 内部封装异步逻辑，阻塞等待结果）。
    返回 [{"url": ..., "revised_prompt": ...}, ...]
    """
    import asyncio

    loop = asyncio.get_event_loop()

    def _blocking_call():
        return ImageSynthesis.call(
            api_key=api_key,
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            n=n,
            size=size,
            prompt_extend=prompt_extend,
            watermark=watermark,
        )

    # ImageSynthesis.call 是阻塞调用，放到线程池中执行
    rsp = await loop.run_in_executor(None, _blocking_call)

    if rsp.status_code != HTTPStatus.OK:
        error_msg = getattr(rsp, "message", None) or str(rsp)
        logger.error(
            "ImageSynthesis 返回错误: status=%s, code=%s, message=%s",
            rsp.status_code, getattr(rsp, "code", ""), error_msg,
        )
        raise HTTPException(
            status_code=rsp.status_code if 400 <= rsp.status_code < 600 else 502,
            detail=build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(rsp, "code", None),
            ),
        )

    # 解析结果
    results = []
    try:
        for item in rsp.output.results:
            url = item.url if hasattr(item, "url") else item.get("url", "")
            revised = ""
            if hasattr(item, "actual_prompt"):
                revised = item.actual_prompt
            elif isinstance(item, dict):
                revised = item.get("actual_prompt", "")
            results.append({"url": url, "revised_prompt": revised})
    except (AttributeError, TypeError) as e:
        logger.error("解析 ImageSynthesis 响应失败: %s | raw=%s", e, rsp)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"无法解析上游响应: {e}",
                error_type="upstream_error",
            ),
        )

    return results


# ---------------------------------------------------------------------------
# 内部：同步 API 调用（MultiModalConversation.call）
# ---------------------------------------------------------------------------
async def _call_sync_api(
    *,
    api_key: str,
    model: str,
    prompt: str,
    negative_prompt: str,
    n: int,
    size: str,
    prompt_extend: bool,
    watermark: bool,
) -> list[dict]:
    """
    通过 MultiModalConversation.call 调用（真正同步接口）。
    返回 [{"url": ..., "revised_prompt": ...}, ...]
    """
    import asyncio

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    loop = asyncio.get_event_loop()

    def _blocking_call():
        return MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            stream=False,
            watermark=watermark,
            prompt_extend=prompt_extend,
            negative_prompt=negative_prompt or "",
            size=size,
            n=n,
        )

    rsp = await loop.run_in_executor(None, _blocking_call)

    if rsp.status_code != 200:
        error_msg = getattr(rsp, "message", None) or str(rsp)
        logger.error(
            "MultiModalConversation 图像生成错误: status=%s, code=%s, message=%s",
            rsp.status_code, getattr(rsp, "code", ""), error_msg,
        )
        raise HTTPException(
            status_code=rsp.status_code if 400 <= rsp.status_code < 600 else 502,
            detail=build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(rsp, "code", None),
            ),
        )

    # 解析结果
    results = []
    try:
        choices = rsp.output.choices
        for choice in choices:
            content_list = choice.message.content
            for content_item in content_list:
                if isinstance(content_item, dict) and "image" in content_item:
                    results.append({
                        "url": content_item["image"],
                        "revised_prompt": "",
                    })
    except (AttributeError, TypeError, KeyError) as e:
        logger.error("解析 MultiModalConversation 图像响应失败: %s | raw=%s", e, rsp)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"无法解析上游响应: {e}",
                error_type="upstream_error",
            ),
        )

    if not results:
        logger.error("MultiModalConversation 图像响应中未找到图片: raw=%s", rsp)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                "上游响应中未包含图片数据。",
                error_type="upstream_error",
            ),
        )

    return results


async def _call_edit_api(
    *,
    api_key: str,
    model: str,
    prompt: str,
    input_images: list[str],
    negative_prompt: str,
    n: int,
    size: str,
    prompt_extend: bool,
    watermark: bool,
) -> list[dict]:
    """通过 MultiModalConversation.call 调用图像编辑模型。"""
    import asyncio

    content = [{"image": image} for image in input_images]
    content.append({"text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    loop = asyncio.get_event_loop()

    def _blocking_call():
        return MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format="message",
            stream=False,
            watermark=watermark,
            prompt_extend=prompt_extend,
            negative_prompt=negative_prompt or " ",
            size=size,
            n=n,
        )

    rsp = await loop.run_in_executor(None, _blocking_call)

    if rsp.status_code != 200:
        error_msg = getattr(rsp, "message", None) or str(rsp)
        logger.error(
            "MultiModalConversation 图像编辑错误: status=%s, code=%s, message=%s",
            rsp.status_code, getattr(rsp, "code", ""), error_msg,
        )
        raise HTTPException(
            status_code=rsp.status_code if 400 <= rsp.status_code < 600 else 502,
            detail=build_openai_error(
                f"DashScope 错误: {error_msg}",
                error_type="upstream_error",
                code=getattr(rsp, "code", None),
            ),
        )

    results = []
    try:
        choices = rsp.output.choices
        for choice in choices:
            content_list = choice.message.content
            for content_item in content_list:
                if isinstance(content_item, dict) and "image" in content_item:
                    results.append(
                        {
                            "url": content_item["image"],
                            "revised_prompt": "",
                        }
                    )
    except (AttributeError, TypeError, KeyError) as e:
        logger.error("解析 MultiModalConversation 图像编辑响应失败: %s | raw=%s", e, rsp)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                f"无法解析上游响应: {e}",
                error_type="upstream_error",
            ),
        )

    if not results:
        logger.error("MultiModalConversation 图像编辑响应中未找到图片: raw=%s", rsp)
        raise HTTPException(
            status_code=502,
            detail=build_openai_error(
                "上游响应中未包含图片数据。",
                error_type="upstream_error",
            ),
        )

    return results


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def _extract_input_images(body: dict, form_data) -> list[str]:
    """从 JSON / Form 中提取输入图像（URL、data URI 或上传文件转 data URI）。"""
    raw_items = []

    image_keys = (
        "image",
        "images",
        "image[]",
        "images[]",
        "input_image",
        "input_images",
        "image_url",
        "image_urls",
    )

    for key in image_keys:
        if key in body and body[key] is not None:
            raw_items.append(body[key])

    if form_data is not None:
        for key in image_keys:
            if hasattr(form_data, "getlist"):
                raw_items.extend(form_data.getlist(key))

    expanded_items = []
    for item in raw_items:
        if isinstance(item, list):
            expanded_items.extend(item)
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        expanded_items.extend(parsed)
                        continue
                except Exception:
                    pass
            expanded_items.append(text)
            continue
        expanded_items.append(item)

    normalized = []
    for item in expanded_items:
        if isinstance(item, str):
            if item.startswith("http://") or item.startswith("https://") or item.startswith("data:"):
                normalized.append(item)
            continue

        if hasattr(item, "read") and hasattr(item, "filename"):
            content = await item.read()
            if not content:
                continue
            mime = getattr(item, "content_type", None) or "image/png"
            normalized.append(f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}")

    return normalized


# ---------------------------------------------------------------------------
# 辅助：下载图片转 base64
# ---------------------------------------------------------------------------
async def _download_as_base64(url: str) -> str | None:
    """下载图片并返回 base64 字符串，失败返回 None。"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.error("下载图片转 base64 失败: %s", e)
        return None
