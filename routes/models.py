"""
路由：GET /v1/models - 列出可用模型（OpenAI 兼容）
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from routes.transcriptions import SUPPORTED_MODELS as ASR_MODELS
from routes.speech import SUPPORTED_MODELS as TTS_MODELS

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    """返回所有支持的模型列表（OpenAI 兼容格式）。"""
    models = []

    for m in ASR_MODELS:
        models.append(
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "dashscope",
            }
        )

    for m in TTS_MODELS:
        models.append(
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "dashscope",
            }
        )

    return JSONResponse(content={"object": "list", "data": models})
