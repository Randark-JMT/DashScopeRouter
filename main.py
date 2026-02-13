"""
DashScope Router - OpenAI 兼容的多模态 API 中转服务
"""

import asyncio
import gc
import logging
import os
import time

import dashscope
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.config import check_ua_allowed, get_config, load_config
from routes.transcriptions import router as transcriptions_router
from routes.speech import router as speech_router
from routes.models import router as models_router

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dashscope-router")

# ---------------------------------------------------------------------------
# 加载配置
# ---------------------------------------------------------------------------
config = load_config()

# ---------------------------------------------------------------------------
# DashScope 基础配置
# ---------------------------------------------------------------------------
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
dashscope.base_http_api_url = DASHSCOPE_BASE_URL

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DashScope Router",
    description="OpenAI 兼容的多模态 API 中转服务",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 中间件：请求日志 + UA 白名单
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_and_ua_filter(request: Request, call_next):
    """记录每个请求的基本信息（含 UA），并校验 UA 白名单。"""
    user_agent = request.headers.get("user-agent", "")
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "请求: %s %s | IP=%s | UA=%s",
        method, path, client_ip, user_agent,
    )

    # UA 白名单校验
    if not check_ua_allowed(user_agent):
        logger.warning("UA 被拒绝: %s | IP=%s", user_agent, client_ip)
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": "Forbidden: User-Agent not allowed.",
                    "type": "permission_error",
                    "code": "ua_blocked",
                }
            },
        )

    start_ts = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_ts

    logger.info(
        "响应: %s %s | status=%d | %.2fs",
        method, path, response.status_code, elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# 注册路由
# ---------------------------------------------------------------------------
app.include_router(transcriptions_router)
app.include_router(speech_router)
app.include_router(models_router)


# ---------------------------------------------------------------------------
# 后台任务：定时 GC 回收，防止内存碎片累积
# ---------------------------------------------------------------------------
async def _periodic_gc():
    """根据配置的间隔定时执行 gc.collect()。"""
    cfg = get_config()
    interval = cfg.get("memory", {}).get("gc_interval_seconds", 300)
    logger.info("定时 GC 已启动, 间隔=%d 秒", interval)
    while True:
        await asyncio.sleep(interval)
        collected = gc.collect()
        if collected:
            logger.debug("定时 GC: 回收了 %d 个对象", collected)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_periodic_gc())


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
