"""
配置加载器 - 从 config.yaml 读取并提供全局配置
"""

import fnmatch
import logging
import os
import pathlib
from typing import Any

import yaml

logger = logging.getLogger("dashscope-router")

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: dict[str, Any] = {
    "ua_whitelist": {
        "enabled": False,
        "rules": [],
    },
    "memory": {
        "gc_interval_seconds": 300,
    },
}

# ---------------------------------------------------------------------------
# 全局配置实例
# ---------------------------------------------------------------------------
_config: dict[str, Any] = {}


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """加载 config.yaml，合并默认值后缓存到全局。"""
    global _config

    if path is None:
        # 默认：项目根目录下的 config.yaml
        path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        path = pathlib.Path(path)

    cfg = dict(_DEFAULT_CONFIG)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # 浅合并顶层 key
        for key, val in user_cfg.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
                cfg[key] = {**cfg[key], **val}
            else:
                cfg[key] = val
        logger.info("配置已从 %s 加载", path)
    else:
        logger.warning("配置文件 %s 不存在，使用默认配置", path)

    _config = cfg
    return _config


def get_config() -> dict[str, Any]:
    """获取已加载的配置（若尚未加载则自动加载）。"""
    if not _config:
        load_config()
    return _config


def check_ua_allowed(user_agent: str | None) -> bool:
    """
    检查 User-Agent 是否在白名单内。

    - 白名单未启用 → 始终返回 True
    - UA 为空且白名单启用 → 返回 False
    - 使用 fnmatch 通配符匹配（不区分大小写）
    """
    cfg = get_config().get("ua_whitelist", {})
    if not cfg.get("enabled", False):
        return True

    rules: list[str] = cfg.get("rules", [])
    if not rules:
        return True  # 没有配置任何规则，视为不启用

    if not user_agent:
        return False

    ua_lower = user_agent.lower()
    for rule in rules:
        if fnmatch.fnmatch(ua_lower, rule.lower()):
            return True

    return False
