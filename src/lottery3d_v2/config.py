# -*- coding: utf-8 -*-
"""配置加载与管理"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).parent.parent.parent / "configs"


class ConfigManager:
    """配置管理器，支持分层加载与缓存"""

    def __init__(self, config_root: Path | str | None = None):
        self.config_root = Path(config_root) if config_root else CONFIG_ROOT
        self._cache: dict[str, dict] = {}

    def load(self, *parts: str) -> dict[str, Any]:
        """加载配置文件，支持嵌套路径
        例: load("predictors", "markov") -> configs/predictors/markov.yaml
        """
        key = "/".join(parts)
        if key in self._cache:
            return self._cache[key]

        path = self.config_root / Path(*parts).with_suffix(".yaml")
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[key] = data
        return data

    def load_base(self) -> dict[str, Any]:
        return self.load("base")

    def load_predictor(self, name: str) -> dict[str, Any]:
        return self.load("predictors", name)

    def load_ensemble(self, name: str = "full") -> dict[str, Any]:
        data = self.load("ensemble", "ensemble")
        return data.get(name, {})

    def load_calibration(self) -> dict[str, Any]:
        return self.load("calibration", "calibration")

    def clear_cache(self) -> None:
        self._cache.clear()


@lru_cache(maxsize=1)
def get_config() -> ConfigManager:
    """获取全局配置管理器单例"""
    return ConfigManager()


def merge_configs(base: dict, override: dict) -> dict:
    """深度合并配置，override 优先"""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result