# -*- coding: utf-8 -*-
"""配置加载与管理 - 支持 Pydantic 验证"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .config_schema import GlobalConfig, validate_config


CONFIG_ROOT = Path(__file__).parent.parent.parent / "configs"


class ConfigManager:
    """配置管理器，支持分层加载、缓存与 Pydantic 验证"""

    def __init__(self, config_root: Path | str | None = None):
        self.config_root = Path(config_root) if config_root else CONFIG_ROOT
        self._cache: dict[str, dict] = {}
        self._validated_cache: dict[str, GlobalConfig] = {}
        logger.debug(f"ConfigManager initialized with root: {self.config_root}")

    def load(self, *parts: str) -> dict[str, Any]:
        """加载配置文件，支持嵌套路径
        例：load("predictors", "markov") -> configs/predictors/markov.yaml
        """
        key = "/".join(parts)
        if key in self._cache:
            return self._cache[key]

        path = self.config_root / Path(*parts).with_suffix(".yaml")
        if not path.exists():
            logger.warning(f"配置文件不存在：{path}")
            raise FileNotFoundError(f"配置文件不存在：{path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded config from {path}: {len(data)} keys")
            self._cache[key] = data
            return data
        except yaml.YAMLError as e:
            logger.error(f"YAML 解析错误 {path}: {e}")
            raise
        except Exception as e:
            logger.error(f"读取配置文件失败 {path}: {e}")
            raise

    def load_validated(self, *parts: str) -> GlobalConfig:
        """加载并验证配置文件
        
        Args:
            *parts: 配置文件路径部分
            
        Returns:
            GlobalConfig: 验证后的配置模型
            
        Raises:
            ValidationError: 当配置不合法时
        """
        key = "/".join(parts)
        if key in self._validated_cache:
            return self._validated_cache[key]
        
        raw_config = self.load(*parts)
        validated = validate_config(raw_config)
        self._validated_cache[key] = validated
        
        logger.info(f"Validated config: {key}")
        return validated

    def load_base(self) -> dict[str, Any]:
        return self.load("base")

    def load_predictor(self, name: str) -> dict[str, Any]:
        try:
            return self.load("predictors", name)
        except FileNotFoundError:
            logger.warning(f"预测器配置未找到：{name}，使用默认配置")
            return {}

    def load_ensemble(self, name: str = "full") -> dict[str, Any]:
        try:
            data = self.load("ensemble", "ensemble")
            result = data.get(name, {})
            if not result:
                logger.warning(f"集成模型配置未找到：{name}")
            return result
        except FileNotFoundError:
            logger.warning("集成配置文件未找到")
            return {}

    def load_calibration(self) -> dict[str, Any]:
        try:
            return self.load("calibration", "calibration")
        except FileNotFoundError:
            logger.warning("校准配置文件未找到")
            return {}

    def clear_cache(self) -> None:
        self._cache.clear()
        self._validated_cache.clear()
        logger.debug("Config cache cleared")


@lru_cache(maxsize=1)
def get_config() -> ConfigManager:
    """获取全局配置管理器单例"""
    logger.debug("Creating ConfigManager singleton")
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
