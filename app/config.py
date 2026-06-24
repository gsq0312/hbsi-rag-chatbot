"""Configuration management module for RAG Service"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from app.models import RAGConfig


class ConfigManager:
    DEFAULT_CONFIG_PATH = Path("data/config.json")
    DEFAULT_CONFIG: Dict[str, Any] = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k_retrieval": 3,
        "model_name": "deepseek-chat",
        "max_tokens": 2000,
        "temperature": 0.7
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[RAGConfig] = None
        load_dotenv()
        self._load_or_create_config()

    def _load_or_create_config(self) -> None:
        if self.config_path.exists():
            self._load_from_file()
        else:
            self._create_default_config()

    def _load_from_file(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            self._config = RAGConfig(**config_data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid configuration file: {e}")

    def _create_default_config(self) -> None:
        self._config = RAGConfig(**self.DEFAULT_CONFIG)
        self.save()

    def get_config(self) -> RAGConfig:
        if self._config is None:
            self._load_or_create_config()
        return self._config

    def update_config(self, **kwargs) -> RAGConfig:
        current_config = self.get_config()
        current_dict = current_config.model_dump()
        for key, value in kwargs.items():
            if key in current_dict:
                current_dict[key] = value
        self._config = RAGConfig(**current_dict)
        return self._config

    def save(self) -> None:
        if self._config is None:
            raise ValueError("No configuration to save")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config.model_dump(), f, indent=2, ensure_ascii=False)

    def reset_to_default(self) -> RAGConfig:
        self._config = RAGConfig(**self.DEFAULT_CONFIG)
        self.save()
        return self._config

    def reload(self) -> RAGConfig:
        self._load_or_create_config()
        return self.get_config()


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> RAGConfig:
    return get_config_manager().get_config()
