# src/config_loader.py
"""Утилита для загрузки YAML конфигурационных файлов."""
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Загрузка YAML-конфига в dict.
    
    Args:
        path: Путь к YAML файлу
        
    Returns:
        Словарь с содержимым YAML файла
        
    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если файл содержит некорректный YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

