#!/usr/bin/env python3
"""
base_agent.py — Базовый класс для всех агентов пайплайна cardiac optical mapping.

Особенности:
- Поддержка OmegaConf конфигов
- save_must / save_debug
- File contract система (MUST, DEBUG, META, EXTERNAL, INTERMEDIATE)
- Автоматическое создание output_dir
- Логирование + метрики
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from omegaconf import OmegaConf, DictConfig


class BaseAgent(ABC):
    """
    Базовый агент.

    Все агенты должны наследоваться от него и реализовывать:
        - run(self, **kwargs) -> Dict[str, Any]
    """

    # Классы файлов (file contracts)
    MUST_FILES: Set[str] = set()
    DEBUG_FILES: Set[str] = set()
    META_FILES: Set[str] = {"metadata.json", "config.yaml"}
    EXTERNAL_FILES: Set[str] = set()

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        output_dir: Optional[str | Path] = None,
        debug: bool = False,
    ):
        self.config = config or OmegaConf.create({})
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Any] = {}
        self.warnings: list[str] = []
        self.start_time = datetime.now()

    def save_must(self, name: str, obj: Any, subdir: str = "") -> Path:
        """Сохраняет MUST-файл (обязательный артефакт)."""
        path = self._get_path(name, subdir)
        self._save_object(path, obj)
        return path

    def save_debug(self, name: str, obj: Any, subdir: str = "debug") -> Optional[Path]:
        """Сохраняет DEBUG-файл (только если debug=True)."""
        if not self.debug:
            return None
        path = self._get_path(name, subdir)
        self._save_object(path, obj)
        return path

    def save_meta(self, name: str, obj: Any) -> Path:
        """Сохраняет META-файл (метаданные, конфиги)."""
        path = self._get_path(name)
        self._save_object(path, obj)
        return path

    def _get_path(self, name: str, subdir: str = "") -> Path:
        if subdir:
            (self.output_dir / subdir).mkdir(exist_ok=True)
            return self.output_dir / subdir / name
        return self.output_dir / name

    def _save_object(self, path: Path, obj: Any):
        if isinstance(obj, (dict, list)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        elif isinstance(obj, (np.ndarray)):
            np.save(path.with_suffix(".npy"), obj)
        elif hasattr(obj, "savefig"):  # matplotlib figure
            obj.savefig(path, dpi=150, bbox_inches="tight")
        else:
            # fallback
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(obj))

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Основной метод агента. Должен быть реализован в наследниках."""
        pass

    def finalize(self) -> Dict[str, Any]:
        """Вызывается в конце работы агента."""
        self.metrics["elapsed_s"] = (datetime.now() - self.start_time).total_seconds()
        self.metrics["warnings"] = self.warnings
        self.save_meta("run_metrics.json", self.metrics)
        return self.metrics

    def log_warning(self, msg: str):
        self.warnings.append(msg)
        print(f"⚠️  {msg}")


# Пример использования в наследнике
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import numpy as np

    class ExampleAgent(BaseAgent):
        MUST_FILES = {"result.npy"}

        def run(self, data: np.ndarray):
            result = data * 2
            self.save_must("result.npy", result)
            self.metrics["mean"] = float(result.mean())
            return {"status": "ok"}

    cfg = OmegaConf.create({"debug": True})
    agent = ExampleAgent(config=cfg, output_dir="test_output", debug=True)
    agent.run(np.random.randn(100, 100))
    print(agent.finalize())
