import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

# OmegaConf support (optional but recommended)
try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = dict  # fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """Load configuration from YAML using OmegaConf (with fallback)."""
    default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    
    if config_path is None:
        config_path = default_path
    
    config_path = Path(config_path)
    
    if OMEGACONF_AVAILABLE:
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
        else:
            cfg = OmegaConf.create({})
        # Merge with defaults if needed
        return OmegaConf.create(cfg)  # returns DictConfig
    else:
        # Simple fallback without OmegaConf
        import yaml
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}


class PipelineConfig:
    """
    Конфигурация пайплайна.
    Поддерживает OmegaConf DictConfig + простой dict fallback.
    """
    def __init__(self, cfg: Optional[Union[DictConfig, dict]] = None):
        if cfg is None:
            cfg = load_config()
        
        if OMEGACONF_AVAILABLE and isinstance(cfg, DictConfig):
            self._cfg = cfg
        else:
            self._cfg = OmegaConf.create(cfg) if OMEGACONF_AVAILABLE else cfg
        
        # Основные поля с дефолтами
        self.results_root = Path(self._get("results_root", "results"))
        self.data_root = Path(self._get("data_root", "data"))
        # NOTE: fps is NEVER a silent default — metadata_extractor raises if not found.
        # pixel_size_mm: canonical for MiCAM ULTIMA ×10 (from .bvx or fallback)
        self.pixel_size_mm = float(self._get("pixel_size_mm", 0.85))
        
        # Вложенные секции
        self.loader = self._get("loader", {})
        self.mask = self._get("mask", {})
        self.preprocess = self._get("preprocess", {})
        self.activation = self._get("activation", {})
        self.apd = self._get("apd", {})
        self.conduction = self._get("conduction", {})
        self.alternans = self._get("alternans", {})
        self.qc = self._get("qc", {})

    def _get(self, key: str, default: Any = None) -> Any:
        if OMEGACONF_AVAILABLE and hasattr(self._cfg, key):
            val = getattr(self._cfg, key, default)
            return OmegaConf.to_container(val) if isinstance(val, DictConfig) else val
        elif isinstance(self._cfg, dict):
            return self._cfg.get(key, default)
        return default

    def to_dict(self) -> dict:
        if OMEGACONF_AVAILABLE:
            return OmegaConf.to_container(self._cfg, resolve=True)
        return dict(self._cfg) if isinstance(self._cfg, dict) else {}


class BaseAgent(ABC):
    """
    Базовый класс для всех агентов пайплайна.
    Обеспечивает:
    - работу с путями (must / debug)
    - сохранение результатов
    - загрузку конфигурации (OmegaConf)
    - единый интерфейс run()
    - lazy-механику (заготовка)
    """

    def __init__(self, sample_id: str, config: Optional[Union[PipelineConfig, dict, DictConfig]] = None):
        self.sample_id = sample_id
        
        if isinstance(config, PipelineConfig):
            self.config = config
        else:
            self.config = PipelineConfig(config)
        
        # Директории
        self.must_dir = Path(self.config.results_root) / sample_id / "must"
        self.debug_dir = Path(self.config.results_root) / sample_id / "debug"
        self.must_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{sample_id}")
        self.logger.info(f"Initialized {self.__class__.__name__} for sample {sample_id}")

    def get_path(self, filename: str, kind: Literal['must', 'debug'] = 'must') -> Path:
        base = self.must_dir if kind == 'must' else self.debug_dir
        return base / filename

    def save_must(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> Path:
        path = self.get_path(filename, 'must')
        
        if isinstance(data, np.ndarray):
            np.save(path, data)
        elif isinstance(data, (dict, list)):
            with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            np.save(path.with_suffix('.npy'), np.asarray(data))
        
        self.logger.info(f"[MUST] Saved: {path.name}")
        return path

    def save_debug(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> Path:
        path = self.get_path(filename, 'debug')
        if isinstance(data, np.ndarray):
            np.save(path, data)
        elif isinstance(data, (dict, list)):
            with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[DEBUG] Saved: {path.name}")
        return path

    def load_must(self, filename: str) -> Any:
        path = self.get_path(filename, 'must')
        npy_path = path if path.suffix == '.npy' else path.with_suffix('.npy')
        
        if npy_path.exists():
            return np.load(npy_path, allow_pickle=True)
        if path.exists() and path.suffix == '.json':
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"MUST file not found: {path} / {npy_path}")

    def exists(self, filename: str, kind: Literal['must', 'debug'] = 'must') -> bool:
        path = self.get_path(filename, kind)
        return path.exists() or path.with_suffix('.npy').exists()

    def ensure_upstream(self, upstream_class, **kwargs):
        """Заготовка для lazy-вызова предыдущих агентов."""
        self.logger.debug(f"ensure_upstream called for {upstream_class.__name__} (not implemented yet)")
        # Пример реализации будет в конкретных агентах

    @abstractmethod
    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Главный метод. Должен быть реализован в наследниках."""
        pass

    def _log_metrics(self, metrics: Dict[str, Any]):
        metrics_path = self.must_dir / "metrics.json"
        try:
            existing = {}
            if metrics_path.exists():
                with open(metrics_path, encoding='utf-8') as f:
                    existing = json.load(f)
            existing.update(metrics)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to write metrics.json: {e}")


# Быстрый тест
if __name__ == "__main__":
    cfg = PipelineConfig()
    print("Config loaded successfully")
    print("pixel_size_mm:", cfg.pixel_size_mm)
    print("loader.crop_left:", cfg.loader.get("crop_left") if isinstance(cfg.loader, dict) else "N/A")
    
    class TestAgent(BaseAgent):
        def run(self, force=False, **kwargs):
            self.save_must(np.zeros((10, 100, 100)), "test_video.npy")
            return {"status": "ok", "pixel_size_mm": self.config.pixel_size_mm}
    
    agent = TestAgent("test_001", config=cfg)
    print(agent.run())
    print("BaseAgent + PipelineConfig test passed")
