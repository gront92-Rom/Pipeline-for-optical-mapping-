#!/usr/bin/env python3
"""
Демо-запуск APDAgent на синтетических данных крысиного желудочка.
Генерирует реалистичные трейсы (VSD, dye=A) и запускает полный Stage 4.
"""

import sys
import json
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")

from cardiac_pipeline.base_agent import PipelineConfig

# -----------------------------------------------------------------------
# Параметры синтетического эксперимента
# -----------------------------------------------------------------------
SAMPLE_ID = "demo_005A"
FPS       = 500.0      # Гц — типичная MiCAM ULTIMA
N_FRAMES  = 1500       # 3 секунды
H, W      = 48, 48
N_BEATS   = 7          # ~2.3 Гц стимуляция
DYE       = "A"        # VSD (вольтаж)

RESULTS_ROOT = Path("/home/ubuntu/pipeline/results_demo")

# -----------------------------------------------------------------------
# Генерация синтетического видео
# -----------------------------------------------------------------------
def make_ap_trace(n_frames, fps, beat_frames, apd80_ms=145.0, noise_std=0.04):
    """
    Синтетический трейс потенциала действия (VSD, уже инвертирован — пики вверх).
    Форма: быстрый апстрок (2 мс) + плато + экспоненциальная реполяризация.
    """
    trace = np.zeros(n_frames, dtype=np.float32)
    tau = apd80_ms / fps * 1000 / np.log(5)  # tau для APD80 ≈ 80% реполяризации

    for pk in beat_frames:
        # Апстрок: 2 кадра
        up = max(0, pk - 2)
        trace[up:pk] = np.linspace(0, 1, pk - up)
        # Плато + реполяризация
        end = min(n_frames, pk + int(apd80_ms / 1000 * fps * 2.5))
        t = np.arange(end - pk)
        trace[pk:end] = np.exp(-t / tau).astype(np.float32)

    trace += np.random.normal(0, noise_std, n_frames).astype(np.float32)
    return trace


rng = np.random.default_rng(42)

# Пики биений (равномерно, с небольшим джиттером)
beat_interval = N_FRAMES // (N_BEATS + 1)
beat_frames = np.array([
    beat_interval * (i + 1) + rng.integers(-5, 5)
    for i in range(N_BEATS)
])

# Маска: эллипс в центре
mask = np.zeros((H, W), dtype=bool)
cy, cx = H // 2, W // 2
for y in range(H):
    for x in range(W):
        if ((y - cy) / (H * 0.38))**2 + ((x - cx) / (W * 0.38))**2 <= 1:
            mask[y, x] = True

# Видео: каждый пиксель под маской — AP-трейс с пространственным разбросом APD80
video = np.zeros((N_FRAMES, H, W), dtype=np.float32)
for y in range(H):
    for x in range(W):
        if mask[y, x]:
            # Пространственный градиент APD80: 120–180 мс (реалистичная дисперсия)
            apd80_local = 120.0 + 60.0 * (y / H)
            amp_local   = 0.8 + 0.4 * rng.random()
            video[:, y, x] = amp_local * make_ap_trace(
                N_FRAMES, FPS, beat_frames, apd80_ms=apd80_local,
                noise_std=0.03 + 0.02 * rng.random()
            )

print(f"Синтетическое видео: {video.shape}, маска: {mask.sum()} пикселей, биений: {N_BEATS}")

# -----------------------------------------------------------------------
# Подготовка директорий (имитация BaseAgent-структуры)
# -----------------------------------------------------------------------
must_dir  = RESULTS_ROOT / SAMPLE_ID / "must"
debug_dir = RESULTS_ROOT / SAMPLE_ID / "debug"
must_dir.mkdir(parents=True, exist_ok=True)
debug_dir.mkdir(parents=True, exist_ok=True)

# Сохраняем входные артефакты в нужные места
np.save(debug_dir / "preproc_video_apd.npy", video)
np.save(must_dir  / "peaks.npy",             beat_frames.astype(np.int64))
np.save(must_dir  / "mask.npy",              mask)

metadata = {
    "fps":            FPS,
    "dye":            DYE,
    "sample_id":      SAMPLE_ID,
    "pixel_size_mm":  0.085,
    "n_frames":       N_FRAMES,
    "height":         H,
    "width":          W,
}
with open(must_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Входные данные сохранены.")

# -----------------------------------------------------------------------
# Запуск APDAgent
# -----------------------------------------------------------------------
cfg = PipelineConfig({"results_root": str(RESULTS_ROOT)})

from cardiac_pipeline.agents.apd_agent import APDAgent
agent = APDAgent(SAMPLE_ID, config=cfg)

print("\nЗапускаю APDAgent (Stage 4)...\n")
result = agent.run(force=True)

# -----------------------------------------------------------------------
# Читаем и выводим полный отчёт
# -----------------------------------------------------------------------
report_path = must_dir / "apd_report.json"
with open(report_path) as f:
    report = json.load(f)

print("\n" + "="*60)
print("ПОЛНЫЙ ОТЧЁТ APD (apd_report.json)")
print("="*60)
print(json.dumps(report, indent=2, ensure_ascii=False))
print("="*60)
print(f"\nФайлы сохранены в: {RESULTS_ROOT / SAMPLE_ID}")
