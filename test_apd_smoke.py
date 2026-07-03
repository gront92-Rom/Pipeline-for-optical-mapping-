#!/usr/bin/env python3
"""
Smoke-тесты для utils/signal.py и APDAgent.
Запуск: python3 test_apd_smoke.py
"""

import sys
import numpy as np

# Добавляем src в путь
sys.path.insert(0, "src")

from cardiac_pipeline.utils.signal import (
    masked_spatial_pool,
    find_upstroke_start,
    find_repol_crossing_with_fallback,
    get_4_corners_snapped,
    validate_apd_semantics,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS  {name}" + (f"  [{detail}]" if detail else ""))
        PASS += 1
    else:
        print(f"  ❌ FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        FAIL += 1

print("=== Smoke Tests: utils/signal.py + APDAgent ===\n")

# -----------------------------------------------------------------------
# 1. masked_spatial_pool
# -----------------------------------------------------------------------
print("1. masked_spatial_pool()")
T, H, W = 100, 32, 32
video = np.random.rand(T, H, W).astype(np.float32)
mask = np.zeros((H, W), dtype=bool)
mask[8:24, 8:24] = True

roi = masked_spatial_pool(video, mask, size=3)
check("shape preserved", roi.shape == (T, H, W), f"{roi.shape}")
check("dtype float32", roi.dtype == np.float32)
check("outside mask is zero", np.all(roi[:, 0, 0] == 0.0))
check("inside mask is non-zero", np.any(roi[:, 16, 16] != 0.0))

# -----------------------------------------------------------------------
# 2. find_upstroke_start
# -----------------------------------------------------------------------
print("\n2. find_upstroke_start()")
fps = 1000.0
# Синтетический трейс: пик в 100, апстрок с 60 по 100 (0→1), плато после
trace = np.zeros(200, dtype=np.float32)
trace[60:100] = np.linspace(0, 1, 40)  # апстрок
trace[100:] = 1.0                       # плато/плавный спад

peak = 100
amp = 1.0

up_idx, frac, tag = find_upstroke_start(trace, peak, amp, fps, primary_frac=0.30)
check("upstroke found", up_idx is not None, f"up_idx={up_idx}, tag={tag}")
check("upstroke before peak", up_idx is not None and up_idx < peak)
check("tag contains upstroke", "upstroke" in (tag or ""))

# Нет амплитуды
up_idx2, _, tag2 = find_upstroke_start(trace, peak, 0.0, fps)
check("no_amp returns None", up_idx2 is None, f"tag={tag2}")

# -----------------------------------------------------------------------
# 3. find_repol_crossing_with_fallback
# -----------------------------------------------------------------------
print("\n3. find_repol_crossing_with_fallback()")
# Синтетический трейс: пик в 50, затем экспоненциальный спад
trace2 = np.zeros(300, dtype=np.float32)
t = np.arange(250)
trace2[50:300] = np.exp(-t / 80.0).astype(np.float32)
peak2 = 50
amp2 = float(trace2[peak2])

cross80, found80, status80 = find_repol_crossing_with_fallback(
    trace2, peak2, amp2, fps=1000.0, threshold=80, next_peak_idx=250, total_frames=300
)
check("APD80 found", found80, f"cross={cross80:.2f}, status={status80}")
check("APD80 > peak", cross80 is not None and cross80 > peak2)

cross50, found50, _ = find_repol_crossing_with_fallback(
    trace2, peak2, amp2, fps=1000.0, threshold=50, next_peak_idx=250, total_frames=300
)
check("APD50 found", found50)
check("APD80 > APD50", cross80 is not None and cross50 is not None and cross80 > cross50,
      f"APD80={cross80:.2f}, APD50={cross50:.2f}")

# Нет амплитуды
_, found_na, tag_na = find_repol_crossing_with_fallback(trace2, peak2, 0.0, fps=1000.0)
check("no_amp returns not_found", not found_na, f"tag={tag_na}")

# -----------------------------------------------------------------------
# 4. get_4_corners_snapped
# -----------------------------------------------------------------------
print("\n4. get_4_corners_snapped()")
mask2 = np.zeros((64, 64), dtype=bool)
mask2[10:54, 10:54] = True

corners = get_4_corners_snapped(mask2, padding=5)
check("returns 4 corners", len(corners) == 4, f"n={len(corners)}")
check("all corners have label/y/x", all("label" in c and "y" in c and "x" in c for c in corners))
check("all corners inside mask", all(mask2[c["y"], c["x"]] for c in corners))

labels = [c["label"] for c in corners]
check("all 4 labels present", set(labels) == {"Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"})

# Пустая маска
empty_corners = get_4_corners_snapped(np.zeros((32, 32), dtype=bool))
check("empty mask returns []", empty_corners == [])

# -----------------------------------------------------------------------
# 5. validate_apd_semantics
# -----------------------------------------------------------------------
print("\n5. validate_apd_semantics()")
v, r = validate_apd_semantics(150.0, 80.0, "A")
check("VSD in range → PASS", v == "PASS", r)

v, r = validate_apd_semantics(350.0, 80.0, "A")
check("VSD out of range → FAIL", v == "FAIL", r)

v, r = validate_apd_semantics(200.0, 80.0, "B")
check("CaT in range → FAIL (default range)", v == "PASS", r)

v, r = validate_apd_semantics(50.0, 80.0, "A")
check("APD80 < APD30 → FAIL", v == "FAIL", r)

v, r = validate_apd_semantics(float("nan"), 80.0, "A")
check("NaN → FAIL", v == "FAIL", r)

v, r = validate_apd_semantics(150.0, 0.0, "A")
check("APD30=0 → FAIL", v == "FAIL", r)

# -----------------------------------------------------------------------
# 6. APDAgent импорт
# -----------------------------------------------------------------------
print("\n6. APDAgent import")
try:
    from cardiac_pipeline.agents.apd_agent import APDAgent
    check("APDAgent imports OK", True)
    from cardiac_pipeline.base_agent import PipelineConfig
    agent = APDAgent("test_001", config=PipelineConfig())
    check("APDAgent instantiates", True)
    check("has run() method", callable(getattr(agent, "run", None)))
    check("has _get_fps() method", callable(getattr(agent, "_get_fps", None)))
    check("has _get_dye() method", callable(getattr(agent, "_get_dye", None)))
except Exception as e:
    check("APDAgent imports OK", False, str(e))

# -----------------------------------------------------------------------
# Итог
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Результат: {PASS}/{PASS+FAIL} тестов прошло")
if FAIL == 0:
    print("Все тесты прошли ✅")
else:
    print(f"FAILED: {FAIL} тест(а) ❌")
    sys.exit(1)
