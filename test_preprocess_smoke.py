#!/usr/bin/env python3
"""
test_preprocess_smoke.py — Smoke-тесты для preprocess.py v6.

Проверяет:
  1. Синтаксис и импорт
  2. should_invert() — все ветки
  3. spatial_smooth() — форма и тип
  4. temporal_lowpass() — форма, тип, dual-stage cutoffs
  5. asls_baseline_correct_trace() — убирает линейный дрейф
  6. preprocess_video() activation path (без ASLS)
  7. preprocess_video() apd path (без ASLS)
  8. preprocess_video() с do_asls=True
  9. Порядок операций: инверсия ДО ASLS (пики смотрят вверх)
 10. Ошибка при do_asls=True без mask
"""

import sys
import numpy as np

sys.path.insert(0, "src")

from cardiac_pipeline.utils.preprocess import (
    should_invert,
    spatial_smooth,
    temporal_lowpass,
    asls_baseline_correct_trace,
    asls_baseline_correct,
    normalize_traces,
    preprocess_video,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))

print("\n=== Smoke Tests: preprocess.py v6 ===\n")

# --- 1. should_invert ---
print("1. should_invert()")
check("recording_mode='voltage' → True",  should_invert(recording_mode="voltage") is True)
check("recording_mode='calcium' → False", should_invert(recording_mode="calcium") is False)
check("dye='A' → True",                   should_invert(dye="A") is True)
check("dye='B' → False",                  should_invert(dye="B") is False)
check("sample_name='005A' → True",        should_invert(sample_name="005A") is True)
check("sample_name='005B' → False",       should_invert(sample_name="005B") is False)
check("no args → True (default VSD)",     should_invert() is True)

# --- 2. spatial_smooth ---
print("\n2. spatial_smooth()")
vid = np.random.rand(100, 32, 32).astype(np.float32)
out = spatial_smooth(vid, sigma=2.0)
check("shape preserved", out.shape == vid.shape, str(out.shape))
check("dtype float32", out.dtype == np.float32)

# --- 3. temporal_lowpass ---
print("\n3. temporal_lowpass()")
out_act = temporal_lowpass(vid, fps=666.67, cutoff=80.0)
out_apd = temporal_lowpass(vid, fps=666.67, cutoff=150.0)
check("activation cutoff: shape OK", out_act.shape == vid.shape)
check("apd cutoff: shape OK",        out_apd.shape == vid.shape)
check("dtype float32", out_act.dtype == np.float32)
# 150 Hz filter should preserve more high-freq content than 80 Hz
std_act = float(np.std(out_act))
std_apd = float(np.std(out_apd))
check("apd (150Hz) has more variance than activation (80Hz)",
      std_apd > std_act, f"std_apd={std_apd:.4f} std_act={std_act:.4f}")

# --- 4. asls_baseline_correct_trace ---
print("\n4. asls_baseline_correct_trace()")
t = np.linspace(0, 1, 300)
drift = 5.0 * t  # линейный дрейф
signal = np.sin(2 * np.pi * 5 * t) + drift
corrected = asls_baseline_correct_trace(signal.astype(np.float64))
# После коррекции среднее должно быть близко к 0
residual_drift = np.polyfit(t, corrected, 1)[0]  # наклон линейного тренда
check("linear drift removed (slope < 0.5)", abs(residual_drift) < 0.5,
      f"slope={residual_drift:.3f}")

# --- 5. preprocess_video activation path ---
print("\n5. preprocess_video() — activation path")
vid_small = np.random.rand(200, 16, 16).astype(np.float32)
out = preprocess_video(vid_small, fps=500.0, target_stage="activation", dye="A")
check("shape preserved", out.shape == vid_small.shape)
check("dtype float32", out.dtype == np.float32)
# После инверсии (dye='A') среднее должно быть отрицательным (исходный rand > 0)
check("inversion applied (mean < 0)", float(out.mean()) < 0,
      f"mean={float(out.mean()):.4f}")

# --- 6. preprocess_video apd path ---
print("\n6. preprocess_video() — apd path")
out_apd = preprocess_video(vid_small, fps=500.0, target_stage="apd", dye="B")
check("shape preserved", out_apd.shape == vid_small.shape)
# dye='B' → no inversion → mean should be positive
check("no inversion for CaT (mean > 0)", float(out_apd.mean()) > 0,
      f"mean={float(out_apd.mean()):.4f}")

# --- 7. preprocess_video with do_asls ---
print("\n7. preprocess_video() — do_asls=True")
mask = np.ones((16, 16), dtype=bool)
out_asls = preprocess_video(vid_small, fps=500.0, target_stage="activation",
                             dye="A", mask=mask, do_asls=True)
check("shape preserved with ASLS", out_asls.shape == vid_small.shape)
check("dtype float32 with ASLS", out_asls.dtype == np.float32)

# --- 8. Порядок операций: инверсия ДО ASLS ---
print("\n8. Порядок: invert BEFORE ASLS")
# Создаём сигнал, который идёт вниз (VSD-like) с дрейфом
t2 = np.linspace(0, 1, 200)
downward = -(np.sin(2 * np.pi * 3 * t2) + 1) - 2.0 * t2  # вниз + дрейф
vid_vsd = np.tile(downward[:, None, None], (1, 4, 4)).astype(np.float32)
mask_small = np.ones((4, 4), dtype=bool)
out_ordered = preprocess_video(vid_vsd, fps=500.0, target_stage="activation",
                                dye="A", mask=mask_small, do_asls=True,
                                sigma=0.0)  # sigma=0 чтобы не менять форму
# После инверсии + ASLS пики должны смотреть вверх → max > 0
check("after invert+ASLS: max > 0 (peaks up)", float(out_ordered.max()) > 0,
      f"max={float(out_ordered.max()):.4f}")

# --- 9. Ошибка при do_asls без mask ---
print("\n9. do_asls=True без mask → ValueError")
try:
    preprocess_video(vid_small, fps=500.0, do_asls=True)
    check("raises ValueError", False, "Исключение НЕ было брошено!")
except ValueError as e:
    check("raises ValueError", True, str(e)[:60])

# --- Итог ---
print("\n" + "=" * 45)
passed = sum(1 for _, s, _ in results if s == PASS)
total  = len(results)
print(f"Результат: {passed}/{total} тестов прошло")
if passed < total:
    print("\nПровалившиеся тесты:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  {status}  {name}  [{detail}]")
    sys.exit(1)
else:
    print("Все тесты прошли ✅")
    sys.exit(0)
