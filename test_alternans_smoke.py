#!/usr/bin/env python3
"""
Smoke-тесты для utils/alternans.py и AlternansAgent.
Запуск: python3 test_alternans_smoke.py
"""

import sys
import numpy as np

sys.path.insert(0, "src")

from cardiac_pipeline.utils.alternans import (
    compute_spatial_alternans,
    compute_concordance_map,
    compute_temporal_spectrum,
    compute_poincare_correlation,
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

print("=== Smoke Tests: utils/alternans.py + AlternansAgent ===\n")

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# 1. compute_spatial_alternans
# -----------------------------------------------------------------------
print("1. compute_spatial_alternans()")

H, W, N = 32, 32, 8
mask = np.zeros((H, W), dtype=bool)
mask[8:24, 8:24] = True

# Синтетический 3D стек с выраженным альтернансом:
# чётные биения = 150 мс, нечётные = 120 мс → AC = 15 мс
apd3d = np.full((H, W, N), np.nan, dtype=np.float32)
for bi in range(N):
    val = 150.0 if bi % 2 == 0 else 120.0
    apd3d[mask, bi] = val + rng.normal(0, 0.5, mask.sum())

ac_ms, ac_pct, phase_map = compute_spatial_alternans(apd3d, mask, sign_floor_ms=0.5)

check("ac_ms shape", ac_ms.shape == (H, W), f"{ac_ms.shape}")
check("ac_pct shape", ac_pct.shape == (H, W))
check("phase_map shape", phase_map.shape == (H, W))
check("ac_ms inside mask > 0", np.nanmean(ac_ms[mask]) > 0,
      f"mean={np.nanmean(ac_ms[mask]):.2f}")
check("ac_ms ≈ 15 ms (half of 30 ms diff)", 12 < np.nanmedian(ac_ms[mask]) < 18,
      f"median={np.nanmedian(ac_ms[mask]):.2f}")
check("phase outside mask is NaN", np.all(np.isnan(phase_map[~mask])))
check("phase inside mask is +1 or -1 or NaN",
      np.all((phase_map[mask] == 1) | (phase_map[mask] == -1) | np.isnan(phase_map[mask])))

# Нет альтернанса: все биения одинаковы
apd3d_flat = np.full((H, W, N), 150.0, dtype=np.float32)
apd3d_flat[~mask] = np.nan
ac_ms_flat, _, phase_flat = compute_spatial_alternans(apd3d_flat, mask, sign_floor_ms=0.5)
check("no alternans → ac_ms ≈ 0", np.nanmedian(ac_ms_flat[mask]) < 0.1,
      f"median={np.nanmedian(ac_ms_flat[mask]):.4f}")
check("no alternans → phase is NaN (below floor)", np.all(np.isnan(phase_flat[mask])))

# -----------------------------------------------------------------------
# 2. compute_concordance_map
# -----------------------------------------------------------------------
print("\n2. compute_concordance_map()")

# Полностью конкордантная фаза: все пиксели = +1
phase_concordant = np.full((H, W), np.nan, dtype=np.float32)
phase_concordant[mask] = 1.0
conc = compute_concordance_map(phase_concordant, mask)

check("concordance shape", conc.shape == (H, W))
check("full concordance → median ≈ 1.0",
      np.nanmedian(conc[mask]) > 0.85,
      f"median={np.nanmedian(conc[mask]):.3f}")
check("outside mask is NaN", np.all(np.isnan(conc[~mask])))

# Полностью дискордантная фаза: шахматный паттерн +1/-1
phase_discord = np.full((H, W), np.nan, dtype=np.float32)
for y in range(H):
    for x in range(W):
        if mask[y, x]:
            phase_discord[y, x] = 1.0 if (y + x) % 2 == 0 else -1.0
conc_discord = compute_concordance_map(phase_discord, mask)
check("checkerboard → concordance < 0.6",
      np.nanmedian(conc_discord[mask]) < 0.6,
      f"median={np.nanmedian(conc_discord[mask]):.3f}")

# -----------------------------------------------------------------------
# 3. compute_temporal_spectrum
# -----------------------------------------------------------------------
print("\n3. compute_temporal_spectrum()")

# Чистый альтернанс: синусоида на частоте Найквиста
n_beats = 16
t = np.arange(n_beats)
pure_alternans = np.sin(np.pi * t)  # f = 0.5 cycles/beat

spec, freqs, purity = compute_temporal_spectrum(np.diff(pure_alternans))
check("spec is not None", spec is not None)
check("freqs is not None", freqs is not None)
check("purity > 0.5 for pure alternans", purity > 0.5, f"purity={purity:.3f}")

# Случайный шум → низкая purity
noise = rng.normal(0, 1, 20)
_, _, purity_noise = compute_temporal_spectrum(np.diff(noise))
check("random noise → purity < 0.5", purity_noise < 0.5, f"purity={purity_noise:.3f}")

# Слишком короткий ряд
spec_short, freqs_short, purity_short = compute_temporal_spectrum(np.array([1.0]))
check("len=1 → None, None, 0.0",
      spec_short is None and freqs_short is None and purity_short == 0.0)

# -----------------------------------------------------------------------
# 4. compute_poincare_correlation
# -----------------------------------------------------------------------
print("\n4. compute_poincare_correlation()")

# Идеальный альтернанс: L-S-L-S...
alternating = np.array([150.0, 120.0] * 6, dtype=float)
corr_alt = compute_poincare_correlation(alternating)
check("perfect alternans → corr ≈ -1", corr_alt < -0.9, f"corr={corr_alt:.3f}")

# Монотонный тренд → положительная корреляция
monotone = np.linspace(100, 200, 12)
corr_mono = compute_poincare_correlation(monotone)
check("monotone trend → corr > 0.9", corr_mono > 0.9, f"corr={corr_mono:.3f}")

# Слишком короткий ряд
corr_short = compute_poincare_correlation(np.array([150.0]))
check("len=1 → corr = 0.0", corr_short == 0.0, f"corr={corr_short}")

# -----------------------------------------------------------------------
# 5. AlternansAgent импорт и инициализация
# -----------------------------------------------------------------------
print("\n5. AlternansAgent import")
try:
    from cardiac_pipeline.agents.alternans_agent import AlternansAgent
    from cardiac_pipeline.base_agent import PipelineConfig
    check("AlternansAgent imports OK", True)
    agent = AlternansAgent("test_001", config=PipelineConfig())
    check("AlternansAgent instantiates", True)
    check("has run() method",           callable(getattr(agent, "run", None)))
    check("has _get_dye() method",      callable(getattr(agent, "_get_dye", None)))
    check("min_beats from config",      agent.min_beats == 4, f"min_beats={agent.min_beats}")
    check("sign_floor_ms from config",  agent.sign_floor_ms == 0.5,
          f"sign_floor_ms={agent.sign_floor_ms}")
    check("discordant_threshold",       agent.discordant_threshold == 0.7,
          f"discordant_threshold={agent.discordant_threshold}")
except Exception as e:
    check("AlternansAgent imports OK", False, str(e))

# -----------------------------------------------------------------------
# Итог
# -----------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Результат: {PASS}/{PASS+FAIL} тестов прошло")
if FAIL == 0:
    print("Все тесты прошли ✅")
else:
    print(f"FAILED: {FAIL} тест(а) ❌")
    sys.exit(1)
