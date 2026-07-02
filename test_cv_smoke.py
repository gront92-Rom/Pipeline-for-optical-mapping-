#!/usr/bin/env python3
"""
Smoke-test для cv_estimators и ConductionAgent (judge_conduction).
Запускать из корня репозитория: python3 test_cv_smoke.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from cardiac_pipeline.utils.cv_estimators import (
    compute_hybrid_structure_tensor,
    compute_polynomial_bayly,
    estimate_cv_stats,
)
from cardiac_pipeline.agents.conduction_agent import judge_conduction

# ---------------------------------------------------------------------------
# Синтетическая карта активации: плоская волна слева направо
# ---------------------------------------------------------------------------
H, W = 50, 60
mask = np.ones((H, W), dtype=bool)
mask[:3, :] = False
mask[-3:, :] = False
mask[:, :3] = False
mask[:, -3:] = False

pixel_size_mm = 0.085
CV_true = 0.5  # м/с = мм/мс

x_coords = np.arange(W)[np.newaxis, :] * np.ones((H, 1))
act_map = x_coords * pixel_size_mm / CV_true  # мс

# ---------------------------------------------------------------------------
# Тест 1: compute_hybrid_structure_tensor
# ---------------------------------------------------------------------------
cv1, ang1, coh1 = compute_hybrid_structure_tensor(act_map, mask, pixel_size_mm)
s1 = estimate_cv_stats(cv1, mask)
print(f"[hybrid_ST] median CV = {s1['cv_median_m_per_s']:.4f} m/s (expected ~{CV_true})")
print(f"            valid_fraction = {s1['valid_fraction']:.3f}")
err1 = abs(s1['cv_median_m_per_s'] - CV_true) / CV_true
assert err1 < 0.05, f"hybrid_ST error too large: {err1:.3f}"
print("  PASS")

# ---------------------------------------------------------------------------
# Тест 2: compute_polynomial_bayly
# ---------------------------------------------------------------------------
cv2, ang2, coh2 = compute_polynomial_bayly(act_map, mask, pixel_size_mm, window_size=4.0)
s2 = estimate_cv_stats(cv2, mask)
print(f"[bayly]     median CV = {s2['cv_median_m_per_s']:.4f} m/s (expected ~{CV_true})")
print(f"            valid_fraction = {s2['valid_fraction']:.3f}")
err2 = abs(s2['cv_median_m_per_s'] - CV_true) / CV_true
assert err2 < 0.05, f"bayly error too large: {err2:.3f}"
print("  PASS")

# ---------------------------------------------------------------------------
# Тест 3: judge_conduction — PASS
# ---------------------------------------------------------------------------
verdict, reason, metrics = judge_conduction(cv1, mask, cv_min=0.05, cv_max=2.0, qc_threshold=0.20)
print(f"[judge PASS] verdict={verdict}, reason={reason}")
assert verdict in ("PASS", "WARN"), f"Expected PASS/WARN, got {verdict}"
print("  PASS")

# ---------------------------------------------------------------------------
# Тест 4: judge_conduction — REJECT (all NaN)
# ---------------------------------------------------------------------------
nan_map = np.full((H, W), np.nan)
verdict_r, reason_r, _ = judge_conduction(nan_map, mask, cv_min=0.05, cv_max=2.0, qc_threshold=0.20)
print(f"[judge REJECT] verdict={verdict_r}, reason={reason_r}")
assert verdict_r == "REJECT", f"Expected REJECT, got {verdict_r}"
print("  PASS")

# ---------------------------------------------------------------------------
# Тест 5: pixel_size_mm <= 0 → ValueError
# ---------------------------------------------------------------------------
try:
    compute_hybrid_structure_tensor(act_map, mask, pixel_size_mm=0.0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"[pixel_size=0 guard] ValueError: {e}")
    print("  PASS")

# ---------------------------------------------------------------------------
# Тест 6: estimate_cv_stats — пустая маска
# ---------------------------------------------------------------------------
empty_mask = np.zeros((H, W), dtype=bool)
stats_empty = estimate_cv_stats(cv1, empty_mask)
assert stats_empty["valid_pixels"] == 0
assert stats_empty["cv_median_m_per_s"] is None
print("[empty mask stats] PASS")

print("\n=== ALL SMOKE TESTS PASSED ===")
