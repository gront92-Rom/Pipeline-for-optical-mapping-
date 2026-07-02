#!/usr/bin/env python3
"""
Smoke-test для conduction_consensus_agent.py (CLI-режим).
Запускать из корня репозитория: python3 test_consensus_agent_cli.py
"""
import sys
import os
import subprocess
import tempfile
import json
import numpy as np

sys.path.insert(0, "src")

SCRIPT = "src/cardiac_pipeline/agents/conduction_consensus_agent.py"

def make_synthetic_data(tmpdir, cv_true=0.5, pixel_size_mm=0.085, n_beats=3):
    """Создаёт синтетические per_beat_activation.npy и mask.npy."""
    H, W = 40, 50
    mask = np.ones((H, W), dtype=bool)
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False

    x_coords = np.arange(W)[np.newaxis, :] * np.ones((H, 1))
    base_map = x_coords * pixel_size_mm / cv_true  # мс

    # Небольшой шум между битами
    rng = np.random.default_rng(42)
    per_beat = np.stack([
        base_map + rng.normal(0, 0.01, (H, W)) for _ in range(n_beats)
    ])

    tat_path  = os.path.join(tmpdir, "per_beat_activation.npy")
    mask_path = os.path.join(tmpdir, "mask.npy")
    np.save(tat_path, per_beat)
    np.save(mask_path, mask)
    return tat_path, mask_path

# ---------------------------------------------------------------------------
# Тест 1: SUCCESS — нормальные данные
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    tat_path, mask_path = make_synthetic_data(tmpdir)
    out_dir = os.path.join(tmpdir, "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, tat_path, mask_path, out_dir,
         "--pixel-size", "0.085", "--tolerance", "0.15", "--qc-threshold", "0.20"],
        capture_output=True, text=True
    )
    print(f"[TEST 1 SUCCESS] exit={result.returncode}")
    print(result.stdout.strip())
    assert result.returncode == 0, f"Expected 0, got {result.returncode}\n{result.stderr}"
    assert os.path.exists(os.path.join(out_dir, "cvl_mean.npy")), "cvl_mean.npy missing"
    assert os.path.exists(os.path.join(out_dir, "cvl_sd.npy")),   "cvl_sd.npy missing"
    assert os.path.exists(os.path.join(out_dir, "cv_report.json")), "cv_report.json missing"
    report = json.load(open(os.path.join(out_dir, "cv_report.json")))
    assert report["verdict"] in ("PASS", "WARN"), f"Expected PASS/WARN, got {report['verdict']}"
    cv_mean = np.load(os.path.join(out_dir, "cvl_mean.npy"))
    mask = np.load(mask_path).astype(bool)
    valid = cv_mean[mask & np.isfinite(cv_mean)]
    median_cv = float(np.median(valid))
    err = abs(median_cv - 0.5) / 0.5
    assert err < 0.05, f"CV error too large: {err:.3f} (median={median_cv:.4f})"
    print(f"  median CV = {median_cv:.4f} m/s (expected ~0.5), error={err:.3f}  PASS")

# ---------------------------------------------------------------------------
# Тест 2: REJECT — exit code = 2 (BUG-3 fix)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    H, W = 40, 50
    mask = np.ones((H, W), dtype=bool)
    # Все NaN — ни один бит не пройдёт
    per_beat = np.full((3, H, W), np.nan)
    tat_path  = os.path.join(tmpdir, "per_beat_activation.npy")
    mask_path = os.path.join(tmpdir, "mask.npy")
    np.save(tat_path, per_beat)
    np.save(mask_path, mask)
    out_dir = os.path.join(tmpdir, "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, tat_path, mask_path, out_dir,
         "--pixel-size", "0.085"],
        capture_output=True, text=True
    )
    print(f"\n[TEST 2 REJECT all-NaN] exit={result.returncode}")
    print(result.stdout.strip())
    assert result.returncode == 2, f"Expected 2 (REJECT), got {result.returncode}"
    assert not os.path.exists(os.path.join(out_dir, "cvl_mean.npy")), "cvl_mean.npy should NOT exist on REJECT"
    print("  exit=2, cvl_mean.npy absent  PASS")

# ---------------------------------------------------------------------------
# Тест 3: REJECT — QC ниже порога (acceptance_rate < qc_threshold)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    H, W = 40, 50
    mask = np.ones((H, W), dtype=bool)
    # Очень маленький pixel_size → CV будет огромной → всё вне диапазона → acceptance=0
    per_beat = np.ones((3, H, W)) * 0.001  # почти нулевые времена → огромный градиент
    tat_path  = os.path.join(tmpdir, "per_beat_activation.npy")
    mask_path = os.path.join(tmpdir, "mask.npy")
    np.save(tat_path, per_beat)
    np.save(mask_path, mask)
    out_dir = os.path.join(tmpdir, "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, tat_path, mask_path, out_dir,
         "--pixel-size", "0.085", "--cv-max", "0.001"],  # очень узкий диапазон
        capture_output=True, text=True
    )
    print(f"\n[TEST 3 REJECT QC] exit={result.returncode}")
    print(result.stdout.strip())
    assert result.returncode == 2, f"Expected 2 (REJECT), got {result.returncode}\n{result.stderr}"
    print("  exit=2  PASS")

# ---------------------------------------------------------------------------
# Тест 4: CRASH — pixel-size не передан (BUG-2 fix: required=True)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    tat_path, mask_path = make_synthetic_data(tmpdir)
    out_dir = os.path.join(tmpdir, "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, tat_path, mask_path, out_dir],
        capture_output=True, text=True
    )
    print(f"\n[TEST 4 missing --pixel-size] exit={result.returncode}")
    # argparse выходит с кодом 2 при missing required arg
    assert result.returncode != 0, "Should fail without --pixel-size"
    print(f"  exit={result.returncode} (non-zero)  PASS")

print("\n=== ALL CLI SMOKE TESTS PASSED ===")
