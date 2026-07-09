"""
MaskAgent v4.2 — Полностью рабочая версия

- Primary: Полная RSM non-phys логика (stage2 → stage3_bisect → stage4_cleanup → stage5_smoothing)
- Fallback: Каскад методов (bandpower + foreground) + судейство + финальное сглаживание
- Исправлены все критические замечания ревью (R2, FPS, геометрия, кэширование)

Исправления при интеграции (2026-07-02):
- crop_left/crop_right читаются из config.loader (не config.mask)
- stim_hz=None передаётся в bandpower_stim → добавлена защита + logger.warning
- run(): _prepare_data() вызывается до try/except raw_rsm (иначе metadata не загружена)
- run(): если raw_rsm не найден после LoaderAgent — graceful fallback вместо crash
- _log_metrics() вызывается перед return (как в v4.1)
- PREPROCESS_AVAILABLE флаг добавлен для явной проверки
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

try:
    from cardiac_pipeline.utils.preprocess import preprocess_video, should_invert
    PREPROCESS_AVAILABLE = True
except ImportError:
    preprocess_video = None
    should_invert = None
    PREPROCESS_AVAILABLE = False

try:
    from skimage.morphology import (
        remove_small_objects, binary_opening, binary_closing, disk
    )
    from skimage.measure import regionprops
    from scipy.ndimage import label, binary_fill_holes as scipy_fill_holes
    import cv2
    # Alias for convenience (skimage.morphology.binary_fill_holes was removed in 0.26)
    binary_fill_holes = scipy_fill_holes
    HEAVY_DEPS = True
except ImportError:
    HEAVY_DEPS = False


PERC_EX = 0.005
N_ROUGH = 7


class MaskAgent(BaseAgent):
    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)
        # crop параметры живут в config.loader (не в config.mask)
        loader_cfg = getattr(self.config, 'loader', {}) or {}
        self.crop_left = int(loader_cfg.get('crop_left', 20))
        self.crop_right = int(loader_cfg.get('crop_right', 8))

        self.mcfg = getattr(self.config, 'mask', {}) or {}
        self.CRITERIA_LOOSE = {
            'cov_lo':    float(self.mcfg.get('cov_lo', 0.30)),
            'cov_hi':    0.75,
            'solid_lo':  0.70,
            'extent_lo': 0.45,
            'compact_lo': 0.20,
        }
        self.CRITERIA_STRICT = {
            'cov_lo':    float(self.mcfg.get('cov_lo', 0.30)),
            'cov_hi':    float(self.mcfg.get('cov_hi', 0.35)),
            'solid_lo':  float(self.mcfg.get('solid_lo', 0.75)),
            'extent_lo': float(self.mcfg.get('extent_lo', 0.50)),
            'compact_lo': float(self.mcfg.get('compact_lo', 0.25)),
        }
        self.COV_FLOOR = float(self.mcfg.get('cov_floor', 0.35))
        self.metadata: Dict[str, Any] = {}
        self.raw_video: Optional[np.ndarray] = None

    # ==================== ПОДГОТОВКА ====================

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.logger.warning("metadata.json not found — fps/dye/stim_hz will be missing")
        return self.metadata

    def _get_fps(self) -> float:
        fps = self.metadata.get("fps")
        if fps is None:
            raise ValueError(
                "fps отсутствует в metadata.json. "
                "LoaderAgent должен сохранить его заранее."
            )
        return float(fps)

    def _load_and_crop_video(self) -> np.ndarray:
        video = self.load_must("raw_video.npy")
        # Кроп применяется только если ширина == 128 (MiCAM ULTIMA стандарт)
        if video.ndim == 3 and video.shape[2] == 128:
            video = video[:, :, self.crop_left : self.crop_left + (128 - self.crop_left - self.crop_right)]
        self.raw_video = video
        return video

    def _prepare_data(self) -> None:
        self._load_metadata()
        self._load_and_crop_video()

    # ==================== PRIMARY: RSM non-phys v3 (full port from standalone) ====================

    def build_mask(self, bg: np.ndarray, thr: float) -> np.ndarray:
        if not HEAVY_DEPS:
            return (bg > thr).astype(bool)
        m = bg > thr
        m = remove_small_objects(m, max_size=int(np.ceil(PERC_EX * bg.size)))
        m = binary_fill_holes(m)
        if m.any():
            lbl, ncc = label(m)
            if ncc > 0:
                sizes = np.bincount(lbl.ravel())[1:]
                m = (lbl == np.argmax(sizes) + 1)
        return m.astype(bool)

    def morph_metrics(self, m: np.ndarray) -> Optional[Dict]:
        """Full morph metrics — matches rsm_mask_worker_v3_nophys.py."""
        if not HEAVY_DEPS or m.sum() == 0:
            return None
        try:
            from skimage.morphology import convex_hull_image
            p = regionprops(m.astype(int))[0]
            area, perim = p.area, p.perimeter
            conv = convex_hull_image(m)
            conv_perim = regionprops(conv.astype(int))[0].perimeter
            filled = binary_fill_holes(m)
            holes_pixels = filled & ~m
            if holes_pixels.any():
                _, n_holes = label(holes_pixels)
            else:
                n_holes = 0
            area_filled = int(filled.sum())
            fill_ratio = area / area_filled if area_filled > 0 else 1.0
            return {
                'coverage':       float(m.mean()),
                'solidity':       float(p.solidity),
                'extent':         float(p.extent),
                'eccentricity':   float(p.eccentricity),
                'compactness':    float(4 * np.pi * area / (perim ** 2 + 1e-9)),
                'boundary_smooth': float(perim / conv_perim),
                'n_holes':        int(n_holes),
                'fill_ratio':     float(fill_ratio),
            }
        except Exception:
            return None

    def composite_score(self, morph: Optional[Dict]) -> float:
        """solidity × compactness, penalized ×0.5 if cov < COV_FLOOR (banana fix)."""
        if morph is None:
            return 0.0
        base = morph['solidity'] * morph['compactness']
        cov = morph.get('coverage', 1.0)
        if cov < self.COV_FLOOR:
            base *= 0.5
        return base

    def evaluate(self, bg: np.ndarray, thr: float, criteria: Dict) -> Dict:
        """Full evaluate with red/yellow system — matches standalone v3."""
        m = self.build_mask(bg, thr)
        morph = self.morph_metrics(m)
        if morph is None:
            return {'threshold': thr, 'mask': m, 'morph': None,
                    'verdict': 'REJECT', 'red': ['empty mask'], 'yellow': [],
                    'n_pixels': 0, 'n_yellow': 0, 'n_red': 1}
        yellow, red = [], []
        C = criteria
        cov = morph['coverage']
        if cov < 0.10:
            red.append(f'cov {cov:.3f}')
        elif cov < C['cov_lo']:
            yellow.append(f'cov {cov:.3f}<{C["cov_lo"]}')
        elif cov > C['cov_hi']:
            yellow.append(f'cov {cov:.3f}>{C["cov_hi"]}')
        if morph['solidity'] < 0.7:
            red.append(f"solid {morph['solidity']:.3f}")
        elif morph['solidity'] < C['solid_lo']:
            yellow.append(f"solid {morph['solidity']:.3f}<{C['solid_lo']}")
        # Topology checks
        if morph.get('n_holes', 0) >= 2:
            red.append(f"holes {morph['n_holes']}")
        elif morph.get('n_holes', 0) == 1:
            yellow.append(f"1 hole")
        if morph.get('fill_ratio', 1.0) < 0.90:
            yellow.append(f"fill_r {morph['fill_ratio']:.3f}<0.90")
        if morph['extent'] < 0.50:
            red.append(f"ext {morph['extent']:.3f}")
        elif morph['extent'] < C['extent_lo']:
            yellow.append(f"ext {morph['extent']:.3f}<{C['extent_lo']}")
        if morph['compactness'] < 0.30:
            red.append(f"comp {morph['compactness']:.3f}")
        elif morph['compactness'] < C['compact_lo']:
            yellow.append(f"comp {morph['compactness']:.3f}<{C['compact_lo']}")
        if len(red) >= 2:
            verdict = 'REJECT'
        elif len(red) == 1 or yellow:
            verdict = 'WARN'
        else:
            verdict = 'PASS'
        return {'threshold': thr, 'mask': m, 'morph': morph,
                'verdict': verdict, 'red': red, 'yellow': yellow,
                'n_pixels': int(m.sum()), 'n_yellow': len(yellow), 'n_red': len(red)}

    def score_strict(self, r: Dict) -> Tuple[float, float]:
        """3-tier pick logic: PASS > WARN > REJECT, tie-break by lowest cov (compactest)."""
        if r.get('morph') is None:
            return (-1, 0)
        if r['verdict'] == 'PASS':
            return (1000 - r['morph']['coverage'], r['threshold'])
        if r['verdict'] == 'WARN':
            return (500 - r['morph']['coverage'], r['threshold'])
        return (100 - r['morph']['coverage'], r['threshold'])

    def stage2_adaptive(self, bg: np.ndarray) -> List[Dict]:
        """7 evenly-spaced thresholds in [BG_min, BG_max], judged with LOOSE criteria."""
        lo, hi = float(bg.min()), float(bg.max())
        step = (hi - lo) / (N_ROUGH + 1)
        thresholds = [lo + (i + 1) * step for i in range(N_ROUGH)]
        return [self.evaluate(bg, t, self.CRITERIA_LOOSE) for t in thresholds]

    def stage3_bisect_interval(self, bg: np.ndarray, lo_t: float, hi_t: float,
                                level: int = 0) -> Dict:
        """Bisect on a specific [lo, hi] interval — 5 STRICT sub-thresholds, pick best.

        Used by multi-level orchestrator on different PASS intervals.
        """
        if hi_t - lo_t < 1:
            sub_thresholds = [lo_t]
        else:
            sub_thresholds = [lo_t + i * (hi_t - lo_t) / 4 for i in range(5)]
        results = [self.evaluate(bg, t, self.CRITERIA_STRICT) for t in sub_thresholds]
        best = max(results, key=self.score_strict)
        if best.get('morph'):
            self.logger.info(
                f"  [S3-L{level}] bisect [{lo_t:.0f}, {hi_t:.0f}] → "
                f"PICK t={best['threshold']:.0f} ({best['verdict']}) "
                f"cov={100*best['morph']['coverage']:.1f}%"
            )
        else:
            self.logger.info(
                f"  [S3-L{level}] bisect [{lo_t:.0f}, {hi_t:.0f}] → "
                f"PICK t={best['threshold']:.0f} (REJECT)"
            )
        return best

    def stage4_cleanup(self, bg: np.ndarray, refined: Dict) -> Optional[Dict]:
        """15 open/close operations, ranked by composite_score (with banana penalty).

        Returns FINAL winner (clean, highest score) or None if no clean candidate.
        Clean filter: n_holes==0 AND solidity>=0.85 AND compactness>=0.55.
        """
        if not HEAVY_DEPS or refined is None:
            return None
        base = refined['mask']
        operations = [('baseline', base)]
        for r in [1, 2, 3]:
            operations.append((f'open(r={r})', binary_opening(base, disk(r))))
            operations.append((f'close(r={r})', binary_closing(base, disk(r))))
        for r in [1, 2, 3]:
            operations.append((f'open+close(r={r})',
                               binary_closing(binary_opening(base, disk(r)), disk(r))))
            operations.append((f'close+open(r={r})',
                               binary_opening(binary_closing(base, disk(r)), disk(r))))
        operations.append(('open(r=2)+close(r=3)',
                           binary_closing(binary_opening(base, disk(2)), disk(3))))
        operations.append(('close(r=2)+open(r=3)',
                           binary_opening(binary_closing(base, disk(2)), disk(3))))

        results = []
        for name, m in operations:
            morph = self.morph_metrics(m)
            if morph is None:
                results.append({'op': name, 'mask': m, 'morph': None,
                                'score': 0, 'clean': False})
                continue
            score = self.composite_score(morph)
            # 6-metric clean filter (matching standalone v3)
            is_clean = (
                morph.get('n_holes', 0) == 0
                and morph['solidity'] >= 0.85
                and morph['compactness'] >= 0.55
            )
            results.append({'op': name, 'mask': m, 'morph': morph,
                            'score': score, 'clean': is_clean})

        valid = [r for r in results if r.get('morph') is not None]
        if not valid:
            return None
        valid_sorted = sorted(valid, key=lambda r: r['score'], reverse=True)
        clean = [r for r in valid_sorted if r['clean']]

        if clean:
            final = clean[0]
            self.logger.info(
                f"  [S4] FINAL (clean): {final['op']}  "
                f"({final['mask'].sum()} px, {100*final['morph']['coverage']:.1f}%)"
            )
            return final
        else:
            self.logger.info(f"  [S4] NO CLEAN candidates — fallback triggered")
            return None

    def stage5_smoothing(self, winner: Dict) -> Dict:
        """Contour-coordinate LPF smoothing (same as standalone v3 S5)."""
        if not HEAVY_DEPS:
            return winner
        base = (winner['mask'].astype(np.uint8) * 255)
        contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            from skimage.morphology import binary_closing as bc, disk as d
            smoothed = bc(winner['mask'], d(1))
            smoothed = binary_fill_holes(smoothed)
        else:
            main = max(contours, key=cv2.contourArea)
            if len(main) < 6:
                from skimage.morphology import binary_closing as bc, disk as d
                smoothed = bc(winner['mask'], d(1))
                smoothed = binary_fill_holes(smoothed)
            else:
                ws = min(15, max(3, (len(main) // 8) | 1))  # odd, scales with contour
                box = np.ones(ws, dtype=np.float32) / ws
                x = main[:, 0, 0].astype(np.float32)
                y = main[:, 0, 1].astype(np.float32)
                x_padded = np.pad(x, ws, mode='wrap')
                y_padded = np.pad(y, ws, mode='wrap')
                x_smooth = np.convolve(x_padded, box, mode='same')[ws:-ws]
                y_smooth = np.convolve(y_padded, box, mode='same')[ws:-ws]
                smooth_contour = np.stack((x_smooth, y_smooth), axis=1).astype(np.int32)
                new_mask = np.zeros_like(base)
                cv2.drawContours(new_mask, [smooth_contour], -1, 255, -1)
                smoothed = binary_fill_holes(new_mask > 0)
        morph_after = self.morph_metrics(smoothed)
        return {
            'mask': smoothed,
            'morph': morph_after,
            'op': winner.get('op', '') + ' + LPF',
        }

    def _primary_rsm_bg_pipeline(self, raw_rsm: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """Full v3 pipeline with multi-level orchestrator + skip rule.

        Flow:
          S2: 7-way LOOSE sweep
          Skip col 0 (always)
          Build intervals: L0=[last_PASS, first_WARN], Lk=[pass[n-k-1], pass[n-k]]
          For each level: S3 bisect → S4 cleanup → if clean: S5 smoothing → return
          If all levels fail: return None
        """
        if raw_rsm is None:
            return None, {"status": "no_rsm"}
        bg = raw_rsm[0].astype(float)
        # Кроп фона согласован с _load_and_crop_video
        if bg.ndim == 2 and bg.shape[1] == 128:
            bg = bg[:, self.crop_left : self.crop_left + (128 - self.crop_left - self.crop_right)]

        # ── Stage 2: 7-way LOOSE sweep ──────────────────────────────────────
        stage2_res = self.stage2_adaptive(bg)
        sorted_s2 = sorted(stage2_res, key=lambda r: r['threshold'])

        # Log S2 results
        n_pass_all = sum(1 for r in stage2_res if r['verdict'] == 'PASS')
        n_warn = sum(1 for r in stage2_res if r['verdict'] == 'WARN')
        n_rej = sum(1 for r in stage2_res if r['verdict'] == 'REJECT')
        self.logger.info(
            f"[S2] 7-way: PASS={n_pass_all} WARN={n_warn} REJECT={n_rej}"
        )
        for r in sorted_s2:
            cov = 100 * r['morph']['coverage'] if r.get('morph') else 0
            self.logger.info(
                f"  t={r['threshold']:7.1f}  {r['verdict']:6s}  "
                f"cov={cov:5.1f}%  px={r.get('n_pixels', 0)}"
            )

        # ── Skip rule: always skip col 0 ────────────────────────────────────
        pass_indices = [i for i, r in enumerate(sorted_s2) if r['verdict'] == 'PASS']
        skip_i = {0}
        skipped = [i for i in pass_indices if i in skip_i]
        pass_indices = [i for i in pass_indices if i not in skip_i]
        if skipped:
            self.logger.info(
                f"[ORCHESTRATOR] Skip rule: col {skipped} skipped → "
                f"PASS indices after skip: {pass_indices}"
            )
        else:
            self.logger.info(
                f"[ORCHESTRATOR] PASS indices (skip col 0 was not PASS): {pass_indices}"
            )

        # ── No PASS after skip — use highest-threshold non-empty WARN ───────
        if not pass_indices:
            self.logger.info("[ORCHESTRATOR] No S2 PASS after skip — using highest WARN")
            anchor = None
            for r in sorted(sorted_s2, key=lambda x: x['threshold'], reverse=True):
                if r.get('morph') is not None:
                    anchor = r
                    break
            if anchor is None:
                return None, {"status": "no_usable_s2"}
            s3_pick = anchor
            s4_winner = self.stage4_cleanup(bg, s3_pick)
            if s4_winner is not None:
                final = self.stage5_smoothing(s4_winner)
                mask = final['mask']
                morph = final.get('morph') or self.morph_metrics(mask)
                return mask, {
                    'method':      'rsm_bg_v3_port',
                    'coverage':    morph['coverage'] if morph else 0,
                    'solidity':    morph['solidity'] if morph else 0,
                    'n_holes':     morph.get('n_holes', 0) if morph else 0,
                    'compactness': morph.get('compactness', 0) if morph else 0,
                    's4_op':       s4_winner.get('op', ''),
                }
            return None, {"status": "no_clean_s4"}

        # ── Build intervals: L0=[last_PASS, first_WARN], Lk=[pass[n-k-1], pass[n-k]] ─
        n_pass = len(pass_indices)
        first_warn_t = None
        for r in sorted_s2[pass_indices[-1] + 1:]:
            if r['verdict'] in ('WARN', 'REJECT') and r.get('morph') is not None:
                first_warn_t = float(r['threshold'])
                break
        if first_warn_t is None:
            first_warn_t = float(bg.max())

        intervals = [(0, float(sorted_s2[pass_indices[-1]]['threshold']), first_warn_t)]
        for k in range(1, n_pass):
            right = pass_indices[n_pass - k]
            left = pass_indices[n_pass - k - 1] if (n_pass - k - 1) >= 0 else None
            lo = float(sorted_s2[left]['threshold']) if left is not None else float(bg.min())
            hi = float(sorted_s2[right]['threshold'])
            intervals.append((k, lo, hi))

        # ── Run orchestrator: S3 → S4 → S5 for each level ───────────────────
        for level, lo_t, hi_t in intervals:
            self.logger.info(
                f"[ORCHESTRATOR] Level {level}: interval = [{lo_t:.0f}, {hi_t:.0f}]"
            )
            s3_pick = self.stage3_bisect_interval(bg, lo_t, hi_t, level=level)
            s4_winner = self.stage4_cleanup(bg, s3_pick)
            if s4_winner is not None:
                final = self.stage5_smoothing(s4_winner)
                mask = final['mask']
                morph = final.get('morph') or self.morph_metrics(mask)
                self.logger.info(
                    f"[ORCHESTRATOR] Level {level} SUCCESS: "
                    f"{s4_winner.get('op', '')} + LPF → "
                    f"{mask.sum()} px, cov={100*morph['coverage']:.1f}%"
                    if morph else
                    f"[ORCHESTRATOR] Level {level} SUCCESS"
                )
                return mask, {
                    'method':      'rsm_bg_v3_port',
                    'coverage':    morph['coverage'] if morph else 0,
                    'solidity':    morph['solidity'] if morph else 0,
                    'n_holes':     morph.get('n_holes', 0) if morph else 0,
                    'compactness': morph.get('compactness', 0) if morph else 0,
                    'level':       level,
                    's4_op':       s4_winner.get('op', ''),
                }
            else:
                self.logger.info(
                    f"[ORCHESTRATOR] S4 failed at level {level} — try next"
                )

        return None, {"status": "all_levels_exhausted"}

    # ==================== FALLBACK ====================

    def _get_video_for_fallback(self) -> np.ndarray:
        filtered_path = self.get_path("filtered_video.npy", kind="debug")
        if filtered_path.exists():
            self.logger.info("[Fallback] Используется готовый debug/filtered_video.npy")
            return np.load(filtered_path)

        if self.raw_video is None:
            self._load_and_crop_video()

        if not PREPROCESS_AVAILABLE:
            self.logger.warning("[Fallback] preprocess недоступен — используем сырое видео")
            return self.raw_video

        fps = self._get_fps()
        invert = should_invert(
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
        )

        processed = preprocess_video(
            self.raw_video,
            fps=fps,
            invert=invert,
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
            sigma=2.0,
            lp_cutoff=80.0,
        )
        self.save_debug(processed, "filtered_video.npy")
        self.logger.info("[Fallback] Предобработанное видео сохранено в debug/filtered_video.npy")
        return processed

    def _generate_mask_by_type(self, video: np.ndarray, method: Dict[str, Any], fps: float) -> np.ndarray:
        if method["type"] == "foreground":
            mean_frame = video.mean(axis=0)
            return (mean_frame > method.get("threshold", 600)).astype(bool)

        # Bandpower (упрощённая реализация)
        from scipy.fft import rfft, rfftfreq
        from scipy.ndimage import gaussian_filter as gf
        mid = video[len(video) // 4 : 3 * len(video) // 4]
        sig = mid - mid.mean(0, keepdims=True)
        power = np.abs(rfft(sig, axis=0)) ** 2
        freqs = rfftfreq(mid.shape[0], d=1.0 / fps)

        if method["type"] == "bandpower":
            lo, hi = 5.0, 15.0
        elif method["type"] == "bandpower_stim":
            stim = method.get("stim_hz")
            if stim is None:
                self.logger.warning("[bandpower_stim] stim_hz is None — using 10 Hz")
                stim = 10.0
            lo, hi = stim - 1.0, stim + 1.0
        else:
            raise ValueError(f"Unknown mask method type: {method['type']}")

        bp = power[(freqs >= lo) & (freqs <= hi)].sum(axis=0)
        log_bp = np.log10(gf(bp, 1.5) + 1e-10)
        thr = np.percentile(log_bp, method.get("pct", 55))
        return (log_bp > thr).astype(bool)

    def _compute_mask_qc_fallback(self, mask: np.ndarray, video: np.ndarray) -> Dict[str, Any]:
        m: Dict[str, Any] = {"coverage": float(mask.mean())}
        if HEAVY_DEPS and mask.sum() > 0:
            from skimage.measure import label as sk_label, regionprops
            props = regionprops(sk_label(mask))
            if props:
                p = props[0]
                m["solidity"]    = float(p.solidity)
                m["compactness"] = float(4 * np.pi * p.area / (p.perimeter ** 2 + 1e-9))
        return m

    def _judge_mask_fallback(self, qc: Dict[str, Any]) -> Tuple[str, str]:
        cov = qc.get("coverage", 0)
        if cov < 0.05 or cov > 0.95:
            return "REJECT", f"bad coverage={cov:.3f}"
        if qc.get("solidity", 1.0) < 0.5:
            return "REJECT", f"low solidity={qc.get('solidity', 0):.3f}"
        if qc.get("compactness", 1.0) < 0.3:
            return "RETRY", f"low compactness={qc.get('compactness', 0):.3f}"
        return "PASS", "ok"

    def _apply_contour_smoothing(self, mask: np.ndarray) -> np.ndarray:
        if not HEAVY_DEPS:
            return mask
        base = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        main = max(contours, key=cv2.contourArea)
        if len(main) < 6:
            return mask
        ws = min(15, max(3, len(main) // 8))
        box = np.ones(ws, dtype=np.float32) / ws
        x = main[:, 0, 0].astype(float)
        y = main[:, 0, 1].astype(float)
        xs = np.convolve(np.pad(x, ws, mode="wrap"), box, mode="same")[ws:-ws]
        ys = np.convolve(np.pad(y, ws, mode="wrap"), box, mode="same")[ws:-ws]
        new_cont = np.stack((xs, ys), axis=1).astype(np.int32)
        new_mask = np.zeros_like(base)
        cv2.drawContours(new_mask, [new_cont], -1, 255, -1)
        return binary_fill_holes(new_mask > 0)

    def _fallback_with_cascade(self) -> Tuple[np.ndarray, Dict]:
        self.logger.info("[Fallback] Запуск каскада методов + судейство")
        video = self._get_video_for_fallback()
        fps = self._get_fps()
        stim_hz = self.metadata.get("stim_hz")
        if stim_hz is None:
            self.logger.warning("[Fallback] stim_hz не найден в metadata — bandpower_stim использует 10 Hz")

        methods: List[Dict[str, Any]] = [
            {"name": "bandpower_5_15_p55", "type": "bandpower",      "pct": 55},
            {"name": "bandpower_5_15_p45", "type": "bandpower",      "pct": 45},
            {"name": "bandpower_stim",     "type": "bandpower_stim", "stim_hz": stim_hz},
            {"name": "foreground_600",     "type": "foreground",     "threshold": 600},
        ]

        best_mask: Optional[np.ndarray] = None
        best_qc:   Optional[Dict]       = None
        best_name: Optional[str]        = None

        for m in methods:
            try:
                mask = self._generate_mask_by_type(video, m, fps)
                if HEAVY_DEPS:
                    mask = binary_opening(mask, iterations=1)
                    mask = binary_closing(mask, iterations=2)
                    mask = binary_fill_holes(mask)

                qc = self._compute_mask_qc_fallback(mask, video)
                verdict, reason = self._judge_mask_fallback(qc)
                self.logger.info(f"  {m['name']} → {verdict} ({reason})")

                if verdict == "PASS":
                    best_mask, best_qc, best_name = mask, qc, m["name"]
                    break
                elif verdict == "RETRY":
                    if best_mask is None or qc.get("compactness", 0) > (best_qc or {}).get("compactness", 0):
                        best_mask, best_qc, best_name = mask, qc, m["name"]
            except Exception as e:
                self.logger.warning(f"  {m['name']} failed: {e}")

        # Строгий гейтинг (R2 fix)
        if best_mask is None:
            raise ValueError(
                "All fallback methods rejected the mask. Sample requires quarantine."
            )

        if HEAVY_DEPS:
            best_mask = self._apply_contour_smoothing(best_mask)

        return best_mask.astype(bool), {
            "method":      f"fallback_{best_name}",
            "coverage":    (best_qc or {}).get("coverage", 0),
            "compactness": (best_qc or {}).get("compactness", 0),
        }

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Основной метод агента.

        1. Skip если mask.npy существует и force=False
        2. Загрузить metadata + video (_prepare_data)
        3. Попытаться загрузить raw_rsm.npy; если нет — запустить LoaderAgent
        4. PRIMARY: RSM non-phys pipeline
        5. FALLBACK: каскад если PRIMARY вернул None или маска с дырами
        6. Сохранить mask.npy + метрики
        """
        if not force and self.exists("mask.npy"):
            self.logger.info("mask.npy already exists, skipping (use force=True to rerun)")
            return {"status": "skipped"}

        # _prepare_data должен идти ДО попытки загрузить raw_rsm
        # (иначе metadata не загружена и LoaderAgent не может быть вызван корректно)
        try:
            self._prepare_data()
        except FileNotFoundError:
            self.logger.info("raw_video.npy not found — running LoaderAgent first")
            from cardiac_pipeline.agents.loader_agent import LoaderAgent
            LoaderAgent(self.sample_id, self.config).run()
            self._prepare_data()

        try:
            raw_rsm = self.load_must("raw_rsm.npy")
        except FileNotFoundError:
            self.logger.warning("raw_rsm.npy not found — skipping PRIMARY, going to FALLBACK")
            raw_rsm = None

        mask, metrics = self._primary_rsm_bg_pipeline(raw_rsm)

        if mask is None or metrics.get("n_holes", 99) > 0:
            self.logger.info("[MaskAgent] Переход в Fallback")
            mask, fb = self._fallback_with_cascade()
            metrics.update(fb)

        self.save_must(mask.astype(np.uint8), "mask.npy")
        self._log_metrics(metrics)

        return {
            "status":  "success",
            "method":  metrics.get("method"),
            "metrics": metrics,
        }


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="MaskAgent v4.2 — Stage 2 mask generation")
    parser.add_argument("sample_id", help="Sample ID (e.g. 004A)")
    parser.add_argument("--results-root", default=".", help="Results root directory (containing <sample_id>/must/)")
    parser.add_argument("--config", default=None, help="Path to config YAML (optional)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    # Load config
    cfg = None
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig()

    # Override results_root BEFORE creating agent (must_dir is set in __init__)
    cfg.results_root = str(args.results_root)

    agent = MaskAgent(args.sample_id, cfg)

    print(f"MaskAgent v4.2 — running on sample {args.sample_id}, results_root={args.results_root}")
    result = agent.run(force=args.force)
    print(json.dumps(result, indent=2, default=str))
