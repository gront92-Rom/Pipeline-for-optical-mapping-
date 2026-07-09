#!/usr/bin/env python3
"""
metadata_extractor.py — Read-only metadata extractor for MiCAM optical files.

Извлекает метаданные из:
    1. .bvx (XML sidecar) — лучший источник (pixel_size, gain, bit_depth, OperationHistory)
    2. .rsh (MiCAM ULTIMA v1505+) — fps, n_frames, dims, stim, camera
    3. .gsh (MiCAM05 fallback) — fps, n_frames, dims, date

Дополнительно парсит filename для: sample_id, dye (A/B), stim_hz,
protocol (baseline/iso/stretch/bleb/...), timepoint, model (TAC/SHAM/WT),
drug (iso/bleb/nola/ach/carb), tissue (SAN/LAA/RAA).

compute_dominant_freq() — определяет основную частоту из сигнала (FFT),
надёжнее чем stim_hz из имени для спонтанной/нерегулярной активности.

Сохраняет metadata.json и возвращает dict.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Filename parsing
# =============================================================================
_SAMPLE_ID_RE = re.compile(r'(?<![0-9])(\d{3,4}[AB])(?:[_.\-]|$)', re.IGNORECASE)
_STIM_HZ_RE = re.compile(r'(\d+(?:\.\d+)?)\s*H?z', re.IGNORECASE)
_VERSION_RE = re.compile(r'version\s+(\d+)', re.IGNORECASE)


def parse_sample_id_from_filename(filename: str) -> Optional[str]:
    m = _SAMPLE_ID_RE.search(filename)
    if not m:
        return None
    sid = m.group(1).upper()
    tail = filename[m.end():]
    if tail.startswith("_") or tail.startswith("-"):
        qual = re.match(r'[_\-](\w+)', tail)
        if qual:
            return f"{sid}_{qual.group(1)}"
    return sid


def parse_dye_from_filename(filename: str, sample_id: Optional[str] = None) -> Optional[str]:
    target = sample_id or parse_sample_id_from_filename(filename) or filename
    token = target.upper().split("_")[0]
    if token.endswith("B"):
        return "B"
    if token.endswith("A"):
        return "A"
    return None


def recording_mode_from_dye(dye: Optional[str]) -> Optional[str]:
    if dye == "A":
        return "voltage"
    if dye == "B":
        return "calcium"
    return None


def parse_stim_hz_from_filename(filename: str) -> Optional[float]:
    m = _STIM_HZ_RE.search(filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# =============================================================================
# Protocol / condition parsing from filename
# =============================================================================
_PROTOCOL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("arrhythmia",     re.compile(r'arrh(?:yth|y)', re.I)),
    ("pace_and_stop",  re.compile(r'pace\s+and\s+stop|[-_]st[-_]', re.I)),
    ("baseline",       re.compile(r'bsl(?:ine|2|3|[-_]\d)?|bs2|bs3|baseline\d?|BSL|BSL2', re.I)),
    ("bleb",           re.compile(r'bleb(?:bistatin)?', re.I)),
    ("iso",            re.compile(r'(?<![a-zA-Z])iso(?:50|100)?(?![a-zA-Z])|(?<![a-zA-Z])ISO\d*(?![a-zA-Z])', re.I)),
    ("stretch",        re.compile(r'stretch', re.I)),
    ("test",           re.compile(r'(?:^|[-_])tes(?:t)?(?:[-_]|$)', re.I)),
]

_MODEL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("mTACc",   re.compile(r'mTACc', re.I)),
    ("TAC",     re.compile(r'(?<!m)TAC(?!c)', re.I)),
    ("mSHAM",   re.compile(r'mSHAM', re.I)),
    ("SHAM",    re.compile(r'(?<!m)SHAM', re.I)),
    ("WT",      re.compile(r'(?<![a-z])WT(?![a-z])', re.I)),
]

_DRUG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("iso",           re.compile(r'(?<![a-zA-Z])iso(?:50|100)?(?![a-zA-Z])|(?<![a-zA-Z])ISO\d*(?![a-zA-Z])', re.I)),
    ("blebbistatin",  re.compile(r'bleb(?:bistatin)?', re.I)),
    ("nola",          re.compile(r'(?<![a-zA-Z])nola(?![a-zA-Z])', re.I)),
    ("acetylcholine", re.compile(r'(?<![a-zA-Z])ach(?![a-zA-Z])', re.I)),
    ("carbachol",     re.compile(r'carb(?:ochol)?', re.I)),
    ("high_calcium",  re.compile(r'(?<![a-zA-Z])ca(?:50)?(?![a-zA-Z])', re.I)),
]

_TISSUE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("SAN",  re.compile(r'(?<![a-zA-Z])SAN(?![a-zA-Z])')),
    ("LAA",  re.compile(r'(?<![a-zA-Z])LAA(?![a-zA-Z])')),
    ("RAA",  re.compile(r'(?<![a-zA-Z])RAA(?![a-zA-Z])')),
]

_TIMEPOINT_RE = re.compile(
    r'bsl[-_]?(?:2_)?(\d)|bs(\d)|baseline(\d)|bsl[-_](\d)', re.I
)


def parse_protocol_from_filename(filename: str) -> Optional[str]:
    """Extract protocol: baseline, iso, stretch, bleb, calcium, arrhythmia, etc."""
    # Pre-split concatenated tokens like 'isoCa' so 'iso' is found
    pre = re.sub(r'(iso(?:50|100)?)([Cc][Aa])', r'\1-\2', filename)
    for proto, pat in _PROTOCOL_PATTERNS:
        if pat.search(pre):
            return proto
    return None


def parse_timepoint_from_filename(filename: str) -> Optional[int]:
    """Extract timepoint (1,2,3...) from patterns like bsl-2, bs2, baseline2."""
    m = _TIMEPOINT_RE.search(filename)
    if m:
        for g in m.groups():
            if g is not None:
                try:
                    return int(g)
                except ValueError:
                    pass
    return None


def parse_model_from_filename(filename: str) -> Optional[str]:
    """Extract animal model: TAC, mTACc, SHAM, mSHAM, WT."""
    for model, pat in _MODEL_PATTERNS:
        if pat.search(filename):
            return model
    return None


def parse_drug_from_filename(filename: str) -> Optional[str]:
    """Extract drug: iso, blebbistatin, nola, acetylcholine, carbachol.

    Returns the first match. For multiple drugs, use parse_drugs_from_filename.
    """
    for drug, pat in _DRUG_PATTERNS:
        if pat.search(filename):
            return drug
    return None


def parse_drugs_from_filename(filename: str) -> List[str]:
    """Extract ALL drugs mentioned in filename.

    Handles concatenated tokens like 'isoCa' (iso + high_calcium)
    by splitting on known drug boundaries before searching.
    """
    # Pre-split concatenated drug tokens: isoCa → iso-Ca, isoCA → iso-CA, etc.
    # Insert separator between 'iso' and 'Ca' when concatenated
    pre = re.sub(r'(iso(?:50|100)?)([Cc][Aa])', r'\1-\2', filename)
    # Also split 'Iso' followed by 'Ca' in any case combination
    pre = re.sub(r'(ISO(?:50|100)?)(CA)', r'\1-\2', pre, flags=re.I)
    found: List[str] = []
    for drug, pat in _DRUG_PATTERNS:
        if pat.search(pre):
            found.append(drug)
    return found


def parse_tissue_from_filename(filename: str) -> Optional[str]:
    """Extract tissue: SAN, LAA, RAA."""
    for tissue, pat in _TISSUE_PATTERNS:
        if pat.search(filename):
            return tissue
    return None


def compute_dominant_freq(video: np.ndarray, fps: float, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Compute dominant frequency (Hz) from the actual optical signal.

    Uses FFT on the mean trace across valid pixels. More reliable than
    filename-derived stim_hz for spontaneous or irregular rhythms.

    Parameters
    ----------
    video : np.ndarray (T, H, W) or (T, N)
        Raw or loaded optical video.
    fps : float
        Sampling rate in Hz.
    mask : np.ndarray (H, W), optional
        Boolean mask of valid tissue pixels. If None, uses all pixels.

    Returns
    -------
    float or None
        Dominant frequency in Hz, or None if cannot be determined.
    """
    if video is None or fps is None or fps <= 0:
        return None
    T = video.shape[0]
    if T < 8:
        return None
    # Compute mean trace
    if video.ndim == 3 and mask is not None:
        trace = video[:, mask].mean(axis=1)
    elif video.ndim == 3:
        trace = video.mean(axis=(1, 2))
    elif video.ndim == 2:
        trace = video.mean(axis=1)
    else:
        return None
    # Detrend + FFT
    trace = trace - trace.mean()
    # Use scipy if available, else numpy
    try:
        from scipy.signal import detrend, find_peaks
        trace = detrend(trace)
    except ImportError:
        pass
    spectrum = np.abs(np.fft.rfft(trace))
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    if len(freqs) < 2:
        return None
    # Skip DC bin (index 0)
    spectrum[0] = 0
    # Find dominant peak
    peak_idx = np.argmax(spectrum[1:]) + 1
    return round(float(freqs[peak_idx]), 2)


# =============================================================================
# Text header parsing
# =============================================================================
def _read_version(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                m = _VERSION_RE.search(line)
                if m:
                    return f"v{m.group(1)}"
    except OSError:
        pass
    return None


def _parse_key_value_file(path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not path.exists():
        return meta
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                elif ":" in line:
                    key, val = line.split(":", 1)
                else:
                    continue
                meta[key.strip()] = val.strip()
    except OSError:
        pass
    return meta


def _parse_size_tuple(s: str) -> Optional[Tuple[int, int]]:
    m = re.search(r'\((\d+)\s*,\s*(\d+)\)', s)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except ValueError:
            return None
    return None


def _ms_to_hz(s: str) -> Optional[float]:
    if not s:
        return None
    m = re.search(r'([\d.]+)\s*(u?sec|msec)', s, re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "usec":
            return round(1_000_000.0 / v, 2)
        if unit == "msec":
            return round(1000.0 / v, 2)
        return round(1.0 / v, 2)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_int(s: Any) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Format-specific parsers
# =============================================================================
def parse_rsh(rsh_path: Path) -> Dict[str, Any]:
    if not rsh_path.exists():
        return {}
    raw = _parse_key_value_file(rsh_path)
    if not raw:
        return {}

    height = width = None
    for k_h, k_w in [("DataYsize", "DataXsize"), ("y", "x")]:
        if k_h in raw and k_w in raw:
            try:
                height = int(raw[k_h])
                width = int(raw[k_w])
                break
            except ValueError:
                continue

    n_frames = None
    for k in ["frame_number", "page_frames", "Frame Size"]:
        if k in raw:
            try:
                n_frames = int(raw[k])
                break
            except ValueError:
                continue

    fps = _ms_to_hz(raw.get("sample_time", ""))
    pls_interval = raw.get("pls_interval", "")
    stim_hz = _ms_to_hz(pls_interval) if pls_interval else None

    return {
        "source": "rsh",
        "source_file": str(rsh_path),
        "version": _read_version(rsh_path),
        "fps": fps,
        "n_frames": n_frames,
        "height": height,
        "width": width,
        "date": raw.get("acquisition_date") or raw.get("AcquisitionDate"),
        "sample_mode": raw.get("sample_mode"),
        "dual_cam": _safe_int(raw.get("dual_cam")),
        "gain_mode": _safe_int(raw.get("gain_mode")),
        "average": _safe_int(raw.get("average")),
        "shutter_delay_ms": raw.get("shutter_delay"),
        "trigger_src": raw.get("trigger_src"),
        "stim_mode": _safe_int(raw.get("stim_mode")),
        "pls_delay_ms": raw.get("pls_delay"),
        "pls_width_ms": raw.get("pls_width"),
        "pls_interval_ms": raw.get("pls_interval"),
        "pls_number": _safe_int(raw.get("pls_number")),
        "pls2_interval_ms": raw.get("pls2_interval"),
        "stim_hz": stim_hz,
        "raw": raw,
    }


def parse_gsh(gsh_path: Path) -> Dict[str, Any]:
    if not gsh_path.exists():
        return {}
    raw = _parse_key_value_file(gsh_path)
    if not raw:
        return {}

    fps = _ms_to_hz(raw.get("Sampling Time", ""))
    n_frames = None
    for k in ["Frame Size", "frame_number", "Number of frames"]:
        if k in raw:
            try:
                n_frames = int(raw[k])
                break
            except ValueError:
                continue

    width = height = None
    size = _parse_size_tuple(raw.get("Data Size", ""))
    if size:
        width, height = size
    else:
        for k_w, k_h in [("Image width", "Image height"), ("DataXsize", "DataYsize")]:
            if k_w in raw and k_h in raw:
                try:
                    width = int(raw[k_w])
                    height = int(raw[k_h])
                    break
                except ValueError:
                    continue

    return {
        "source": "gsh",
        "source_file": str(gsh_path),
        "version": _read_version(gsh_path),
        "fps": fps,
        "n_frames": n_frames,
        "height": height,
        "width": width,
        "date": raw.get("AcquisitionDate") or raw.get("acquisition_date"),
        "raw": raw,
    }


def parse_bvx(bvx_path: Path) -> Dict[str, Any]:
    if not bvx_path.exists():
        return {}
    try:
        tree = ET.parse(bvx_path)
    except ET.ParseError:
        return {"_parse_error": "invalid XML"}

    root = tree.getroot()
    out: Dict[str, Any] = {"source": "bvx", "source_file": str(bvx_path)}

    for tag in ["Name", "DateCreated", "DateModified", "Id", "CameraIndex"]:
        el = root.find(tag)
        if el is not None and el.text:
            out[tag.lower()] = el.text.strip()

    acq = root.find("Acquisition")
    if acq is not None:
        for tag in ["ExposureTime", "FrameRate", "StartTime", "NumberOfFrames"]:
            el = acq.find(tag)
            if el is not None and el.text:
                try:
                    val: Any = float(el.text)
                    if tag == "NumberOfFrames":
                        val = int(val)
                    out[tag.lower()] = val
                except ValueError:
                    out[tag.lower()] = el.text.strip()

    img = root.find("Image")
    if img is not None:
        for tag in ["Averaging", "Binning", "Gain", "BitDepth", "Height", "Width", "PixelSize"]:
            el = img.find(tag)
            if el is not None and el.text:
                try:
                    v: Any = float(el.text)
                    if tag in ("Averaging", "Binning", "Gain", "BitDepth", "Height", "Width"):
                        v = int(v)
                    if tag == "PixelSize":
                        out["pixel_size_mm"] = v
                    elif tag == "BitDepth":
                        out["bit_depth"] = v
                    else:
                        out[f"image_{tag.lower()}"] = v
                except ValueError:
                    pass

    hist = root.find("OperationHistory")
    if hist is not None:
        ops: List[str] = [s.text.strip() for s in hist.findall("string") if s.text]
        if ops:
            out["operation_history"] = ops
            if any("Normaliz" in op for op in ops):
                out["_warning"] = "Image was Normalized (dF/F)"
            if any("Invert" in op for op in ops):
                out["_warning_polarity"] = "Polarity was Inverted post-acquisition"

    return out


# =============================================================================
# Helpers
# =============================================================================
def _merge_metadata(*sources: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for src in sources:
        for k, v in src.items():
            if k.startswith("_"):
                continue
            if k == "raw":
                if "raw" not in out:
                    out["raw"] = {}
                out["raw"].update(v)
                continue
            if v is None:
                continue
            if k not in out or out[k] is None:
                out[k] = v
    return out


def _find_files_by_stem(directory: Path, suffix: str, base_name: Optional[str]) -> List[Path]:
    """
    Find files in directory matching the given suffix, filtered by base_name.
    
    Matching strategy (in order):
      1. Parse sample_id from base_name (e.g. "0823-004A" → "004A"),
         then match against sample_id parsed from each filename.
      2. Soft match: base_name appears as substring in filename (case-insensitive).
      3. Fallback: return all files with that suffix (don't starve the pipeline).
    
    This replaces the old split("_")[0] approach which broke on long filenames
    like "20250822_TAC_bleb_baseline_6HZ0823-004A.rsh" (F42 bug).
    """
    files = sorted(directory.glob(f"*.{suffix}"))
    if not base_name:
        return files

    # 1. Sample-id match (primary)
    target_sid = parse_sample_id_from_filename(base_name)
    if target_sid:
        matches = [
            f for f in files
            if (parse_sample_id_from_filename(f.name) or "").upper() == target_sid.upper()
        ]
        if matches:
            return matches

    # 2. Soft substring match (secondary)
    base_upper = base_name.upper()
    soft = [f for f in files if base_upper in f.name.upper()]
    if soft:
        return soft

    # 3. No match found — return empty (don't return wrong files)
    return []


def _validate_frame_count(
    metadata: Dict[str, Any],
    directory: Path,
    base_name: Optional[str],
    bytes_per_pixel: int = 2
) -> Optional[str]:
    if metadata.get("n_frames") is None:
        return None
    declared = metadata["n_frames"]
    if base_name:
        target_sid = parse_sample_id_from_filename(base_name)
        if target_sid:
            chunks = sorted([
                c for c in directory.glob("*.rsd")
                if (parse_sample_id_from_filename(c.name) or "").upper() == target_sid.upper()
            ])
        else:
            base_upper = base_name.upper()
            chunks = sorted([
                c for c in directory.glob("*.rsd")
                if base_upper in c.name.upper()
            ])
    else:
        chunks = sorted(directory.glob("*.rsd"))
    if not chunks:
        return None
    w = metadata.get("width")
    h = metadata.get("height")
    if not (w and h):
        return None
    actual = sum(c.stat().st_size // (w * h * bytes_per_pixel) for c in chunks if c.exists())
    if actual != declared:
        return (
            f"declared n_frames={declared} but rsd chunks sum to {actual} "
            f"({len(chunks)} files). Recording may be incomplete."
        )
    return None


# =============================================================================
# Public API
# =============================================================================
def extract_micam_metadata(
    data_path: str | Path,
    base_name: Optional[str] = None,
    write_json: bool = True,
) -> Dict[str, Any]:
    data_path = Path(data_path)
    directory = data_path if data_path.is_dir() else data_path.parent
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    bvx = _find_files_by_stem(directory, "bvx", base_name)
    rsh = _find_files_by_stem(directory, "rsh", base_name)
    gsh = _find_files_by_stem(directory, "gsh", base_name)

    sources: List[Tuple[str, Path]] = []
    if bvx:
        sources.append(("bvx", bvx[0]))
    if rsh:
        sources.append(("rsh", rsh[0]))
    if gsh:
        sources.append(("gsh", gsh[0]))

    if not sources:
        raise ValueError(
            f"No .bvx/.rsh/.gsh found in {directory}"
            + (f" with stem '{base_name}'" if base_name else "")
        )

    parsed = []
    primary_file = None
    for kind, path in sources:
        if kind == "bvx":
            parsed.append(parse_bvx(path))
        elif kind == "rsh":
            parsed.append(parse_rsh(path))
        elif kind == "gsh":
            parsed.append(parse_gsh(path))
        primary_file = path

    merged = _merge_metadata(*parsed)

    fname = primary_file.name
    sid = parse_sample_id_from_filename(fname)
    if sid and merged.get("sample_id") is None:
        merged["sample_id"] = sid
    if merged.get("dye") is None:
        d = parse_dye_from_filename(fname, sid)
        if d:
            merged["dye"] = d
    if merged.get("stim_hz") is None:
        s = parse_stim_hz_from_filename(fname)
        if s is not None:
            merged["stim_hz"] = s
            merged["stim_hz_source"] = "filename"

    # Protocol / condition metadata from filename
    if merged.get("protocol") is None:
        p = parse_protocol_from_filename(fname)
        if p:
            merged["protocol"] = p
    if merged.get("timepoint") is None:
        tp = parse_timepoint_from_filename(fname)
        if tp is not None:
            merged["timepoint"] = tp
    if merged.get("model") is None:
        mdl = parse_model_from_filename(fname)
        if mdl:
            merged["model"] = mdl
    if merged.get("drug") is None:
        drugs = parse_drugs_from_filename(fname)
        if drugs:
            if len(drugs) == 1:
                merged["drug"] = drugs[0]
            else:
                merged["drug"] = drugs[0]  # primary drug
                merged["drugs"] = drugs     # all drugs
    if merged.get("tissue") is None:
        tis = parse_tissue_from_filename(fname)
        if tis:
            merged["tissue"] = tis

    merged["recording_mode"] = recording_mode_from_dye(merged.get("dye"))

    warnings: List[str] = []
    if merged.get("fps") is None:
        raise ValueError(f"Could not extract fps from {primary_file}")

    if merged.get("n_frames") is None:
        warnings.append("n_frames not found in header")
    if merged.get("height") is None or merged.get("width") is None:
        warnings.append("dimensions not found in header")

    if not merged.get("pixel_size_mm"):
        merged["pixel_size_mm"] = 0.085
        merged["pixel_size_source"] = "fallback_default_0.085mm_x10"
        warnings.append("pixel_size_mm missing in .bvx; using fallback 0.085 mm (MiCAM ULTIMA ×10)")

    warn_validate = _validate_frame_count(merged, directory, base_name)
    if warn_validate:
        warnings.append(warn_validate)

    for k in ("_warning", "_warning_polarity"):
        if k in merged:
            warnings.append(merged.pop(k))

    if warnings:
        merged["_warnings"] = warnings

    if write_json:
        out_path = primary_file.parent / "metadata.json"
        try:
            save = {k: v for k, v in merged.items() if k != "raw"}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(save, f, indent=2, ensure_ascii=False)
            merged["_metadata_json"] = str(out_path)
        except OSError as e:
            merged["_metadata_json_error"] = str(e)

    return merged


# =============================================================================
# CLI
# =============================================================================
def _main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Extract MiCAM metadata from .bvx/.rsh/.gsh")
    p.add_argument("path", help="Directory or file path")
    p.add_argument("--base-name", help="Sample stem e.g. 014A")
    p.add_argument("--no-write", action="store_true", help="Skip writing metadata.json")
    args = p.parse_args(argv)

    try:
        meta = extract_micam_metadata(args.path, base_name=args.base_name, write_json=not args.no_write)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(json.dumps({k: v for k, v in meta.items() if k != "raw"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
