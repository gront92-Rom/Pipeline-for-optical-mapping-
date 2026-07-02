# Consistency Report — First Pipeline Agents (v1)

**Дата:** 2026-07-02  
**Файлы проверены:**
1. `base_agent.py` — Базовый класс (PipelineConfig + BaseAgent)
2. `base_agent.txt` — Более ранний драфт (архивирован)
3. `metadata_extractor_v3.txt` → `metadata_extractor.py`
4. `preprocess_v5_final.txt` → `preprocess.py`

---

## Решения

### Две версии base_agent — выбрана `base_agent.py`

| Аспект | `base_agent.py` (выбран) | `base_agent.txt` (архив) |
|--------|--------------------------|--------------------------|
| OmegaConf | Optional (graceful fallback) | Required |
| Конструктор | `sample_id` + `PipelineConfig` | `output_dir` + raw DictConfig |
| Директории | `results/{sample_id}/must\|debug` | `output_dir/debug/` |
| Секции конфига | loader, mask, preprocess, activation... | Нет |
| `import numpy` | На уровне модуля ✅ | Только в `__main__` ❌ (баг) |

**Решение:** `base_agent.py` — каноническая версия. Старый драфт сохранён в `docs/archive/`.

---

## Исправления при интеграции

### 1. `pixel_size_um = 0.1` → `pixel_size_mm = 0.85`

**Проблема:** В `PipelineConfig` было `pixel_size_um = 0.1` (0.0001 мм), что противоречит `metadata_extractor` (fallback = 0.85 мм для MiCAM ULTIMA ×10).

**Исправление:** Заменено на `pixel_size_mm = 0.85` — единая единица с `metadata_extractor`.

### 2. `fps_default = 666.67` → удалено

**Проблема:** Наличие `fps_default` в конфиге создаёт соблазн использовать его как тихий fallback, что противоречит принципу F1 (fps всегда из metadata_extractor, raise если нет).

**Исправление:** Поле удалено из `PipelineConfig`. Комментарий объясняет почему.

### 3. `gaussian_filter(..., axes=(1,2))` → убран `axes`

**Проблема:** Параметр `axes` в `scipy.ndimage.gaussian_filter` доступен только в scipy ≥1.12. В нашем окружении scipy 1.18 — но `axes` с tuple sigma длиной 3 вызывает ошибку (rank mismatch).

**Исправление:** Убран `axes` — `sigma=(0, sigma, sigma)` уже обеспечивает spatial-only smoothing (0 по оси времени).

---

## Проверка согласованности

### fps — СОГЛАСОВАНО ✅

| Модуль | Поведение |
|--------|-----------|
| `metadata_extractor` | `raise ValueError` если fps не найден в .rsh/.gsh/.bvx |
| `preprocess` | `raise ValueError("fps должен быть получен из metadata_extractor!")` |
| `base_agent` | Нет `fps_default` — не предоставляет тихий fallback |

**Вывод:** Цепочка `metadata_extractor → preprocess` гарантирует, что fps всегда явный. Фикс F1 закрыт на уровне новых агентов.

### pixel_size — СОГЛАСОВАНО ✅

| Модуль | Значение | Единица |
|--------|----------|---------|
| `metadata_extractor` | 0.85 (fallback) / из .bvx | mm |
| `base_agent.PipelineConfig` | 0.85 | mm |
| `config/default.yaml` | 0.85 | mm |

### dye / invert logic — СОГЛАСОВАНО ✅

| Модуль | A (VSD) | B (Ca²⁺) |
|--------|---------|-----------|
| `metadata_extractor.parse_dye_from_filename` | "A" | "B" |
| `metadata_extractor.recording_mode_from_dye` | "voltage" | "calcium" |
| `preprocess.should_invert` | True (invert) | False (no invert) |

### sample_id regex — СОГЛАСОВАНО ✅

Оба модуля используют идентичный паттерн: `r'(?<![0-9])(\d{3,4}[AB])(?:[_.\-]|$)'`

### File contract — СОГЛАСОВАНО ✅

`BaseAgent` сохраняет в `results/{sample_id}/must/` и `results/{sample_id}/debug/` — соответствует 5-классной системе из `PIPELINE_STRUCTURE.md`.

---

## Тесты пройдены

```
✓ PipelineConfig: pixel_size_mm=0.85
✓ metadata_extractor parsing OK
✓ should_invert OK
✓ preprocess raises on missing fps
✓ preprocess_video runs OK
✓ BaseAgent instantiation OK
=== ALL 7 CONSISTENCY TESTS PASSED ===
```

---

## Структура после интеграции

```
src/cardiac_pipeline/
├── __init__.py
├── base_agent.py              ← BaseAgent + PipelineConfig + load_config
├── agents/
│   └── __init__.py            ← (будущие агенты)
└── utils/
    ├── __init__.py
    ├── metadata_extractor.py  ← extract_micam_metadata (read-only)
    └── preprocess.py          ← preprocess_video (spatial + temporal + invert)

config/
└── default.yaml               ← Все параметры пайплайна

docs/archive/
└── base_agent_v1_draft.py     ← Старый драфт (для справки)
```

---

## Оставшиеся замечания (не блокеры)

1. **`metadata_extractor` pixel_size fallback warning** — при отсутствии .bvx используется 0.85 мм с предупреждением. Это корректное поведение (лучше чем crash), но в будущем стоит сделать configurable.

2. **`preprocess.should_invert` print warning** — использует `print()` вместо `logging`. Минорно, но стоит перевести на `logging.warning()` для consistency с BaseAgent.

3. **`preprocess._parse_sample_id`** — дублирует логику из `metadata_extractor.parse_sample_id_from_filename`. В будущем стоит импортировать из одного места (DRY).

4. **`normalize_traces` — `np.percentile(axis=0, q=q)`** — порядок аргументов нестандартный (positional `axis` vs keyword `q`). Работает, но может сбить с толку.
