# PIPELINE MASTER REVIEW — Сводный документ
> Объединяет 4 сессии ревью: activation_agent, conduction_analysis, alternans_detection, source_cv_agent, apd_agent, alternans_worker, optical_pipeline_worker, auto_mask, rsm_mask_worker v2/v3.
> Источник: REVIEW_GUIDE.md + PIPELINE_REVIEW.md + PIPELINE_REVIEW_session4.md + PIPELINE_REVIEW_apd_source_cv.md + файл2.md

---

## I. КАРТА ФАЙЛОВ И СТАДИЙ

| Файл | Стадия | Строк | Охват ревью |
|------|--------|-------|-------------|
| `rsm_mask_worker_v2/v3.py` | Stage 1/2 (mask) | 851/983 | дифф full; compute sampled |
| `auto_mask.py` | mask builder | 370 | сигнатуры + константы; compute sampled |
| `activation_agent.py` | Stage 3/4 (activation map) | 887 | **full** (4 метода целиком) |
| `conduction_analysis.py` | библиотека CV | 326 | **full** |
| `source_cv_agent.py` | standalone CV (Stage ?) | 960 | compute+main full; 3 plot sampled |
| `apd_agent.py` | Stage 4 (APD traces) | 543 | **full** (compute+main+validate) |
| `alternans_detection.py` | Stage 5 (alternans) | 490 | **full** |
| `alternans_worker.py` | Stage 5 standalone | 652 | почти full (plot структурно) |
| `optical_pipeline_worker.py` | Оркестратор всех стадий | 3066 | **sampled** (judge/retry/run_pipeline/stage_load/stage_activation/stage_alternans/stage_apd/__main__) |

**НЕ РЕВЬЮИРОВАЛИСЬ:** stage_mask, stage_cv, stage_ows, stage_phase_df, stage_phenotype целиком; CSV/summary/plot-хелперы воркера; `qc_thresholds.yaml` (упоминается, но не приложен); `peak_detector_agent.py`.

---

## II. ГЛАВНОЕ: 5 СИСТЕМНЫХ ПАТТЕРНОВ (кросс-файловые)

### R1 — Дрейф калибровки (FPS + PIXEL)
**fps:** ни один агент (activation, conduction, alternans_detection, apd) не читает fps из заголовка `.gsh`. Везде дефолт `1000.0` или `dt_ms=1.0`. Реальные значения: **666.67 Hz (MiCAM ULTIMA)** и **500 Hz (SHAM 5-08)** → ошибка 33–50% во всех ms-величинах (TAT, APD, BCL, CV) при немолчаливом exit 0.

**Единственное исключение:** `optical_pipeline_worker._resolve_fps` — читает из `.gsh`, бросает при отсутствии. Но этот fps НЕ доходит до sub-агентов надёжно.

**pixel_size_mm:** канон = `1.0` в `conduction_analysis`, но `source_cv_agent` имеет дефолт `0.85` в 4 сигнатурах и хардкод `*0.85` в графике. `optical_pipeline_worker` переопределяет `pixel_size_mm=0.085` (вероятно опечатка 0.85→0.085) в ≥3 местах.

**crop:** `CROP_LEFT=20 / CROP_RIGHT=8` захардкожены в `rsm_mask_worker` и воркере.

**filter bands:** `auto_mask` имеет `highcut=450` — при 500 Hz SHAM это выше Найквиста (250 Hz) → невалидный фильтр.

### R2 — Тихий проход неправды (Silent pass)
Провал расчёта или пропуск биений → `'normal'`/`PASS`/exit 0:

| Файл | Паттерн | Конкретно |
|------|---------|-----------|
| `conduction_analysis` | NaN-CV → `'normal'` | C2 |
| `alternans_detection` | NaN → `'normal'`; 2:1 → `'normal'` | AL2, AL3 |
| `apd_agent` | exit 0 на абсурдном APD | A2, A5 |
| `alternans_worker` | всегда exit 0, нет judge | A1 |
| `source_cv_agent` | нет QC вообще | SC1 |
| `optical_pipeline_worker` | judge пропускает `None`-метрику; quarantined → exit 0; `soft_fail` считает return 1 успехом | P1, P2, P11 |

**Критическая точка:** SHAM 5-08 002A (KNOWN ANOMALOUS, 2:1 alternans) классифицируется как `'normal'` — прямое нарушение MEMORY.

### R3 — Дублирование с дрейфом
- **Карта активации:** `activation_agent.activation_vectorized_interp` ≈ `conduction_analysis.compute_activation_map` — расхождение по fps.
- **Структурный тензор:** `source_cv_agent.raw_structure_tensor` ≈ `conduction_analysis.cv_method_structure_tensor`.
- **Детектор биений:** `activation_agent.detect_beats` (find_peaks) vs `alternans_detection` (argrelmax) vs `alternans_worker.detect_dye` (3-я копия dye-detect).
- **dye-detect:** `peak_detector_agent.detect_dye` (стр.704 воркера) + `optical_pipeline_worker._detect_dye` + `alternans_worker.detect_dye` — три реализации, все дефолтят в `"A"`.
- **Острейший случай — CV1:** `source_cv_agent` импортирует `cv_method_local_fit` из `conduction_analysis`, которой там НЕТ → ImportError при прогоне.

### R4 — Монолит / устаревший контракт
- Все 4 основных агента: compute + matplotlib в одном файле.
- `activation_agent` стр.820: `final_verdict = bool-expression` → отчёт получает True/False вместо "PASS"/"WARN"/"REJECT".
- `alternans_detection` AL4: комментарий «удаляем меньший пик», код удаляет по индексу.
- `rsm_mask_worker`: changelog v3 описывает несуществующую дельту (v1→v2 выдаётся за v2→v3).
- `optical_pipeline_worker`: `RETRY` — мёртвый вердикт в коде, но в докстринге — ядро логики.
- Stage mislabel: `apd_agent` помечен «Stage 2», он Stage 4.

### R5 — Дрейф контракта имён метрик producer↔consumer (новая, предложена в session4)
`alternans_worker` (success-путь) пишет ключи `spatial_*`/`temporal_*`, fallback-путь — `ac_median`/`ac_gt5_pct`. `run_pipeline` агрегирует `ac_median` → есть только в fallback → «хороший» путь даёт `KeyError`, деградированный работает. `judge()` судит пути по разным ключам.

---

## III. ПОЛНЫЙ РЕЕСТР НАХОДОК (все сессии)

### conduction_analysis.py
| ID | SEV | Находка |
|----|-----|---------|
| C1 | 1 | `FS_HZ=1000.0` на уровне модуля; `DT_MS` жёстко в `compute_activation_map` (стр.21–23, 88) |
| C2 | 1 | NaN-CV → фенотип `'normal'`; нет гейта/exit-кода (стр.238–245) |
| C3 | 2 | Физиологические границы CV захардкожены в трёх разных местах: 0.05–2.0 (стр.119–128, 191) |
| C4 | 3 | `compute_activation_map` дублирует `activation_vectorized_interp` (R3) |

### activation_agent.py (сводка обеих сессий)
| ID | SEV | Находка |
|----|-----|---------|
| A1/AG1 | 1 | `fps=1000.0` дефолт во всех функциях и argparse; нет чтения из `.gsh` (стр.81,91,134,180,225,312,380,670) |
| A2/AG5 | 2 | `stim_hz` парсится из `loaded_video.npy` (без Hz-токена) → дефолт 10.0 → неверные BCL/beat_pass |
| A3/AG3 | 2 | `final_verdict` получает bool вместо строки (стр.820) |
| A4 | 1 | judge не гейтит пропуск биений; нет сверки с `stim_hz×duration` |
| A5 | 3 | `fft_phase` не в `--method choices` → argparse-ошибка |
| A6/AG7 | 3 | Хардкод геометрии 100×100 (стр.520, 534) |
| AG2 | 1 | Exit 0 на TAT вне 5–100 мс (только WARN, не FAIL) |
| AG4 | 2 | Порог beat_pass_rate рассинхронизирован: judge REJECT при <0.3, validate FAIL при <0.5 |
| AG6 | 2 | Полярность A/B из имени output_dir, не из метаданных (нарушение MEMORY) |
| AG8 | 2 | Окно анализа `ws=pk-50, we=pk+10` в кадрах (fps-слепое), тогда как baseline fps-aware |
| AG9 | 3 | 50pct: fallback на время пика при отсутствии crossing — без флага |
| AG10 | 3 | Двойной сброс биений: detect_beats + каждый метод повторно |

### source_cv_agent.py (сводка)
| ID | SEV | Находка |
|----|-----|---------|
| CV1/SC1 | 1 | Нет гейтинга вообще; всегда exit 0 (стр.650–956) |
| CV2/SC4 | 1→2 | `pixel_size_mm=0.85` дефолт в 4 сигнатурах; latent при main-пути, активен при прямом вызове |
| CV3 | 2 | График конвертирует r_px→мм хардкодом `*0.85` (стр.580), анализ считался на 1.0 |
| CV4/SC3 | 2 | Нет judge/QC; границы CV несогласованы между методами (2.0/4.0/4.5/5.0/5.0) |
| CV5/SC5 | 3 | Абсолютный путь `/home/rymedv/...` (стр.73) |
| — (CV1) | 1 | `import cv_method_local_fit` из `conduction_analysis` — функции не существует → ImportError |
| SC2 | 2 | Нет provenance/hash входов; ms-шкала непроверяема |
| SC6 | 3 | Авто-поиск маски вверх по 4 родителям, first-match — риск взять маску чужого образца |
| SC7 | 3 | Заполнение вне-маски nanmean в `raw_structure_tensor` → завышенный CV на границе |

### apd_agent.py
| ID | SEV | Находка |
|----|-----|---------|
| A1 | 1 | `dt_ms=1.0` зашит; main зовёт `process_points` без dt_ms → fps=1000 молча (стр.121,129,168,486) |
| A2 | 1 | Нет peak-threshold floor вообще; `detect_activation=np.argmax(dvdt)` без мультибит-детекции |
| A3 | 1 | Нет A/B ветки и hard-invert; argmax ловит не тот фронт у неинвертированного VSD |
| A4 | 2 | `APD_raw` бессмысленна: `amplitude=abs(amp_peak)` предполагает baseline=0 (нет у raw-флуоресценции) |
| A5 | 2 | WARN вместо FAIL для абсурдных результатов → exit 0 |
| A6 | 2 | Отчёт врёт про provenance: `--baseline-frames` unused, но печатается как параметр (стр.60,300,455) |
| A7 | 3 | `process_points` переписывает `detect_activation` инлайном — дрейф |
| A8 | 3 | QC-диапазон APD 5–200 ms зашит в `validate` (место в `qc_thresholds.yaml`) |
| A9 | 3 | Stage mislabel: помечен «Stage 2», реально Stage 4 |
| A10 | 3 | Монолит |

### alternans_detection.py
| ID | SEV | Находка |
|----|-----|---------|
| AL1 | 1 | `FS_HZ=1000.0` дефолт (стр.390) |
| AL2 | 1 | 2:1 → ACmedian≈0 → `'normal'`; `argrelmax(order=min_interval//2)` выбрасывает малое биение |
| AL3 | 1 | NaN → `'normal'`; нет гейта (стр.257–262) |
| AL4 | 2 | Контракт ≠ код: «Remove smaller peak», а удаляется позднее-индексированный (стр.86–89) |
| AL5 | 3 | Неиспользуемые top-level импорты `run_pipeline, MaskingParams, compute_apd_per_beat` → падение при сбое `pipeline.py` |

### alternans_worker.py
| ID | SEV | Находка |
|----|-----|---------|
| AW1 | 1 | Нет judge(); всегда exit 0; метрики NaN/пуста → `0` (неотличим от «нет альтернанса») |
| AW2 | 1 | `f_dom=argmax` без проверки ~0.5 cycles/beat → тренд APD даёт высокую «purity» как альтернанс |
| AW3 | 2 | Нет provenance/hash входного `apd_per_beat_3d.npz` |
| AW4 | 2 | `detect_dye` дефолтит в `"A"` → молчаливый mislabel CaT→APD; 3-я копия dye-detect |
| AW5 | 3 | Floor знака `0.5 ms` и `discordant < 0.25` зашиты |
| AW6 | 4 | Лейбл-рассинхрон: «cycles/beat» в main vs «Cycles per beat-pair» в графике |

### optical_pipeline_worker.py (sampled)
| ID | SEV | Находка |
|----|-----|---------|
| P1 | 1 | judge() пропускает `None`-метрику и любой односторонний порог (`[10, null]` никогда не срабатывает) |
| P2 | 1 | Quarantined-файл → exit 0 (`__main__` не зовёт `sys.exit(1)`) |
| P3 | 1 | Рассинхрон имён метрик Stage 5: success-путь `spatial_*`, fallback `ac_median` → KeyError на «хорошем» пути |
| P4 | 1 | `pixel_size_mm=0.085` передефинирован дефолтом ≥3 раз (вероятно опечатка 0.85→0.085) |
| P5 | 2 | `peak_thr_frac=0.5` — soft-дефолт; LLM-попытки могут опустить; нет floor в коде |
| P6 | 2 | MiCAM-crop `20/8` зашит дефолтами функции |
| P7 | 2 | Окно активации `ws=pk-50, we=pk+10` в кадрах, не мс (fps-blind) |
| P8 | 3 | `RETRY` — мёртвый вердикт; реальные ретраи через `REJECT→continue` |
| P9 | 3 | Sentinel-коллизия `stim_hz=10`: дефолт запускает re-parse даже при настоящих 10 Гц |
| P10 | 3 | Тройной `detect_dye` (R3) |
| P11 | 3 | `soft_fail` считает returncode 1 успехом при наличии report |
| P12 | 4 | Дубль ключа `"stage_6_ows"` в `STAGE_DEFS` — второй перетирает первый |

### rsm_mask_worker v2/v3 + auto_mask
| ID | SEV | Находка |
|----|-----|---------|
| MW1 | 2 | Changelog v3 фиктивен: описывает несуществующую дельту v1→v2 |
| MW2 | 2→fixed | v2 нестабильные имена выходов; v3 исправлено → `mask.npy`/`metrics.json` |
| MW3 | 2 | `_cleanup_intermediates` удаляет `data_inv.npy` и `activation_peaks.npy` activation_agent + `paths.json` provenance |
| MW4 | 3 | `CROP_LEFT=20/CROP_RIGHT=8` зашиты модульными константами |
| MW5 | 4 | `rsh_path.replace('.rsh','.rsm')` ломается при `.rsh` в каталоге |
| AM1 | 3 | `highcut=450.0` → выше Найквиста при SHAM 500 Hz; полосы не валидируются против fps |
| AM2 | 4 | `pct=55` дублируется в 3 функциях независимо |

---

## IV. ПЛАН ФИКСОВ (объединённый, P0 → P3)

### P0 — Корректность / безопасность (SEV1) — делать немедленно

**F1. Единый загрузчик fps из `.gsh`** (закрывает R1-fps: C1, A1, AG1, AL1, A1-apd)
- Добавить `pipelines/io_meta.py` → `read_fps_from_gsh(path) -> float`; падать на `.gsd` → резолвить в `.rsh`.
- Убрать все числовые дефолты `fps=1000.0 / FS_HZ=1000.0 / dt_ms=1.0`.
- Сделать fps **обязательным аргументом** во всех агентах (без дефолта → ненулевой exit).
- `conduction_analysis.compute_activation_map`: заменить `DT_MS` на параметр.
- `optical_pipeline_worker` должен явно передавать fps в каждый sub-агент через CLI.

**F2. Единый `PIXEL_SIZE_MM`** (закрывает R1-pixel: CV2, CV3, P4)
- Убрать `pixel_size_mm=0.85` из 4 сигнатур `source_cv_agent`; без дефолта или импорт канона.
- Исправить `*0.85` в графике (стр.580) на `pixel_size_mm`.
- Исправить `pixel_size_mm=0.085` в воркере (P4) — опечатка.

**F3. Запрет тихого `'normal'` на NaN** (закрывает R2-NaN: C2, AL3, AW1)
- При NaN-ключевой метрике → `phenotype='undetermined'` + REJECT + ненулевой exit.
- `source_cv_agent`: добавить judge/QC с ненулевым exit (SC1).
- `alternans_worker`: добавить `judge()` (AW1).

**F4. Code-floor правила порога пиков + сверка числа биений** (закрывает R2-биения: A4, AL2, AW2, P5)
- Вынести min peak-threshold и ожидаемое число биений в `qc_thresholds.yaml`.
- `judge()` обоих детекторов: REJECT при недосчёте биений vs `duration×stim_hz`.
- REJECT (не PASS) при 2:1 capture / alternans pattern.
- Code-floor в коде, не только в yaml/промпте.
- Починить `f_dom` в `alternans_worker`: проверять ~0.5 cycles/beat (AW2).

**F5. Починить сломанный import cv_method_local_fit** (закрывает R3-острое: CV1/ImportError)
- Добавить `cv_method_local_fit` в `conduction_analysis` ИЛИ убрать вызов из `source_cv_agent`.

**F6. Добавить A/B ветку и hard-invert в `apd_agent`** (A3)
- Читать A/B из метаданных/manifest; `assert` перед расчётом.

**F7. Починить `judge()` воркера** (закрывает R2-judge: P1, P2)
- Обработать `val is None` → REJECT (не skip).
- Обработать односторонние пороги `[lo, null]`.
- `__main__`: вызывать `sys.exit(1)` при quarantined.

**F8. Починить рассинхрон ключей метрик Stage 5** (закрывает R5: P3)
- Зафиксировать единую схему ключей для `alternans_worker` output.
- `run_pipeline` читает эту же схему; один путь исполнения, не два несовместимых.

**F9. Добавить регресс-тест SHAM 5-08 002A**
- Гарантировать: 2:1 образец → REJECT/anomalous, не `'normal'`.

### P1 — Воспроизводимость (SEV2)

- **F10.** Починить `final_verdict` bool-баг в `activation_agent` (AG3/A3, стр.820).
- **F11.** `stim_hz`: брать из метаданных/manifest, не из имени файла; при отсутствии → REJECT (A2/AG5).
- **F12.** Унифицировать физиологические границы CV в один конфиг (C3, SC3, CV4).
- **F13.** Синхронизировать порог beat_pass_rate: judge и validate должны использовать одно значение (AG4).
- **F14.** Полярность A/B: читать из метаданных, не из имени output_dir (AG6). 
- **F15.** Окно анализа `ws=pk-50, we=pk+10` перевести в мс: `ws=pk-int(50*fps/1000)` (AG8/P7).
- **F16.** Добавить provenance/hash в `source_cv_agent` и `alternans_worker` (SC2, AW3).
- **F17.** Починить `APD_raw` в `apd_agent`: вычитать baseline перед `amplitude` (A4).
- **F18.** Повысить WARN→FAIL для абсурдных APD/TAT в `apd_agent` и `activation_agent` (A5/AG2).
- **F19.** Починить provenance-ложь в `apd_agent` report (A6).
- **F20.** Исправить `_cleanup_intermediates` в `rsm_mask_worker v3`: добавить `data_inv.npy`, `activation_peaks.npy`, `paths.json` в allowlist (MW3).
- **F21.** `soft_fail` воркера: возвращать ненулевой exit, не считать returncode 1 успехом (P11).
- **F22.** `peak_thr_frac`: добавить код-floor (не только soft-дефолт), чтобы LLM-попытки не могли занизить (P5).
- **F23.** Фиктивный changelog v3: привести в соответствие реальному диффу (MW1).

### P2 — Организация (SEV3)

- **F24.** Дедуп карты активации: один источник (`conduction_analysis`), остальные импортируют (C4, R3).
- **F25.** Дедуп структурного тензора (SC7 / R3).
- **F26.** Дедуп детектора биений: один канонический (`activation_agent`?), остальные импортируют (AL5, AW4, P10).
- **F27.** Убрать абсолютный путь `/home/rymedv/...` в `source_cv_agent` (CV5/SC5).
- **F28.** Убрать неиспользуемые импорты в `alternans_detection` (AL5).
- **F29.** `stim_hz` sentinel-коллизия: сменить дефолт-sentinel (P9).
- **F30.** `fft_phase` добавить в `--method choices` (A5).
- **F31.** Геометрию 100×100 читать из формы маски (AG7/A6).
- **F32.** AG9: флагировать fallback 50pct на время пика.
- **F33.** `auto_mask`: валидировать `highcut` против `fps/2` (Найквист) (AM1).
- **F34.** Исправить `stage_6_ows` дубль-ключ в воркере (P12-подобное).
- **F35.** Дедуп `dye-detect`: три реализации → одна в `io_meta.py` или `utils.py` (AW4, P10).
- **F36.** Документировать `RETRY` как мёртвый вердикт или реализовать (P8).

### P3 — Косметика (SEV4)

- **F37.** Разделить compute и matplotlib (R4) — постепенно, не блокирует.
- **F38.** Исправить stage mislabel в `apd_agent` (A9).
- **F39.** `rsm_mask_worker`: `Path.with_suffix` вместо string replace (MW5).
- **F40.** `pct=55` вынести в единый параметр (AM2).
- **F41.** Синхронизировать лейблы графиков (AW6).

---

## V. КАК РАЗОБРАТЬСЯ С ПАЙПЛАЙНОМ: СТРАТЕГИЯ

### Шаг 0. Немедленная диагностика (до любого кода)

```bash
# 1. Есть ли .gsh у ваших образцов?
find . -name "*.gsh" | head -5

# 2. Что там за fps?
python3 -c "
with open('sample.gsh', 'rb') as f:
    header = f.read(512)
print(header)
"

# 3. Воспроизводится ли ImportError прямо сейчас?
python3 -c "from source_cv_agent import *"

# 4. Какой exit дают агенты на тестовом образце?
python3 activation_agent.py ... ; echo "Exit: $?"
```

### Шаг 1. Минимальный рабочий каркас (1–2 дня)

Создать `pipelines/io_meta.py`:
```python
def read_fps_from_gsh(gsh_path: str) -> float:
    """Единственное место, где fps читается из заголовка."""
    # ... парсинг .gsh
    if fps is None:
        raise ValueError(f"fps not found in {gsh_path}")
    return fps

def canonical_pixel_size_mm() -> float:
    from conduction_analysis import PIXEL_SIZE_MM
    return PIXEL_SIZE_MM

def detect_dye_from_metadata(metadata: dict) -> str:
    """Единственный dye-detector."""
    ...
```

Создать `pipelines/qc_thresholds.yaml`:
```yaml
peak_detection:
  min_threshold_floor: 0.3   # никогда не занижать
  min_beats_floor: 2

activation:
  tat_min_ms: 5
  tat_max_ms: 100
  beat_pass_rate_reject: 0.3
  beat_pass_rate_fail: 0.3   # синхронизировать с judge

apd:
  apd_min_ms: 5
  apd_max_ms: 200

conduction_velocity:
  cv_min: 0.05
  cv_max: 2.0   # единый для всех методов
```

### Шаг 2. P0 фиксы (порядок исполнения)

1. **F1** → `io_meta.read_fps_from_gsh` → заменить все дефолты fps.
2. **F7** → починить `judge()` воркера (P1 + P2 — quarantined exit).
3. **F5** → починить ImportError `cv_method_local_fit` (иначе `source_cv_agent` не запустится).
4. **F8** → зафиксировать схему ключей Stage 5 (P3 — иначе агрегация ломается на хорошем пути).
5. **F3 + F4** → NaN→undetermined + code-floor порога + сверка биений.
6. **F2** → единый pixel_size_mm.
7. **F6** → A/B ветка в `apd_agent`.

### Шаг 3. Тест на известном аномальном образце

```bash
# SHAM 5-08 002A bsl-6Hz должен НЕ давать 'normal'
python3 optical_pipeline_worker.py --sample SHAM_5-08_002A_bsl-6Hz ...
# Ожидаемый результат: exit 1 + phenotype='anomalous'/'undetermined', НЕ 'normal'
```

### Шаг 4. P1 фиксы (воспроизводимость)

После P0 — все ms-числа станут корректными. Тогда:
- Синхронизировать пороги (F10–F23).
- Добавить provenance хэши (F16).

### Шаг 5. P2 фиксы (дедупликация, постепенно)

Дедуп не блокирует корректность, но накапливает дрейф. Делать по одному модулю:
1. fps → `io_meta` (уже в Шаге 1).
2. `detect_dye` → `io_meta` (убрать 3 копии).
3. карта активации → `conduction_analysis` (убрать дубль из `activation_agent`).
4. детектор биений → один модуль.

---

## VI. МЕТРИКИ ПРОГРЕССА

| Метрика | Сейчас | Цель P0 | Цель P1 |
|---------|--------|---------|---------|
| Файлов с fps-дефолтом 1000 | 5 | 0 | 0 |
| SEV1 открытых | ~15 | 0 | 0 |
| Агентов с exit 0 на NaN/FAIL | 5 | 0 | 0 |
| Копий dye-detect | 3 | 3 | 1 |
| `cv_method_local_fit` ImportError | ДА | НЕТ | НЕТ |
| SHAM 002A → 'normal' | ДА | НЕТ | НЕТ |

---

## VII. СЛЕДУЮЩИЕ ФАЙЛЫ ДЛЯ РЕВЬЮ (гипотезы)

Не проревьюированы: `stage_mask`, `stage_cv`, `stage_ows`, `stage_phase_df`, `stage_phenotype`, `peak_detector_agent`, `qc_thresholds.yaml`.

**Гипотезы:**
- `stage_cv`: вероятно дублирует CV-логику (R3); проверить, передаётся ли fps от воркера.
- `peak_detector_agent`: 4-я копия detect_dye? Проверить `detect_dye` сигнатуру (дефолт "A"?).
- `qc_thresholds.yaml`: существует ли вообще? Если да — `judge()` воркера его молча игнорирует (P1, односторонние пороги).
- `stage_phenotype`: финальный фенотип строится на downstream от всего вышеперечисленного — если NaN→'normal' не починен, фенотип будет систематически смещён.
