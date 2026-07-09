# Pipeline Review — Session 4

> Файлы: `alternans_worker.py` (652), `optical_pipeline_worker.py` (3066).
> `optical_pipeline_worker.txt` оказался байт-в-байт копией `.py` — отдельным файлом не считается.
> Охват: `alternans_worker` — почти full (ядро + main + спектр; plot_maps/plot_traces только структурно).
> `optical_pipeline_worker` — **sampled**: judge / retry-loop / run_pipeline / _resolve_fps / _inject_globals /
> stage_load / stage_activation / stage_alternans / хвост stage_apd / __main__ + grep. НЕ читаны целиком:
> stage_mask, stage_cv, stage_source_cv, stage_ows, stage_phase_df, stage_phenotype, голова stage_apd,
> CSV/summary/plot-хелперы. Это не аудит.

---

### alternans_worker.py (652) — Stage 5 standalone (spatial+temporal alternans) — [почти full]
Сильное: fps не нужен (домен «beats», `d=1.0` в FFT) → R1 здесь чист; min-beats реально гейтит с `exit(1)`.

- A1 [SEV1] Тихий проход: стадия не гейтит результат и **всегда exit 0**. Гейты только на наличие файлов и `n_beats` (286–317). Метрики падают в `0` при пустом/NaN-входе (371–389) — `ac_ms_median: 0` от «нет данных» неотличим от «реально нет альтернанса». Нет `judge()`. (R2)
- A2 [SEV1] `f_dom` = `argmax` спектра, `purity` считается на нём же **без проверки ~0.5 cycles/beat** (330–341). Тренд/дрейф APD даёт низкочастотный пик с высокой «purity», отрапортованной как чистота альтернанса. Контракт докстринга («half-rate») в коде не enforced.
- A3 [SEV2] Нет provenance: в `report` нет хэша входного `apd_per_beat_3d.npz`, нет commit-hash, `version="1.0.0"` зашит строкой (420–435). Выход нельзя привязать к конкретному входу (регресс относительно большого воркера).
- A4 [SEV2] `detect_dye` дефолтит в `"A"` (124) → молчаливый mislabel CaT→APD при сбое детекта. Дублирует `_detect_dye` воркера. (R3)
- A5 [SEV3] Зашитые физ-пороги: floor знака `0.5 ms` (166, 170) и `discordant < 0.25` (389) не вынесены в параметры (в отличие от `--ac-pct-thresholds`).
- A6 [SEV4] Лейбл-рассинхрон: `main` печатает «cycles/beat», график подписан «Cycles per beat-pair» (641).

Привязка к R1–R4: R2 (A1, A2), R3 (A4 — третья копия dye-детекта в проекте).

---

### optical_pipeline_worker.py (3066) — оркестратор всех стадий — [sampled]
Сильное: fps-root-cause (R1 по fps) **закрыт** — `_resolve_fps` бросает при отсутствии, нет fallback 1000, источник истины — `.gsh` заголовок в `stage_load`. Provenance крепкий: `file_hash`, `commit_hash`, `params_hash`, `attempts.jsonl`.

- P1 [SEV1] `judge()` (2079–2093) пропускает **отсутствующую метрику** (`val is None` → правило молча скипается) И **любой односторонний порог**: гард `lo is not None and hi is not None` означает, что правило `[10, null]` (reject если ниже 10) **никогда не срабатывает**. Floor по peak-threshold / числу биений из MEMORY, если он в `qc_thresholds.yaml` как нижняя граница — `judge()` его молча игнорирует. (R2)
- P2 [SEV1] Quarantined-файл выходит с **exit 0**: `__main__` (2871–2873) печатает причину, но не зовёт `sys.exit(1)`. Оркестратор по returncode видит «успех» для отбракованного файла. (R2)
- P3 [SEV1] Рассинхрон имён метрик в Stage 5: success-путь (subprocess `alternans_worker`) кладёт ключи `spatial_*`/`temporal_*` (1808–1811), fallback-путь — `ac_median`/`ac_gt5_pct` (1847–1851). `run_pipeline` агрегирует `...["metrics"]["ac_median"]` (2526–2527), который есть **только в fallback** → «хороший» путь даёт `KeyError` на финальной агрегации, деградированный работает; `judge` судит пути по разным ключам. (R3 → кандидат R5)
- P4 [SEV1] `pixel_size_mm=0.085` передефинирован дефолтом ≥3 раз (881, 951, 2269). MEMORY: канонический PIXEL_SIZE_MM в `conduction_analysis`, не передефинировать. CV в мм молча неверен при дрейфе. (R1)
- P5 [SEV2] `peak_thr_frac=0.5` (731) — **soft-дефолт, не код-floor**. Нет защиты от занижения; LLM-/preset-попытки (attempt 3–4) могут его опустить. MEMORY требует floor в коде. Связка с P1: даже floor в yaml не enforce-ится.
- P6 [SEV2] MiCAM-crop `20/8` зашит дефолтами функции (401–402) — калибровочная константа. (R1)
- P7 [SEV2] Окно активации в **кадрах, не мс**: `ws=pk-50, we=pk+10` (746–747) не масштабируется с fps, в отличие от соседнего `int(20*fps/1000)` (753). Скрытая fps-зависимость. (R1-смежное)
- P8 [SEV3] `RETRY` — мёртвый вердикт: `judge()` его не возвращает, но retry-loop и докстринг (13–17) трактуют как ядро. Реальные ретраи идут через `REJECT→continue`. Контракт ≠ код. (R4)
- P9 [SEV3] Sentinel-коллизия `stim_hz`: дефолт `10` запускает реparse из имени файла даже при настоящих 10 Гц (2413–2421).
- P10 [SEV3] Тройной `detect_dye`: `peak_detector_agent.detect_dye` (704), `_detect_dye` (241), `alternans_worker.detect_dye` — три реализации, каждая дефолтит в «A». (R3)
- P11 [SEV3] `soft_fail` (332) считает returncode 1 успехом при наличии report — канал тихого прохода для суб-агентов. (R2)
- P12 [SEV4] Дубль ключа `"stage_6_ows"` в `STAGE_DEFS` (2182 и 2189) — второй молча перетирает первый.

Привязка к R1–R4: R1 (P4, P6, P7), R2 (P1, P2, P5, P11), R3 (P3, P10), R4 (P8).

---

### Предложение новой корневой причины (для трекера)
**R5 — Дрейф контракта имён метрик producer↔consumer.** Продьюсер пишет ключи (`alternans_worker` → `spatial_ac_*`), консьюмер читает другие (`run_pipeline` → `ac_median`); требуемый ключ отсутствует → либо тихий PASS в `judge` (missing-metric, P1), либо жёсткий `KeyError` в агрегации (P3). Паттерн в 2 файлах, не ложится чисто в R1–R4.

---

### Приоритеты фиксов
- **P0 (SEV1):** A1, A2, P1, P2, P3, P4
- **P1 (SEV2):** A3, A4, P5, P6, P7
- **P2 (SEV3):** A5, P8, P9, P10, P11
- **P3 (SEV4):** A6, P12
