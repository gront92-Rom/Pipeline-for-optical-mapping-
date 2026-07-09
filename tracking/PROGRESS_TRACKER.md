# Progress Tracker — cardiac_pipeline_v3

**Последнее обновление:** 2026-07-08  
**Ветка:** `update-pipeline`  
**Коммитов:** 5 (последний: `4e06568` — SidelineAgent)

---

## Сводка

| Категория | Всего | Готово | В процессе | Осталось |
|-----------|-------|--------|------------|----------|
| Агенты (Stage 0-7) | 8 | 7 | 0 | 1 (PhenotypeAgent) |
| Утилиты | 5 | 5 | 0 | 0 |
| Smoke-тесты | 6 | 6 ✅ | 0 | 0 |
| E2E тесты (реальные .rsh) | 8 | 1 (Loader) | 0 | 7 |
| Оркестратор | 1 | 0 | 0 | 1 |
| **Итого** | **28** | **19** | **0** | **9** |

Прогресс: ██████████░░░░░░░░░░ 68%

---

## Что сделано

### 2026-07-02 — Базовая архитектура
- BaseAgent + PipelineConfig (OmegaConf)
- config/default.yaml — единый конфиг
- 5 утилит: preprocess v6, signal, alternans, cv_estimators, metadata_extractor
- 8 агентов (Stage 0-7), все наследуют BaseAgent (кроме ConsensusAgent)
- 6 smoke-тестов (108 тестов, все PASS)
- Ветка `update-pipeline` запушена на GitHub

### 2026-07-08 — LoaderAgent fixes
- **F42 fix**: `_find_files_by_stem()` переписана — 3-уровневая стратегия match (sample-id → soft substring → empty). Monkey-patch больше не нужен.
- **recording_mode**: берётся из metadata (header .rsh), fallback из dye
- **compute_dominant_freq()**: FFT → stim_hz_effective, записывается в metadata.json
- **E2E тест**: 004A bsl-6Hz — 6.4s, (2048, 100, 100), fps=1000, dye=A, mode=voltage ✅

---

## Что осталось

### Приоритет 1 — Оркестратор
- `runner.py` — lazy graph оркестратор
- Загрузка результатов upstream по demand (не обязательный последовательный запуск)
- Проверка статусов (pass / sideline_isolated / fail)
- Exit codes: 0=SUCCESS, 1=crash, 2=REJECT/QC

### Приоритет 2 — ConductionConsensusAgent → BaseAgent
- Сейчас CLI-агент (305 строк), не наследует BaseAgent
- Нужно: refactor в v3-стандарт (run() → dict, save_must/load_must)

### Приоритет 3 — E2E тесты на реальных данных
- MaskAgent + PeakDetector + Activation на 004A (уже загружен)
- Цепочка: Stage 2→3→4 на одном реальном sample
- Проверить inter-stage контракт (must/ files)

### Приоритет 4 — PhenotypeAgent (Stage 8)
- Агрегация метрик всех стадий → фенотип
- NaN → 'undetermined' (не 'normal'!)

### Приоритет 5 — Sideline smoke test
- Синтетическое видео 5000 кадров → sideline_isolated
- Синтетическое видео 2000 кадров → pass

### Не реализовано в v3
- OWS (Optical Wave Similarity) — было в старом `optical_pipeline_worker.py`
- Phase analysis — было в старом воркере
- Эти стадии могут быть добавлены позже как отдельные агенты

---

## Архитектура

```
cardiac_pipeline_v3/
├── config/default.yaml          # единый конфиг (OmegaConf)
├── src/cardiac_pipeline/
│   ├── base_agent.py            # BaseAgent + PipelineConfig
│   ├── agents/
│   │   ├── sideline_agent.py    # Stage 0: long-file isolation
│   │   ├── loader_agent.py      # Stage 1: load .rsh → video.npy
│   │   ├── mask_agent.py        # Stage 2: tissue mask
│   │   ├── peak_detector_agent.py  # Stage 3: beat detection
│   │   ├── activation_agent.py  # Stage 4: activation maps (TAT)
│   │   ├── apd_agent.py         # Stage 5: APD30/50/80
│   │   ├── conduction_agent.py  # Stage 6: CVL/CVT
│   │   ├── conduction_consensus_agent.py  # Stage 6b: CLI (не BaseAgent)
│   │   └── alternans_agent.py   # Stage 7: alternans detection
│   └── utils/
│       ├── preprocess.py        # v6: polarity, ASLS, filters
│       ├── signal.py            # APD detection
│       ├── alternans.py         # alternans math
│       ├── cv_estimators.py     # CV methods
│       └── metadata_extractor.py  # .rsh/.gsh parsing
├── test_*.py                    # 6 smoke-тестов (108 tests, ALL PASS)
├── run_apd_demo.py              # APD демо на синтетике
├── tracking/                    # этот файл + MODULE_STATUS.md
└── docs/                        # архитектурные docs (частично архив)
```

---

## Git

| Ветка | Последний коммит | Описание |
|-------|-----------------|----------|
| `update-pipeline` (active) | `4e06568` | SidelineAgent |
| `main` | — | Базовая версия |

Uncommitted: `loader_agent.py` + `metadata_extractor.py` (F42 fix + recording_mode + dominant_freq)