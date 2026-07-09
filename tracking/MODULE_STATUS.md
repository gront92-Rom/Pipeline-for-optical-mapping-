# Module Status — cardiac_pipeline_v3

**Последнее обновление:** 2026-07-08  
**Ветка:** `update-pipeline`  
**Архитектура:** BaseAgent + PipelineConfig (OmegaConf), `config/default.yaml` — единый конфиг

---

## Легенда

| ✅ Готов | 🔄 В процессе | ⚠️ Частично | ❌ Не начато | 🚫 N/A |

---

## Стадии и агенты

### Stage 0 — SidelineAgent / `sideline_agent.py` (168 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ❌ (нет test_sideline_smoke.py) |
| Интеграция в оркестратор | ❌ (нет оркестратора) |
| Логика | < 4096 кадров → pass; ≥ 4096 → sideline_isolated |
| Блокеры | Нет |

### Stage 1 — LoaderAgent / `loader_agent.py` (416 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ✅ 23/23 PASS |
| F42 fix (_find_files_by_stem) | ✅ Исправлен 2026-07-08 |
| recording_mode из metadata | ✅ Добавлен 2026-07-08 |
| compute_dominant_freq (FFT) | ✅ Подключён 2026-07-08 |
| E2E тест на реальном .rsh | ✅ 6.4s, 004A (2048×100×100, fps=1000) |
| Полярность A/B | ✅ Уже была исправлена (preprocess v6) |
| Блокеры | Нет |

### Stage 2 — MaskAgent / `mask_agent.py` (483 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ❌ |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен тест |

### Stage 3 — PeakDetectorAgent / `peak_detector_agent.py` (312 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ❌ |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен тест |

### Stage 4 — ActivationAgent / `activation_agent.py` (516 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ❌ |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен тест |

### Stage 5 — APDAgent / `apd_agent.py` (578 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ✅ 29/29 PASS |
| Демо (run_apd_demo.py) | ✅ Синтетический трейс |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен E2E |

### Stage 6 — ConductionAgent / `conduction_agent.py` (442 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ✅ ALL PASS |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен E2E |

### Stage 6b — ConductionConsensusAgent / `conduction_consensus_agent.py` (305 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ❌ CLI-агент, не BaseAgent |
| Интеграция | ⚠️ standalone CLI, не часть v3 chain |
| BUG-1..5 фиксы | ✅ (ImportError, pixel_size, exit codes, NaN) |
| Блокеры | Нужноrefactor → BaseAgent |

### Stage 7 — AlternansAgent / `alternans_agent.py` (434 строк)

| Аспект | Статус |
|--------|--------|
| Наследует BaseAgent | ✅ |
| run() | ✅ |
| smoke-тест | ✅ 28/28 PASS |
| E2E на реальных данных | ❌ |
| Блокеры | Неизвестно — нужен E2E |

### Stage 8 — PhenotypeAgent

| Аспект | Статус |
|--------|--------|
| Существует | ❌ Не реализован |
| Блокеры | Зависит от всех upstream |

---

## Утилиты

| Файл | Строк | Статус | smoke-тест |
|------|-------|--------|------------|
| `base_agent.py` | — | ✅ | (в каждом agent-тесте) |
| `utils/preprocess.py` v6 | — | ✅ | ✅ 23/23 PASS |
| `utils/signal.py` | — | ✅ | ✅ (в apd_smoke) |
| `utils/alternans.py` | — | ✅ | ✅ (в alternans_smoke) |
| `utils/cv_estimators.py` | — | ✅ | ✅ (в cv_smoke) |
| `utils/metadata_extractor.py` | — | ✅ (F42 fixed) | ✅ (в loader_smoke) |

---

## Оркестратор / Runner

| Аспект | Статус |
|--------|--------|
| Существует | ❌ Не реализован |
| Назначение | Lazy graph: загрузить результаты upstream по demand |
| Блокеры | Ждёт ConductionConsensus → BaseAgent refactor |

---

## Smoke-тесты — сводка

| Тест | Тестов | Результат |
|------|--------|-----------|
| test_loader_smoke.py | 23 | ✅ 23/23 PASS |
| test_preprocess_smoke.py | 23 | ✅ 23/23 PASS |
| test_apd_smoke.py | 29 | ✅ 29/29 PASS |
| test_cv_smoke.py | ~5 | ✅ ALL PASS |
| test_alternans_smoke.py | 28 | ✅ 28/28 PASS |
| test_consensus_agent_cli.py | — | ⚠️ CLI-тест, не pytest |
| **Итого** | **~108** | **✅ ALL PASS** |

---

## Что осталось сделать

1. **Оркестратор (runner.py)** — lazy graph, запуск цепочки Stage 0→8
2. **ConductionConsensusAgent → BaseAgent** — refactor из CLI в v3-агента
3. **PhenotypeAgent** — Stage 8, агрегация метрик
4. **E2E тесты** — MaskAgent, PeakDetector, Activation на реальных .rsh
5. **test_sideline_smoke.py** — синтетическое видео ≥ 4096 кадров
6. **OWS / Phase** — не реализовано в v3 (было в старом воркере)