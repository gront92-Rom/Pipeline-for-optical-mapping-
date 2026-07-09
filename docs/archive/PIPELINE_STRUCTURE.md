# Cardiac Optical Pipeline — Полная структура (по ревью)

**Дата обновления:** 2026-06-29  
**Статус:** Структура на основе реального ревью существующих файлов + предложения по агентам.

---

## 1. Текущая последовательность шагов (реальный пайплайн)

| № | Шаг / Стадия                  | Основные действия                                                                 | Вход                                      | Выход                                              | Текущие файлы / модули                          | Ключевые проблемы |
|---|-------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------|-------------------------------------------------|-------------------|
| 1 | **Загрузка сырых данных**    | Чтение .gsd → .rsh → .rsm (видео) + .gsh (заголовок). Извлечение fps.            | .gsd файл                                 | `loaded_video.npy`, `paths.json`                   | `loader_agent`, `optical_pipeline_worker.stage_load`, `_resolve_fps` | fps читается **только здесь**. Downstream использует дефолт 1000. |
| 2 | **Маскирование ткани**       | Адаптивный thresholding, composite_score, contour-LPF.                            | `loaded_video.npy`                        | `mask.npy`, `metrics.json`                         | `rsm_mask_worker_v2/v3.py`, `auto_mask.py`      | — |
| 3 | **Карта активации**          | 4 метода детекции → TAT в мс → judge (PASS/WARN/REJECT)                           | `loaded_video.npy` + `mask.npy`           | `activation_map.npy`, `activation_peaks.npy`, `data_inv.npy` | `activation_agent.py`                           | Узел fps-ошибки для CV. |
| 4 | **APD по трейсам**           | detect_activation → measure_apd (30/50/80) → validate                             | activation_map или трейсы + маска         | APD80 (и др.), `apd_per_beat_3d.npz`               | `apd_agent.py`                                  | `dt_ms=1.0` захардкожен. |
| 5 | **Скорость проведения (CV)** | Structure Tensor / gradient → векторное поле → cvl/cvt (м/с)                      | `activation_map.npy` + `mask.npy`         | `cvl.npy`, `cvt.npy`                               | `conduction_analysis.py` + `source_cv_agent.py` | — |
| 6 | **Детекция альтернанса**     | AC по парам биений → FFT → spatial/temporal + фенотип                               | `apd_per_beat_3d.npz`                     | alternans карты, `ac_ms_median`, фенотип           | `alternans_detection.py` + `alternans_worker.py`| — |
| 7 | **OWS / Фазовый анализ**     | Фазовый анализ и OWS-метрики (не ревьюировано)                                    | —                                         | —                                                  | `stage_ows`, `stage_phase_df`                   | Не изучено. |
| 8 | **Фенотипирование**          | Агрегация всех метрик → финальный фенотип                                         | Метрики всех предыдущих стадий            | Итоговый фенотип                                   | `stage_phenotype`                               | NaN → 'normal' по умолчанию. |

---

## 2. Предлагаемые агенты (модульная структура)

| № | Имя агента                  | Основная ответственность                                                                 | Вход (обязательный)              | Выход (MUST)                          | Должен вызывать предыдущий (lazy)? | Примечание |
|---|-----------------------------|------------------------------------------------------------------------------------------|----------------------------------|---------------------------------------|------------------------------------|------------|
| 1 | **LoaderAgent**            | Загрузка сырых данных + надёжное извлечение `fps` из .gsh                               | .gsd / пути                      | `loaded_video.npy`, `paths.json`, fps | Нет                                | **Единственный** источник fps. |
| 2 | **MaskAgent**              | Генерация маски ткани (адаптивный thresholding + composite_score)                       | `loaded_video.npy`               | `mask.npy`, `metrics.json`            | Нет (первый после загрузки)        | Можно сделать lazy из downstream. |
| 3 | **ActivationAgent**        | Построение карты активации (4 метода) + judge + TAT в мс                                | loaded_video + mask              | `activation_map.npy`, peaks, judge    | Да                                 | Главный источник activation_map. |
| 4 | **APDAgent**               | Расчёт APD30/50/80 + `apd_per_beat_3d.npz`                                              | activation_map + mask            | APD карты, `apd_per_beat_3d.npz`      | Да                                 | Убрать хардкод `dt_ms`. Брать из Activation. |
| 5 | **ConductionAgent**        | Расчёт CV (cvl/cvt) через Structure Tensor / gradient + source_cv                       | activation_map + mask            | `cvl.npy`, `cvt.npy` (м/с)            | Да                                 | Можно разделить на CV + SourceCV. |
| 6 | **AlternansAgent**         | Детекция альтернанса (AC + FFT + spatial + фенотип)                                     | `apd_per_beat_3d.npz`            | alternans карты + фенотип             | Да                                 | — |
| 7 | **OWSAgent** / **PhaseAgent** | Фазовый анализ и OWS-метрики (пока не ревьюировано)                                   | activation / apd                 | OWS/phase метрики                     | По необходимости                   | Оставить как есть или вынести. |
| 8 | **PhenotypeAgent**         | Финальное фенотипирование на основе всех upstream метрик                                | Все метрики upstream             | Итоговый фенотип                      | Да (агрегирует)                    | **Обязательно** исправить NaN-handling. |

---

## 3. Зависимости между агентами (кто от кого зависит)

- **LoaderAgent** → MaskAgent → ActivationAgent → (APDAgent, ConductionAgent)
- ActivationAgent → APDAgent → AlternansAgent
- ActivationAgent + MaskAgent → ConductionAgent
- Все предыдущие → PhenotypeAgent

**Lazy-правило (рекомендуется):**
Каждый агент (начиная с ActivationAgent) в начале `run(sample_id)`:
1. Проверяет наличие своих входных файлов (маска, activation_map и т.д.).
2. Если отсутствует — вызывает соответствующий предыдущий агент.

---

## 7. Пути к файлам и как найти код для каждого агента

### Общие советы по поиску в проекте

1. **Поиск по имени файла** (в терминале из корня проекта):
   ```bash
   find . -name "*loader*" -o -name "*mask*" -o -name "*activation*" -o -name "*apd*" -o -name "*conduction*" -o -name "*alternans*" | head -30
   ```

2. **Поиск по ключевым функциям**:
   ```bash
   grep -r "_resolve_fps" . --include="*.py"
   grep -r "composite_score" . --include="*.py"
   grep -r "activation_map" . --include="*.py"
   ```

3. **Поиск по классам/функциям**:
   - Ищи файлы с `class .*Agent` или `def stage_`
   - Ищи `optical_pipeline_worker`, `rsm_mask_worker`

### Конкретные рекомендации по агентам

| Агент              | Где искать код сейчас                          | Что брать / рефакторить                                      | Ключевые функции / переменные                  | Примечание |
|--------------------|------------------------------------------------|--------------------------------------------------------------|------------------------------------------------|------------|
| **LoaderAgent**    | `loader_agent.py` или `optical_pipeline_worker.py` (метод `stage_load`) | Функцию загрузки `.gsd` → `.rsh` → `.rsm` + `.gsh`          | `_resolve_fps`, чтение заголовка `.gsh`        | **Самое важное** — централизовать fps здесь |
| **MaskAgent**      | `rsm_mask_worker_v2.py`, `rsm_mask_worker_v3.py`, `auto_mask.py` | Логику адаптивного thresholding + `composite_score` + contour-LPF | `composite_score`, пороги, LPF                 | Уже довольно модульная |
| **ActivationAgent**| `activation_agent.py`                          | 4 метода детекции + judge + расчёт TAT в мс                  | 4 метода (50pct, derivative_max и др.), `judge` | Главный источник `activation_map` |
| **APDAgent**       | `apd_agent.py`                                 | `detect_activation`, `measure_apd`, работа с `apd_per_beat_3d` | `dt_ms=1.0` (нужно убрать хардкод)             | Исправить зависимость от fps/dt |
| **ConductionAgent**| `conduction_analysis.py` + `source_cv_agent.py` | Structure Tensor / gradient, расчёт cvl/cvt                  | `conduction_analysis`, source_cv функции       | Библиотека + standalone agent |
| **AlternansAgent** | `alternans_detection.py` + `alternans_worker.py` | AC, FFT, spatial/temporal alternans, фенотип                 | `ac_ms_median`, alternans coefficient          | Зависит от APD |
| **OWSAgent**       | `stage_ows`, `stage_phase_df` (внутри воркера) | Фазовый анализ и OWS-метрики                                 | —                                              | Пока не ревьюировано — найти и изучить |
| **PhenotypeAgent** | `stage_phenotype` (внутри воркера)             | Агрегация метрик и финальная классификация                   | Логика обработки NaN                           | **Критично** исправить NaN → 'normal' |

### Полезные поисковые команды

```bash
# Найти все файлы с "agent" или "worker"
find . -name "*agent*.py" -o -name "*worker*.py" | sort

# Найти где определяется fps
grep -r "fps" . --include="*.py" | grep -E "(1000|resolve| gsh)" | head -20

# Найти все упоминания activation_map
grep -r "activation_map" . --include="*.py" | head -15
```

### Что делать дальше с этими путями

1. Открой каждый файл из таблицы выше.
2. Выдели основную логику в отдельные функции/классы.
3. Перенеси в новую структуру (`src/cardiac_pipeline/agents/`).
4. Создай общий `BaseAgent`, который будет предоставлять `save_must`, `save_debug`, проверку путей и lazy-вызовы.

---

**Этот раздел добавлен специально**, чтобы в любой сессии можно было быстро понять:
- Где лежит реальный код для каждого будущего агента.
- Что именно нужно взять из старых файлов.
- Какие ключевые функции/проблемы искать.

Если укажешь корневую папку своего проекта (например `/home/user/cardiac_project/`), я смогу дать более точные команды `find`/`grep` под твою структуру.

## 4. Ключевые проблемы (из ревью)

1. **fps handling** — критично. Только Loader читает корректно. Нужно централизовать.
2. **Временные единицы** — несогласованность (activation в мс, APD dt_ms=1.0 захардкожен).
3. **NaN handling** в фенотипе — приводит к смещению в 'normal'.
4. **Модульность** хорошая, но нет единого BaseAgent и общего runner'а с lazy-вызовами.
5. OWS/Phase — почти не изучено.

---

## 5. Рекомендуемая файловая структура (для новой модульной версии)

```
cardiac_optical_pipeline/
├── config/
│   └── default.yaml
├── src/
│   └── cardiac_pipeline/
│       ├── __init__.py
│       ├── config.py                 # PipelineConfig
│       ├── runner.py                 # CardiacPipeline (оркестратор)
│       ├── base_agent.py             # BaseAgent (пути, save_must/debug, lazy)
│       ├── agents/
│       │   ├── loader_agent.py
│       │   ├── mask_agent.py
│       │   ├── activation_agent.py
│       │   ├── apd_agent.py
│       │   ├── conduction_agent.py
│       │   ├── alternans_agent.py
│       │   ├── ows_agent.py          # (пока заглушка)
│       │   └── phenotype_agent.py
│       └── utils/
│           ├── data_loader.py
│           ├── masking.py
│           ├── activation.py
│           ├── apd.py
│           ├── conduction.py
│           └── alternans.py
├── data/          # сырые .gsd и т.д.
├── results/       # must/ и debug/ по sample_id
└── PIPELINE_STRUCTURE.md
```

---

## 6. Следующие шаги (рекомендация)

1. Создать **BaseAgent** с общими методами (пути, сохранение, lazy-проверка).
2. Реализовать **LoaderAgent** с надёжным `_resolve_fps`.
3. Реализовать **MaskAgent** (можно взять логику из auto_mask.py).
4. Добавить lazy-механику в ActivationAgent и дальше.
5. Починить fps и dt_ms propagation.
6. Улучшить NaN-handling в PhenotypeAgent.

---

**Этот файл** (`PIPELINE_STRUCTURE.md`) создан специально, чтобы в любой новой сессии можно было быстро восстановить полную картину: текущие шаги, проблемы, предлагаемые агенты и зависимости.

Готов продолжать — скажи, что делаем дальше (например, начать писать BaseAgent или LoaderAgent).