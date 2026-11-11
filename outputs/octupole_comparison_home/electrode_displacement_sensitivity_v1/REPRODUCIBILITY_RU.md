# Воспроизводимость

Параметры геометрии, шаблоны смещений, электродные фазы, бюджет и интегратор зафиксированы в `pilot_config.json`. Версии библиотек записаны в `requirements-reproducibility.txt`; локальная папка `.deps/` исключена из Git.

## Сборка рисунков без расчётов

Из каталога `electrode_displacement_sensitivity_v1`:

```powershell
python render_displacement_figures.py
```

Скрипт использует готовые `processed/*.csv`, валидированные trajectory segment CSV и валидатор минимума. Он не запускает FEM, поле или RF интеграции. Результаты записываются в `figures/` как PDF, SVG и PNG 600 dpi.

## Ключевые проверки, уже сохранённые в наборе

- Направления всех начальных скоростей и непрерывность сегментов: `processed/trajectory_direction_audit.json`.
- Исправление знака тангенциальной скорости и исключённые старые траектории: `processed/trajectory_direction_correction.json`.
- Независимый поиск минимума на h35 и h50: `processed/effective_minimum_method_validation.json`.
- Сравнение разрешений одного и того же смещённого поля и номинальный N48 контроль: `processed/resolution_sensitivity_summary.json` и `.csv`.
- Невязки, градиентная проверка, сетки и позиции найденных минимумов: `processed/field_analysis_index.json`, `calculations/*/field_diagnostics.json`, сеточные файлы и локальные solver metadata.
- Сводка парных траекторий с конечновременной цензурой: `processed/matched_rf_trajectory_summary.json`; исходные сегменты и прогресс каждого запуска — `trajectories/*/`.

Скрипты `solve_displaced_p4.py`, `field_analysis.py`, `resolution_check.py`, `run_rf_trajectories.py`, `audit_trajectory_directions.py` и `validate_effective_minimum.py` сохранены для повторения соответствующих стадий. Повторное выполнение расчётных скриптов требует самостоятельного выделенного бюджета и не выполняется при пересборке фигур.
