# RF-off control v1

Этот каталог содержит только аналитическую обработку опубликованных V1/V2 RF-результатов. Новых RF-интеграций и FEM расчётов здесь нет.

## Воспроизведение

Из корня проекта выполните в PowerShell:

```powershell
python outputs\octupole_comparison_home\rf_off_control_v1\rf_off_analysis.py
python -m unittest discover -s outputs\octupole_comparison_home\rf_off_control_v1 -p 'test_*.py' -v
```

Скрипт читает `v2/trajectory_results.csv`, сверяет его с `v2/trajectory_checkpoint.jsonl`, использует таблицу V2 E90 из `v2/energy_threshold_intervals.csv` и отдельно повторно использует восемь завершённых V1 результатов на 1000 RF-периодах. Модульные тесты включают аудит реальных V2 входов. Никакие зависимости не устанавливаются.

## Основные результаты

- `three_way_matched_starts.csv` — 1 160 уникальных физических стартов; обе RF-классификации, аналитический RF-off исход на 300/1000/10000 периодах и парные разности удержания.
- `ballistic_crossing_times.csv` — точные координаты и скорости, время первого баллистического пересечения и горизонты RF-off.
- `energy_retention_comparison.csv`, `radial_retention_comparison.csv` — прямые ячейки retention, без интерполяции.
- `energy_threshold_comparison.csv`, `ballistic_horizon_thresholds.csv` — интервалы V2 и аналитические пороги по массе и горизонту.
- `escape_time_pairs.csv` — только пары, в которых обе системы реально вылетели; retained RF времена оставлены цензурированными в основном matched CSV.
- `v1_1000RF_matched_control.csv` — валидные прежние 1000-периодные RF запуски, без повторных интеграций.
- `input_audit.json`, `analysis_summary.json` — контроль входных данных и сводные исходы.
- `figure_1_...` — `figure_5_...`: PNG, SVG и PDF.

Основной вывод и научные ограничения: [RF_OFF_CONTROL_REPORT_RU.md](RF_OFF_CONTROL_REPORT_RU.md).
