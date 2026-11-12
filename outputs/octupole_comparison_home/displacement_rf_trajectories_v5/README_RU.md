# Прицельная RF-динамика на смещённых P4-полях — V5

Каталог содержит одну matched-пару режимов на пяти геометриях: nominal и radial-out смещения 10, 25, 50 и 100 µm. Для каждой геометрии использовано уже решённое полнодоменное h50 P4-поле. Новых FEM-решений не выполнялось.

Основные файлы:

- [научный отчёт](REPORT_RU.md)
- [настройки и начальные условия](pilot_config_v5.json)
- [контроль точного совпадения старта с V4 и знака скорости](initial_condition_audit.json)
- [таблица matched-результатов](processed_trajectory_pairs_v5.csv)
- [численная проверка](numerical_checks_v5.json)
- [повторы с более строгими допусками](numerical_refinement_comparison_v5.json)
- [контрольная запись по сегментам](checkpoints.jsonl)
- [исходные сегменты и сводки траекторий](trajectories/)
- [обработанные временные ряды](trajectory_samples_combined_v5.csv)
- [рисунки](figures/)

Воспроизводимый запуск после существующего benchmark-чекпойнта:

```powershell
python run_targeted_displacement_v5.py
python run_targeted_displacement_v5.py --tight-validation
python process_and_render_v5.py
python validate_displacement_rf_v5.py
```

Первый запуск содержит `--benchmark-only`, который сохранён в `trajectories/nominal_rotating_standard/`. Скрипт продолжает этот путь с 100 RF и сверяется с общим V5 deadline из `../matched_pseudopotential_v5/integration_deadline.json`.

Ключевые ограничения: один выбранный детерминированный старт; наблюдение до 1000 RF; показатели радиуса вычислены по сохранённым временным сэмплам; результаты не задают статистику частиц и не устанавливают допуск изготовления.
