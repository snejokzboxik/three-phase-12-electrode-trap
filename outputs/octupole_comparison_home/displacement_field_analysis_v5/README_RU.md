# Физический анализ уже решённых смещённых P4-полей

В каталоге обобщены 11 существующих смещённых геометрий и 12 решённых P4-полей. Новые FEM-расчёты и RF-траектории здесь не запускались, исходные крупные коэффициенты не копируются.

Главные материалы: [научный отчёт](REPORT_RU.md), [масштабирование найденного минимума](effective_minimum_scaling.csv), [коэффициенты гармоник](harmonic_amplitude_scaling.csv), [локальная кривизна и saddle-кандидаты](stationary_structure_v5.json), [проверка чисел](numerical_verification_v5.json).

Порядок пересборки: `python analyze_existing_fields.py`, `python stationary_structure_v5.py`, `python render_displacement_analysis_v5.py`, `python validate_displacement_analysis_v5.py`. Анализ ссылается на FEM-сетки и коэффициенты из соседнего каталога `../electrode_displacement_sensitivity_v1/`.
