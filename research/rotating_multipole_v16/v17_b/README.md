# V17B — рабочая ветка исследования

V17B продолжается внутри `rotating_multipole_v16`, переиспользуя валидированные `v16_1/dynamics.py` и `Harmonic`/`potential_jet`/`rigid_rhs`/`lab_jacobian`.

## Воспроизведение

Из каталога `rotating_multipole_v16`:

```powershell
python -m v17_b.analyze
python -m v17_b.standing_lock
python -m v17_b.electrode_synthesis
python -m v17_b.plot_basin_pilot
python -m v17_b.plot_standing
python -m pytest -q
```

`basin_pilot.py` и `standing_basin.py` запускают 1000 RF на траекторию и сохраняют checkpoint после каждого старта под `v17_b/results/`. Временные горизонты длинные; seeds, solver tolerances и состояния зафиксированы в JSON. Используется DOP853 с остановкой по потере полной апертуры.

## Текущий статус

- Завершено: signed charge-moment expansion; нейтральный симметричный диполь в RO3; полный локальный 6D спектр и лабораторная monodromy; SO3 periodic locks со средним spin `±1/3`, полная Floquet-монодромия и 1000-RF локальные пилоты обоих знаков; 12-electrode DFT alias audit.
- Частично: целевой литературный обзор (полные тексты Walz 1993/1994 недоступны через издательский веб-доступ), локальные карты устойчивости, статистика бассейна и физические масштабы.
- NOT_RUN: Arnold tongues/continuation, SO4 standing octupole maps, broad basin comparison, realistic multipole electrode FEM, 3D dynamics, full physical drag.
- GitHub push не завершён из-за отсутствия write credential; локальная ветка `research/v17b` и коммиты сохранены.

См. `REPORT_RU.md`, `LITERATURE_REVIEW_RU.md`, `STANDING_WAVE_DYNAMICS_RU.md` и `MANIFEST.json`.
