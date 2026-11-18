# V17B — рабочая ветка исследования

V17B продолжается внутри `rotating_multipole_v16`, переиспользуя валидированные `v16_1/dynamics.py` и `Harmonic`/`potential_jet`/`rigid_rhs`/`lab_jacobian`.

## Воспроизведение

Из каталога `rotating_multipole_v16`:

```powershell
python -m v17_b.analyze
python -m v17_b.electrode_synthesis
python -m v17_b.plot_basin_pilot
python -m pytest -q
```

`basin_pilot.py` запускает 1000 RF на траекторию и сохраняет checkpoint после каждого старта в `v17_b/results/basin_local_checkpoint.json`. Параметры и seeds записаны в JSON; пилот использует DOP853 и останавливает интегрирование при потере апертуры.

## Текущий статус

- Завершено: signed charge-moment expansion, RO3 neutral-dipole exact center lock, полный локальный 6D спектр и независимая лабораторная monodromy, небольшой nonlinear capture pilot, 12-electrode DFT alias audit.
- Частично: литература (полные тексты Walz 1993/1994 недоступны через издательский веб-доступ), карты локальной устойчивости и физические масштабы.
- NOT_RUN: Arnold tongues/continuation, SO/RO nonlinear comparison, realistic multipole electrode FEM, 3D dynamics, full physical drag and application validation.
- GitHub push не завершён из-за отсутствия write credential; локальная ветка `research/v17b` и коммиты сохранены.

См. `REPORT_RU.md`, `LITERATURE_REVIEW_RU.md` и `MANIFEST.json`.
