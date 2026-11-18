# Публикационный срез V16, V16.1 и V17B

Каталог содержит исходную двумерную модель V16, валидированное расширение V16.1 и аналитический/численный этап V17B.

- Базовый commit репозитория: `e5e6b2473bf519c5670ee1c48882183fbcfc8f2d` (`master`).
- Публикационная ветка: `research/v17b`.
- V16/V16.1: `18 passed`; V17B snapshot вместе с ними: `22 passed` (`python -m pytest -q` из каталога `rotating_multipole_v16`).
- V17B completed slice: signed charge moments; neutral symmetric dipole in ideal RO3; full 6D rotating-frame Jacobian; independent DOP853 lab monodromy; 1000-RF local basin pilot; 12-electrode DFT alias audit.
- V17B `PARTIAL`: standing-field and Arnold-tongue continuation, broad basin maps, physical multipole geometry and FEM, 3D dynamics remain explicitly `NOT_RUN`.
- FEM h50/h35 arrays are absent and are not replaced by h70; V17B does not reuse old c4 for a new electrode geometry.
- Excluded: Python/pytest caches, `.pyc`, temporary runner logs/locks and unavailable heavy FEM basis arrays.

GitHub write access failed with `403 Resource not accessible by integration`; local commits are retained on this branch. No force push or master update was performed.
