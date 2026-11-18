# Трёхмерная динамика

**Статус:** `NOT_RUN`.

Текущий код использует 2D тело с одной координатой ориентации и не содержит tilt, nutation, тензора инерции, Euler equations или кватернионного интегратора. Шесть собственных значений в отчёте относятся только к состоянию `(x,y,vx,vy,ψ,spin)` и ничего не говорят о трёхмерных наклонах или roll-spin coupling.

До 3D этапа нужны тензорные signed multipoles в body frame, torque через `−∂U/∂rotation`, quaternion kinematics, сохранение нормы quaternion, контроль энергетики и 3D monodromy/Floquet-orbit shooting.
