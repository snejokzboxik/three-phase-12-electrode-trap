# Preserved incomplete version

The v1 P4 pilot batch completed h50 DOP853 coarse/fine at both phases and
the two diagnostic h35 A_fit runs. All direct h35 runs stopped on element
ownership checks; no incomplete trajectory is certified or reused.

A local replay saved in ownership_failure_replay.json found a neighboring
face residual -6.091758081812016e-14, just beyond the 256-epsilon geometric
tolerance 5.684341886080802e-14. A local dense-root fraction must be converted
to a global float64 clock. Its position error needs an explicit envelope.
The amended numerical geometry check is versioned in v2 and documented in
docs/DIRECT_FEM_A_PROTOCOL.md. The old worker and dynamics sources are saved
here and match the frozen v1 settings hashes. Physical force, initial states,
and criteria were not changed. Failed cases are rerun in v2.
