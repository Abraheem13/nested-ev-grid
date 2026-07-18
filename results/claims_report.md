# Claims verification report

| ID | Claim | Result | Detail |
|----|-------|--------|--------|
| P1 | every (scenario, method) cell has exactly 50 episodes | PASS | min n = 50 |
| P2 | evaluation seeds follow the stated 7000..7049 protocol | PASS |  |
| P3 | no infeasible/failed episodes in the main grid | **FAIL** |  |
| H1 | nested: ZERO recorded violations across S1-S5 | **FAIL** | max viol = 0.065% |
| H2 | nested: min voltage >= 0.95 p.u. in every scenario mean | **FAIL** | min = 0.9490 |
| H3 | nested: delivers 97-98% of requested energy (paper claim) | PASS | SQ range 0.966-0.982; if outside 0.97-0.98, update the abstract wording |
| H4 | Pareto uniqueness: NO baseline is both safe and complete in any scenario | PASS |  |
| H5 | uncoordinated/TOU violate at high penetration (S3-S5) | PASS |  |
| T1 | all S3 cost comparisons Holm-significant at p<0.05 | PASS | uncoordinated:5.221e-30; tou:4.45e-09; milp:2.715e-97; flat_ddpg:1.431e-91; ppo_lag:2.526e-99; cpo:2.572e-92; hrl:1.723e-96 |
| T2 | near-deterministic voltage outcome (sigma <= 0.0003 p.u.) | PASS | sigma = 1.23e-04 |
| S6 | deadlock scenario: zero violations with forced P=0 window | PASS | viol = 0.000%, SQ = 0.976 |
| S7 | capacity exhaustion: graceful degradation (0 < viol <= 2%, vmin >= 0.94) | PASS | viol = 0.66%, vmin = 0.9481 |
| A1 | removing Level 3 breaks the voltage floor (vmin < 0.95, violations reappear) | PASS | vmin = 0.9457, viol = 4.1% |
| A2 | every ablation is strictly worse than the complete framework on cost | **FAIL** |  |
| G1 | IEEE 69-bus: nested keeps zero violations | **FAIL** | vmin = 0.9501 |
| G2 | cross-dataset (ACN, ElaadNL): zero violations, SQ >= 0.95 | **FAIL** | acn: SQ=0.901; elaad: SQ=0.953 |
| V1 | zero violations across the delta and S-rating sweeps | **FAIL** |  |
