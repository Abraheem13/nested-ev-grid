# NFL-EV v2 — Nested Learning for Multi-Timescale EV Charging Coordination

Rebuild + revision codebase targeting **ACM TAAS**. Every module maps to a
reviewer concern (R1.x / R2.x / R3.x annotations in docstrings).

## Reviewer-concern → code map
| Concern | Fix | Where |
|---|---|---|
| R1.2/R2.2 deadlock | Q from inverter S-rating: `Qmax=√(S²−P²)`, full Q at P=0 | `env/qcontrol.py` |
| R2.2/R2.3 guarantee | v_target margin + Newton closure + curtailment fallback; S6/S7 stress scenarios | `env/qcontrol.py`, `configs/base.yaml` |
| R1.3 price discontinuity | L2 agents select execution price within corridor (extra action dim) | `env/charging_env.py::set_dispatch` |
| R1.5 sum constraint | Projection onto bounded simplex by construction | `agents/projection.py` |
| R1.4 temporal coupling | Level 3 split: behavior @15-min (L3a), Q control within-step (L3b) | `env/behavior.py`, `env/qcontrol.py` |
| R3.4 DSO constraints | Line loading + substation capacity in L1 state & reward | `env/network.py::constraint_features` |
| R3.1 assumptions | Explicit nominal operating point (OLTC 1.03 p.u., 0.85 base scale) | `configs/base.yaml` |
| R1.7 curriculum metric | Advancement on cost + Q-activation + curtailment frequency | `configs/base.yaml` |
| R2.4/R3.6 baselines | PPO-Lagrangian, CPO, HRL baseline (Day 4) | `agents/` |

## Simulation model
Quasi-static time series at 60 s (configurable to 1 s for the high-resolution
validation experiment), full AC Newton–Raphson power flow (pandapower, warm
start), within-step corrective Q convergence.

## Run
```
pip install -r requirements.txt
python tests/test_env.py          # Day-1 smoke suite
```
# nested-ev-grid
