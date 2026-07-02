"""Distribution network builders with operational constraint metadata.

IEEE 33-bus: pandapower built-in (Baran & Wu 1989).
IEEE 69-bus: constructed from standard Baran & Wu (1989) 69-bus data.

Addresses R3.4: line thermal limits and substation transformer capacity are
first-class environment constraints, exposed in the Level 1 state and reward.
"""
from __future__ import annotations
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# IEEE 69-bus radial feeder (Baran & Wu 1989). Columns:
# from_bus, to_bus, r_ohm, x_ohm, p_kw (load at to_bus), q_kvar
_IEEE69 = [
    (1,2,0.0005,0.0012,0,0),(2,3,0.0005,0.0012,0,0),(3,4,0.0015,0.0036,0,0),
    (4,5,0.0251,0.0294,0,0),(5,6,0.3660,0.1864,2.6,2.2),(6,7,0.3811,0.1941,40.4,30),
    (7,8,0.0922,0.0470,75,54),(8,9,0.0493,0.0251,30,22),(9,10,0.8190,0.2707,28,19),
    (10,11,0.1872,0.0619,145,104),(11,12,0.7114,0.2351,145,104),(12,13,1.0300,0.3400,8,5),
    (13,14,1.0440,0.3450,8,5.5),(14,15,1.0580,0.3496,0,0),(15,16,0.1966,0.0650,45.5,30),
    (16,17,0.3744,0.1238,60,35),(17,18,0.0047,0.0016,60,35),(18,19,0.3276,0.1083,0,0),
    (19,20,0.2106,0.0690,1,0.6),(20,21,0.3416,0.1129,114,81),(21,22,0.0140,0.0046,5,3.5),
    (22,23,0.1591,0.0526,0,0),(23,24,0.3463,0.1145,28,20),(24,25,0.7488,0.2475,0,0),
    (25,26,0.3089,0.1021,14,10),(26,27,0.1732,0.0572,14,10),(3,28,0.0044,0.0108,26,18.6),
    (28,29,0.0640,0.1565,26,18.6),(29,30,0.3978,0.1315,0,0),(30,31,0.0702,0.0232,0,0),
    (31,32,0.3510,0.1160,0,0),(32,33,0.8390,0.2816,14,10),(33,34,1.7080,0.5646,19.5,14),
    (34,35,1.4740,0.4873,6,4),(3,36,0.0044,0.0108,26,18.55),(36,37,0.0640,0.1565,26,18.55),
    (37,38,0.1053,0.1230,0,0),(38,39,0.0304,0.0355,24,17),(39,40,0.0018,0.0021,24,17),
    (40,41,0.7283,0.8509,1.2,1),(41,42,0.3100,0.3623,0,0),(42,43,0.0410,0.0478,6,4.3),
    (43,44,0.0092,0.0116,0,0),(44,45,0.1089,0.1373,39.22,26.3),(45,46,0.0009,0.0012,39.22,26.3),
    (4,47,0.0034,0.0084,0,0),(47,48,0.0851,0.2083,79,56.4),(48,49,0.2898,0.7091,384.7,274.5),
    (49,50,0.0822,0.2011,384.7,274.5),(8,51,0.0928,0.0473,40.5,28.3),(51,52,0.3319,0.1114,3.6,2.7),
    (9,53,0.1740,0.0886,4.35,3.5),(53,54,0.2030,0.1034,26.4,19),(54,55,0.2842,0.1447,24,17.2),
    (55,56,0.2813,0.1433,0,0),(56,57,1.5900,0.5337,0,0),(57,58,0.7837,0.2630,0,0),
    (58,59,0.3042,0.1006,100,72),(59,60,0.3861,0.1172,0,0),(60,61,0.5075,0.2585,1244,888),
    (61,62,0.0974,0.0496,32,23),(62,63,0.1450,0.0738,0,0),(63,64,0.7105,0.3619,227,162),
    (64,65,1.0410,0.5302,59,42),(11,66,0.2012,0.0611,18,13),(66,67,0.0047,0.0014,18,13),
    (12,68,0.7394,0.2444,28,20),(68,69,0.0047,0.0016,28,20),
]


def build_ieee33(ext_grid_vm: float = 1.03, load_scale: float = 0.85) -> pp.pandapowerNet:
    net = pn.case33bw()
    # open tie switches -> pure radial operation
    net.line.loc[net.line.in_service & (net.line.index > 31), "in_service"] = False
    net.ext_grid.vm_pu = ext_grid_vm          # OLTC-regulated substation setpoint
    net.load.p_mw *= load_scale               # explicit nominal operating point (R3.1)
    net.load.q_mvar *= load_scale
    _attach_constraints(net, sub_trafo_mva=8.0)
    return net


def build_ieee69(ext_grid_vm: float = 1.03, load_scale: float = 0.85) -> pp.pandapowerNet:
    net = pp.create_empty_network(name="IEEE69", sn_mva=10.0)
    vn = 12.66
    buses = {i: pp.create_bus(net, vn_kv=vn, name=f"bus_{i}") for i in range(1, 70)}
    pp.create_ext_grid(net, buses[1], vm_pu=ext_grid_vm)
    for f, t, r, x, p, q in _IEEE69:
        pp.create_line_from_parameters(
            net, buses[f], buses[t], length_km=1.0,
            r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0.0, max_i_ka=0.4,
        )
        if p > 0:
            pp.create_load(net, buses[t], p_mw=p * load_scale / 1000.0,
                           q_mvar=q * load_scale / 1000.0)
    _attach_constraints(net, sub_trafo_mva=6.0)
    return net


def _attach_constraints(net: pp.pandapowerNet, sub_trafo_mva: float) -> None:
    """R3.4: operational limits used by env state/reward."""
    net.line["max_loading_percent"] = 100.0
    net["constraints"] = {
        "sub_trafo_mva": sub_trafo_mva,
        "line_loading_limit_pct": 100.0,
    }


def constraint_features(net: pp.pandapowerNet) -> tuple[float, float]:
    """Return (min line loading margin, substation loading fraction) post-PF."""
    loading = net.res_line.loading_percent.dropna()
    line_margin = float((100.0 - loading.max()) / 100.0) if len(loading) else 1.0
    s_sub = float(np.hypot(net.res_ext_grid.p_mw.iloc[0], net.res_ext_grid.q_mvar.iloc[0]))
    sub_frac = s_sub / net["constraints"]["sub_trafo_mva"]
    return line_margin, sub_frac


NETWORKS = {"ieee33": build_ieee33, "ieee69": build_ieee69}
