"""Constrained Policy Optimization (Achiam et al., 2017) baseline.

Per update solves the trust-region problem
    max_theta g^T dtheta  s.t.  c + b^T dtheta <= 0,  0.5 dtheta^T H dtheta <= delta
with H the Fisher information (conjugate gradient on Fisher-vector products),
the analytic dual of the two-constraint QP, a cost-only recovery step when the
problem is infeasible, and a backtracking line search. c = J_c - d is the
constraint violation of the last episode; cost advantages are centred but not
rescaled so that b and c share units.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ppo_lagrangian import OnPolicyBase, gae


def _flat(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def _grad(loss, model, retain=False, create=False):
    g = torch.autograd.grad(loss, list(model.parameters()), retain_graph=retain,
                            create_graph=create, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                      for x, p in zip(g, model.parameters())])


class CPO(OnPolicyBase):
    def __init__(self, cfg: dict, s_dim: int, a_dim: int, cost_limit: float = 0.0,
                 delta_kl: float = 0.01, damping: float = 0.1, cg_iters: int = 10):
        super().__init__(cfg, s_dim, a_dim)
        self.opt_v = torch.optim.Adam([*self.vr.parameters(), *self.vc.parameters()], lr=1e-3)
        self.d, self.delta, self.damping, self.cg_iters = cost_limit, delta_kl, damping, cg_iters

    def _fvp(self, kl_fn, v):
        g = _grad(kl_fn(), self.pi, retain=True, create=True)
        return _grad((g * v).sum(), self.pi, retain=True) + self.damping * v

    def _cg(self, fvp, b):
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        rs = r @ r
        for _ in range(self.cg_iters):
            ap = fvp(p)
            alpha = rs / (p @ ap + 1e-10)
            x += alpha * p
            r -= alpha * ap
            rs_new = r @ r
            if rs_new < 1e-10:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    def update(self) -> None:
        if not self.traj:
            return
        s, raw, logp_old, r, c = self._batch()
        with torch.no_grad():
            adv_r, ret_r = gae(r, self.vr(s).numpy(), self.gamma, self.lam)
            adv_c, ret_c = gae(c, self.vc(s).numpy(), self.gamma, self.lam)
        adv_r = torch.as_tensor((adv_r - adv_r.mean()) / (adv_r.std() + 1e-8))
        adv_c = torch.as_tensor(adv_c - adv_c.mean())
        ec = float(c.sum()) - self.d
        with torch.no_grad():
            d0 = self.pi.dist(s)
            mu0, sd0 = d0.mean.clone(), d0.stddev.clone()

        def kl_fn():
            return torch.distributions.kl_divergence(
                torch.distributions.Normal(mu0, sd0), self.pi.dist(s)).sum(-1).mean()

        def surrogates():
            ratio = torch.exp(self.pi.dist(s).log_prob(raw).sum(-1) - logp_old)
            return (ratio * adv_r).mean(), (ratio * adv_c).mean()

        sr, sc = surrogates()
        sr0, sc0 = float(sr), float(sc)
        g = _grad(sr, self.pi, retain=True)
        b = _grad(sc, self.pi)
        fvp = lambda v: self._fvp(kl_fn, v)  # noqa: E731
        hg = self._cg(fvp, g)
        q = float(g @ hg)
        if float(b.norm()) < 1e-8 and ec <= 0:
            step = torch.sqrt(torch.tensor(2 * self.delta / (q + 1e-10))) * hg
        else:
            hb = self._cg(fvp, b)
            rr, ss = float(g @ hb), float(b @ hb)
            A = q - rr ** 2 / (ss + 1e-10)
            B = 2 * self.delta - ec ** 2 / (ss + 1e-10)
            if ec > 0 and B < 0:
                step = -torch.sqrt(torch.tensor(2 * self.delta / (ss + 1e-10))) * hb
            else:
                lam = float(np.sqrt(max(A, 1e-10) / max(B, 1e-10)))
                nu = max(0.0, (lam * ec - rr) / (ss + 1e-10))
                step = (hg - nu * hb) / (lam + 1e-10)
        old = _flat(self.pi)
        for frac in (1.0, 0.5, 0.25, 0.125, 0.0625):
            _set_flat(self.pi, old + frac * step)
            with torch.no_grad():
                sr2, sc2 = surrogates()
                kl = float(kl_fn())
            cost_ok = float(sc2) <= sc0 + 1e-6 if ec > 0 else ec + float(sc2) - sc0 <= max(0.0, ec)
            if kl <= 1.5 * self.delta and float(sr2) >= sr0 - 1e-6 and cost_ok:
                break
        else:
            _set_flat(self.pi, old)
        ret_r, ret_c = torch.as_tensor(ret_r), torch.as_tensor(ret_c)
        for _ in range(20):
            lv = nn.functional.mse_loss(self.vr(s), ret_r) + nn.functional.mse_loss(self.vc(s), ret_c)
            self.opt_v.zero_grad()
            lv.backward()
            self.opt_v.step()
        self.traj.clear()
