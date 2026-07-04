"""Constrained Policy Optimization (Achiam et al., 2017) baseline (R2.4).

Trust-region update solving, per iteration:
    max_theta  g^T (theta - theta_k)
    s.t.       b^T (theta - theta_k) + Jc - d <= 0
               0.5 (theta - theta_k)^T H (theta - theta_k) <= delta
with H the Fisher information (KL Hessian), solved via conjugate gradient and
the analytic dual of the two-constraint QP, followed by backtracking line
search. Infeasible-recovery step reduces cost only, as in the original paper.

Shares the flat state/action interface and cost signal with PPO-Lagrangian.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .ppo_lagrangian import GaussianPolicy, Critic, gae
from ..baselines.flat_ddpg import MAX_EVS_FLAT


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(loss, model, retain=False, create=False):
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                retain_graph=retain, create_graph=create,
                                allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).contiguous().view(-1)
                      for g, p in zip(grads, model.parameters())])


class CPO:
    name = "cpo"

    def __init__(self, cfg: dict, n_agg: int = 5, device: str = "cpu",
                 cost_limit: float = 0.0, delta_kl: float = 0.01,
                 damping: float = 0.1, cg_iters: int = 10):
        self.s_dim = 8 + n_agg * (6 + MAX_EVS_FLAT * 3)
        self.a_dim = n_agg * (MAX_EVS_FLAT + 1)
        self.device = device
        self.pi = GaussianPolicy(self.s_dim, self.a_dim).to(device)
        self.vr = Critic(self.s_dim).to(device)
        self.vc = Critic(self.s_dim).to(device)
        self.opt_v = torch.optim.Adam(
            [*self.vr.parameters(), *self.vc.parameters()], lr=1e-3)
        self.d, self.delta, self.damping, self.cg_iters = cost_limit, delta_kl, damping, cg_iters
        self.gamma, self.lae = 0.99, 0.95
        self.traj = []

    @torch.no_grad()
    def act(self, s, deterministic=False):
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        dist = self.pi.dist(st)
        raw = dist.mean if deterministic else dist.sample()
        return torch.sigmoid(raw).cpu().numpy(), raw.cpu().numpy(), float(dist.log_prob(raw).sum())

    def store(self, s, raw, logp, r, c):
        self.traj.append((s, raw, logp, r, c))

    # ---------------------------------------------------------------- CPO
    def _fvp(self, kl_fn, v):
        kl = kl_fn()
        g = flat_grad(kl, self.pi, retain=True, create=True)
        gv = (g * v).sum()
        hv = flat_grad(gv, self.pi, retain=True)
        return hv + self.damping * v

    def _cg(self, fvp, b):
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        rs = r @ r
        for _ in range(self.cg_iters):
            Ap = fvp(p)
            alpha = rs / (p @ Ap + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r @ r
            if rs_new < 1e-10:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return x

    def update(self) -> dict:
        if not self.traj:
            return {}
        s = torch.as_tensor(np.array([t[0] for t in self.traj]), dtype=torch.float32, device=self.device)
        raw = torch.as_tensor(np.array([t[1] for t in self.traj]), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor([t[2] for t in self.traj], dtype=torch.float32, device=self.device)
        r = np.array([t[3] for t in self.traj], np.float32)
        c = np.array([t[4] for t in self.traj], np.float32)
        Jc = float(c.sum())

        with torch.no_grad():
            adv_r, ret_r = gae(r, self.vr(s).cpu().numpy(), self.gamma, self.lae)
            adv_c, ret_c = gae(c, self.vc(s).cpu().numpy(), self.gamma, self.lae)
        adv_r = torch.as_tensor((adv_r - adv_r.mean()) / (adv_r.std() + 1e-8), device=self.device)
        adv_c = torch.as_tensor((adv_c - adv_c.mean()) / (adv_c.std() + 1e-8), device=self.device)

        with torch.no_grad():
            dist_old = self.pi.dist(s)
            mu_old, std_old = dist_old.mean.detach(), dist_old.stddev.detach()

        def kl_fn():
            dist = self.pi.dist(s)
            return torch.distributions.kl_divergence(
                torch.distributions.Normal(mu_old, std_old), dist).sum(-1).mean()

        def surrogates():
            dist = self.pi.dist(s)
            ratio = torch.exp(dist.log_prob(raw).sum(-1) - logp_old)
            return (ratio * adv_r).mean(), (ratio * adv_c).mean()

        sr, sc = surrogates()
        sr_val, sc_val = float(sr.detach()), float(sc.detach())
        g = flat_grad(sr, self.pi, retain=True)
        b = flat_grad(sc, self.pi)

        Hinv_g = self._cg(lambda v: self._fvp(kl_fn, v), g)
        q = float(g @ Hinv_g)
        ec = Jc - self.d  # constraint slack (>0 means violated)

        if b.norm() < 1e-8 and ec <= 0:
            step = torch.sqrt(torch.tensor(2 * self.delta / (q + 1e-10))) * Hinv_g
        else:
            Hinv_b = self._cg(lambda v: self._fvp(kl_fn, v), b)
            rzb = float(g @ Hinv_b)
            sbb = float(b @ Hinv_b)
            A = q - rzb ** 2 / (sbb + 1e-10)
            B = 2 * self.delta - ec ** 2 / (sbb + 1e-10)
            if ec > 0 and B < 0:
                # infeasible: pure cost-reduction recovery step
                step = -torch.sqrt(torch.tensor(2 * self.delta / (sbb + 1e-10))) * Hinv_b
            else:
                lam = torch.sqrt(torch.tensor(max(A, 1e-10) / max(B, 1e-10)))
                nu = max(0.0, (lam.item() * ec - rzb) / (sbb + 1e-10))
                step = (Hinv_g - nu * Hinv_b) / (lam.item() + 1e-10)

        # backtracking line search
        old = flat_params(self.pi)
        improved = False
        for frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            set_flat_params(self.pi, old + frac * step)
            with torch.no_grad():
                sr2, sc2 = surrogates()
                kl = kl_fn()
            cost_ok = (float(sc2) <= sc_val + 1e-6) if ec > 0 else \
                      (ec + float(sc2) - sc_val <= max(0.0, ec))
            if kl <= self.delta * 1.5 and float(sr2) >= sr_val - 1e-6 and cost_ok:
                improved = True
                break
        if not improved:
            set_flat_params(self.pi, old)

        # value fits
        ret_r_t = torch.as_tensor(ret_r, device=self.device)
        ret_c_t = torch.as_tensor(ret_c, device=self.device)
        for _ in range(20):
            lv = (nn.functional.mse_loss(self.vr(s), ret_r_t)
                  + nn.functional.mse_loss(self.vc(s), ret_c_t))
            self.opt_v.zero_grad(); lv.backward(); self.opt_v.step()
        self.traj.clear()
        return {"Jc": Jc, "stepped": improved}

    def save(self, path):
        torch.save({"pi": self.pi.state_dict(), "vr": self.vr.state_dict(),
                    "vc": self.vc.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.pi.load_state_dict(ck["pi"]); self.vr.load_state_dict(ck["vr"])
        self.vc.load_state_dict(ck["vc"])