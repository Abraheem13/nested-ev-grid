"""Level 1: PPO DSO meta-controller.

State (8-d, R3.4-extended): [V_mean, P_tot, Q_tot, LMP, N_EV, SoC_mean,
                             line_margin, sub_loading]
Action (2-d in [0,1]^2 via Beta distributions): corridor lower bound fraction
and corridor width fraction; mapped to [price_floor, price_ceil] by the env.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta


class BetaPolicy(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 2,
                 hidden=(256, 128, 64)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.alpha = nn.Linear(d, action_dim)
        self.beta = nn.Linear(d, action_dim)

    def dist(self, s: torch.Tensor) -> Beta:
        z = self.body(s)
        a = nn.functional.softplus(self.alpha(z)) + 1.0
        b = nn.functional.softplus(self.beta(z)) + 1.0
        return Beta(a, b)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int = 8, hidden=(256, 128, 64)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: dict, state_dim: int = 8, device: str = "cpu"):
        p = cfg["training"]["ppo"]
        self.gamma, self.lam, self.clip = p["gamma"], p["gae_lambda"], p["clip"]
        self.epochs = p["epochs"]
        self.device = device
        self.pi = BetaPolicy(state_dim).to(device)
        self.v = ValueNet(state_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.pi.parameters()) + list(self.v.parameters()), lr=p["lr"])
        self.buf = []  # (s, a, logp, r, done)

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float]:
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        d = self.pi.dist(st)
        a = d.mean if deterministic else d.sample()
        return a.cpu().numpy(), float(d.log_prob(a).sum())

    def store(self, s, a, logp, r, done):
        self.buf.append((s, a, logp, r, done))

    def update(self) -> dict:
        if len(self.buf) < 2:
            self.buf.clear()
            return {}
        s = torch.as_tensor(np.array([b[0] for b in self.buf]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array([b[1] for b in self.buf]), dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor([b[2] for b in self.buf], dtype=torch.float32, device=self.device)
        r = np.array([b[3] for b in self.buf], dtype=np.float32)
        done = np.array([b[4] for b in self.buf], dtype=np.float32)

        with torch.no_grad():
            vals = self.v(s).cpu().numpy()
        adv = np.zeros_like(r)
        last = 0.0
        for t in reversed(range(len(r))):
            nxt = 0.0 if done[t] else (vals[t + 1] if t + 1 < len(r) else 0.0)
            delta = r[t] + self.gamma * nxt - vals[t]
            last = delta + self.gamma * self.lam * (0.0 if done[t] else last)
            adv[t] = last
        ret = adv + vals
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)

        stats = {}
        for _ in range(self.epochs):
            d = self.pi.dist(s)
            logp = d.log_prob(a.clamp(1e-4, 1 - 1e-4)).sum(-1)
            ratio = torch.exp(logp - logp_old)
            l_pi = -torch.min(ratio * adv_t,
                              ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t).mean()
            l_v = nn.functional.mse_loss(self.v(s), ret_t)
            l_ent = -d.entropy().sum(-1).mean()
            loss = l_pi + 0.5 * l_v + 0.001 * l_ent
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.opt.step()
            stats = {"l1/pi_loss": float(l_pi.detach()), "l1/v_loss": float(l_v.detach())}
        self.buf.clear()
        return stats

    def save(self, path):
        torch.save({"pi": self.pi.state_dict(), "v": self.v.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.pi.load_state_dict(ck["pi"]); self.v.load_state_dict(ck["v"])