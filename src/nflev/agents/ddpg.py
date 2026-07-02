"""Level 2: DDPG aggregator agents.

Actor output (max_evs + 1 dims, sigmoid in [0,1]):
  [:max_evs] per-EV charging rate fractions -> scaled to [0, c_max], then
             projected onto {sum <= P_cap} by the environment (R1.5)
  [-1]       execution-price fraction within the L1 corridor (R1.3)
"""
from __future__ import annotations
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn


def mlp(dims, out_act=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class DDPGAgent:
    def __init__(self, cfg: dict, state_dim: int, max_evs: int = 40,
                 device: str = "cpu"):
        p = cfg["training"]["ddpg"]
        self.tau, self.batch = p["tau"], p["batch"]
        self.gamma = cfg["training"]["ppo"]["gamma"]
        self.device = device
        self.max_evs = max_evs
        self.act_dim = max_evs + 1

        self.actor = mlp([state_dim, 256, 128, 64, self.act_dim], nn.Sigmoid()).to(device)
        self.critic = mlp([state_dim + self.act_dim, 256, 128, 64, 1]).to(device)
        self.actor_t = mlp([state_dim, 256, 128, 64, self.act_dim], nn.Sigmoid()).to(device)
        self.critic_t = mlp([state_dim + self.act_dim, 256, 128, 64, 1]).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=p["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=p["lr_critic"])
        self.buffer = deque(maxlen=p["buffer"])
        self.noise = 0.2

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = self.actor(st).cpu().numpy()
        if not deterministic:
            a = np.clip(a + np.random.normal(0, self.noise, a.shape), 0, 1)
        return a

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def update(self) -> dict:
        if len(self.buffer) < self.batch:
            return {}
        batch = random.sample(self.buffer, self.batch)
        s = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.array([b[4] for b in batch]), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q2 = self.critic_t(torch.cat([s2, self.actor_t(s2)], -1)).squeeze(-1)
            y = r + self.gamma * (1 - d) * q2
        q = self.critic(torch.cat([s, a], -1)).squeeze(-1)
        l_c = nn.functional.mse_loss(q, y)
        self.opt_c.zero_grad(); l_c.backward(); self.opt_c.step()

        l_a = -self.critic(torch.cat([s, self.actor(s)], -1)).mean()
        self.opt_a.zero_grad(); l_a.backward(); self.opt_a.step()

        with torch.no_grad():
            for tp, p_ in zip(self.actor_t.parameters(), self.actor.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p_)
            for tp, p_ in zip(self.critic_t.parameters(), self.critic.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p_)
        return {"l2/critic_loss": float(l_c.detach()), "l2/actor_loss": float(l_a.detach())}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.actor_t.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_t.load_state_dict(ck["critic"])