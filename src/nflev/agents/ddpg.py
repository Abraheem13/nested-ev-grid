"""DDPG (Lillicrap et al., 2016) with actions in [0, 1]^d.

The replay buffer stores the action that was actually *applied* by the
environment (after the feasibility layer), so the critic learns the value of
executed actions. Transitions are (s_t, a_t, r_t, s_{t+1}, done): the reward is
the one produced by a_t (the trainer stores a transition only once s_{t+1} is
known).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(dims, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


HIDDEN = (256, 128, 64)


class ReplayBuffer:
    def __init__(self, s_dim: int, a_dim: int, size: int, rng: np.random.Generator):
        self.s = np.zeros((size, s_dim), np.float32)
        self.a = np.zeros((size, a_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, s_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.n, self.i, self.rng = size, 0, 0, rng

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, b):
        j = self.rng.integers(0, self.n, b)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


class DDPGAgent:
    def __init__(self, cfg: dict, s_dim: int, a_dim: int, seed: int, zero_init: bool = False):
        p = cfg["training"]["ddpg"]
        self.tau, self.batch, self.gamma = p["tau"], p["batch"], p["gamma"]
        self.warmup = p["warmup"]
        self.a_dim = a_dim
        self.rng = np.random.default_rng(seed)
        self.actor = mlp([s_dim, *HIDDEN, a_dim], nn.Sigmoid())
        self.critic = mlp([s_dim + a_dim, *HIDDEN, 1])
        self.actor_t = mlp([s_dim, *HIDDEN, a_dim], nn.Sigmoid())
        self.critic_t = mlp([s_dim + a_dim, *HIDDEN, 1])
        if zero_init:          # output layer ~0 -> sigmoid ~0.5, i.e. zero residual at the start
            last = self.actor[-2]
            nn.init.uniform_(last.weight, -1e-3, 1e-3)
            nn.init.zeros_(last.bias)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=p["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=p["lr_critic"])
        self.buf = ReplayBuffer(s_dim, a_dim, p["buffer"], self.rng)
        self.noise = p["noise_start"]

    @torch.no_grad()
    def act(self, s: np.ndarray, explore: bool) -> np.ndarray:
        if explore and self.buf.n < self.warmup:
            return self.rng.uniform(0.0, 1.0, self.a_dim).astype(np.float32)
        a = self.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
        if explore:
            a = np.clip(a + self.rng.normal(0.0, self.noise, self.a_dim), 0.0, 1.0)
        return a.astype(np.float32)

    def store(self, s, a, r, s2, done):
        self.buf.add(s, a, r, s2, float(done))

    def update(self) -> None:
        if self.buf.n < max(self.batch, self.warmup):
            return
        s, a, r, s2, d = (torch.as_tensor(x) for x in self.buf.sample(self.batch))
        with torch.no_grad():
            y = r + self.gamma * (1.0 - d) * self.critic_t(torch.cat([s2, self.actor_t(s2)], -1)).squeeze(-1)
        q = self.critic(torch.cat([s, a], -1)).squeeze(-1)
        loss_c = nn.functional.mse_loss(q, y)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        loss_a = -self.critic(torch.cat([s, self.actor(s)], -1)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        with torch.no_grad():
            for tp, sp in ((self.actor_t, self.actor), (self.critic_t, self.critic)):
                for pt, ps in zip(tp.parameters(), sp.parameters()):
                    pt.mul_(1.0 - self.tau).add_(self.tau * ps)

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.actor_t.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_t.load_state_dict(sd["critic"])
