"""optimizer.py — 自研 Adam 优化器（纯 NumPy）。"""
from __future__ import annotations

import numpy as np


class Adam:
    """自适应矩估计优化器（Kingma & Ba, Adam）。

    初始化参数会一次性建立一阶/二阶动量缓存。
    """
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        self.t += 1
        for idx, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad.copy()
            if self.wd > 0:
                g = g + self.wd * p.data
            self.m[idx] = self.b1 * self.m[idx] + (1 - self.b1) * g
            self.v[idx] = self.b2 * self.v[idx] + (1 - self.b2) * (g * g)
            mhat = self.m[idx] / (1 - self.b1 ** self.t)
            vhat = self.v[idx] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
