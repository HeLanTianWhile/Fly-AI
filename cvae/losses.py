"""losses.py — 自研损失函数（纯 NumPy + 自定义张量引擎）。"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class GaussianVAELoss:
    """用于 CVAE 的训练目标 = 重构似然（负高斯对数似然）+ KL 散度。

    前向：
      recon_x : Tensor (N, 1, H, W)         解码器输出（像素均值）
      x       : Tensor (N, 1, H, W)         真实图像（[0,1]）
      mu      : Tensor (N, latent)          编码器输出的均值
      logvar  : Tensor (N, latent)          编码器输出的对数方差
      sigma   : 像素噪声标准差（重构项尺度）
      beta    : KL 权重（beta-VAE 风格，可调密程度）

    伪似然采用固定噪声 sigma 的高斯：
      recon  = -log p(x|z) = ||x - recon_x||^2 / (2*sigma^2) + const
    KL（标准正态先验）：
      lse     = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    """
    def __init__(self, sigma=0.3, beta=1.0):
        self.sigma = sigma
        self.beta = beta

    def __call__(self, recon_x: Tensor, x: Tensor, mu: Tensor,
                 logvar: Tensor):
        recon = ((recon_x - x) ** 2) / (2.0 * self.sigma ** 2)
        recon = recon.mean()
        kl = 0.5 * (logvar.exp() + mu ** 2 - 1.0 - logvar).sum()
        kl_d = kl / float(logvar.data.shape[0])  # 按 batch 平均
        loss = recon + self.beta * kl_d
        return loss, recon, kl_d


class MSELoss:
    def __call__(self, pred: Tensor, target: Tensor):
        return ((pred - target) ** 2).mean()
