"""model.py — 自研条件变分自编码器 (CVAE)，用于"输入数字 -> 生成 100x100 手写数字图像"。

全新架构要点（不使用任何 AI 封装库，全部基于项目自制张量引擎）：
  1. CondEmbed：数字标签 -> 可学习嵌入，同时喂给编码器与解码器。
  2. 渐进式条件调制（Progressive Conditioning）：解码器在每个上采样阶段
     再注入标签嵌入，让"数字类别"对整张图都有区分度而非只在 latent 注入一次。
  3. 轻量卷积 + 批归一化 + 4 级下/上采样，latent 为 48 维，CPU 友好。

形状链路（100x100 输入）：
  encoder: 100->50->25->13->7  (通道 16,32,64,64)
  decoder: 7->13->25->50->100  (转置卷积 + 条件调制)

损失见 losses.GaussianVAELoss；优化见 optimizer.Adam。
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor
from .layers import Dense, Conv2D, Conv2DTranspose, BatchNorm2D
from .layers import flatten, reshape4d


# 把 [0,1] 数据缩放到 [-1,1] 以利训练
def to_pm1(x: Tensor) -> Tensor:
    return x * 2.0 - 1.0


def from_pm1(y: Tensor) -> Tensor:
    return (y + 1.0) * 0.5


class CondEmbed:
    """数字标签的可学习嵌入层。label: int/long -> (N, embed_dim)"""
    def __init__(self, num_classes, embed_dim):
        self.embed = Tensor(
            (np.random.randn(num_classes, embed_dim) * 0.1).astype(np.float32),
            requires_grad=True)

    def forward(self, label):
        return self.embed[np.asarray(label)]

    def __call__(self, label):
        return self.forward(label)

    def parameters(self):
        return [self.embed]


class CondModulate:
    """把 (N, cond_dim) 条件经线性投影后广播调制特征图：x*(1+cond*wa)+cond*wb"""
    def __init__(self, channels, cond_dim):
        self.proj = Dense(cond_dim, channels)
        self.wa = Tensor(
            (np.random.randn(1, channels, 1, 1) * 0.05).astype(np.float32),
            requires_grad=True)
        self.wb = Tensor(
            (np.random.randn(1, channels, 1, 1) * 0.05).astype(np.float32),
            requires_grad=True)

    def forward(self, x: Tensor, cond: Tensor):
        c = self.proj(cond).silu()
        cond_r = c.reshape(c.shape[0], -1, 1, 1)
        return x * (1.0 + cond_r * self.wa) + cond_r * self.wb

    def __call__(self, x, cond):
        return self.forward(x, cond)

    def parameters(self):
        return self.proj.parameters() + [self.wa, self.wb]


class CVAE:
    def __init__(self, num_classes=10, latent_dim=48, cond_dim=16, base=16):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.base = base
        c2, c3, c4 = base, base * 2, base * 4

        # ---- 编码器 ----
        self.enc_conv1 = Conv2D(1, c2, 3, stride=2, padding=1)
        self.enc_bn1 = BatchNorm2D(c2)
        self.enc_conv2 = Conv2D(c2, c3, 3, stride=2, padding=1)
        self.enc_bn2 = BatchNorm2D(c3)
        self.enc_conv3 = Conv2D(c3, c4, 3, stride=2, padding=1)
        self.enc_bn3 = BatchNorm2D(c4)
        self.enc_conv4 = Conv2D(c4, c4, 3, stride=2, padding=1)
        self.enc_bn4 = BatchNorm2D(c4)
        self.emb = CondEmbed(num_classes, cond_dim)
        in_flat = c4 * 7 * 7 + cond_dim
        self.enc_fc = Dense(in_flat, latent_dim * 2)

        # ---- 解码器 ----
        self.dec_fc = Dense(latent_dim + cond_dim, c4 * 7 * 7)
        self.dec_bn0 = BatchNorm2D(c4)

        def ct(ci, co, op=0):
            return Conv2DTranspose(ci, co, 3, stride=2, padding=1,
                                   output_padding=op)

        self.dec_mod1 = CondModulate(c4, cond_dim)
        self.dec_conv1 = ct(c4, c3)            # 7 -> 13
        self.dec_bn1 = BatchNorm2D(c3)
        self.dec_mod2 = CondModulate(c3, cond_dim)
        self.dec_conv2 = ct(c3, base)          # 13 -> 25
        self.dec_bn2 = BatchNorm2D(base)
        self.dec_mod3 = CondModulate(base, cond_dim)
        self.dec_conv3 = ct(base, base // 2, op=1)  # 25 -> 50
        self.dec_bn3 = BatchNorm2D(base // 2)
        self.dec_mod4 = CondModulate(base // 2, cond_dim)
        self.dec_conv4 = ct(base // 2, 1, op=1)     # 50 -> 100

        self.modules = [
            self.enc_conv1, self.enc_bn1, self.enc_conv2, self.enc_bn2,
            self.enc_conv3, self.enc_bn3, self.enc_conv4, self.enc_bn4,
            self.emb, self.enc_fc, self.dec_fc, self.dec_bn0,
            self.dec_mod1, self.dec_conv1, self.dec_bn1, self.dec_mod2,
            self.dec_conv2, self.dec_bn2, self.dec_mod3, self.dec_conv3,
            self.dec_bn3, self.dec_mod4, self.dec_conv4,
        ]

    def train_mode(self, flag=True):
        for m in self.modules:
            if hasattr(m, "training"):
                m.training = flag

    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params

    # ------------------------------------------------------------------
    def encode(self, x: Tensor, label):
        h = to_pm1(x)
        h = self.enc_bn1(self.enc_conv1(h)).silu()
        h = self.enc_bn2(self.enc_conv2(h)).silu()
        h = self.enc_bn3(self.enc_conv3(h)).silu()
        h = self.enc_bn4(self.enc_conv4(h)).silu()
        f = flatten(h)
        emb = self.emb.forward(label)
        z_in = Tensor.concat([f, emb], axis=1)
        p = self.enc_fc(z_in)
        mu = p[:, :self.latent_dim]
        logvar = p[:, self.latent_dim:]
        return mu, logvar

    @staticmethod
    def sample(mu: Tensor, logvar: Tensor, rng):
        eps = Tensor((rng.standard_normal(mu.data.shape)).astype(np.float32))
        return mu + eps * ((logvar * 0.5).exp())

    def decode(self, z: Tensor, label):
        emb = self.emb.forward(label)
        zc = Tensor.concat([z, emb], axis=1)
        h = self.dec_fc(zc).silu()
        N = z.data.shape[0]
        h = reshape4d(h, N, self.base * 4, 7, 7)
        h = self.dec_bn0(h).silu()
        h = self.dec_mod1(h, emb).silu()
        h = self.dec_bn1(self.dec_conv1(h)).silu()
        h = self.dec_mod2(h, emb).silu()
        h = self.dec_bn2(self.dec_conv2(h)).silu()
        h = self.dec_mod3(h, emb).silu()
        h = self.dec_bn3(self.dec_conv3(h)).silu()
        h = self.dec_mod4(h, emb).silu()
        h = self.dec_conv4(h)
        return from_pm1(h)

    def forward(self, x: Tensor, label, rng):
        mu, logvar = self.encode(x, label)
        z = self.sample(mu, logvar, rng)
        recon = self.decode(z, label)
        return recon, mu, logvar

    # 生成接口（推理）：给定数字标签，返回 (1,1,100,100) 图像张量（数据 [0,1]）
    def generate(self, label, rng, z_scale=0.6):
        z = Tensor((rng.standard_normal((1, self.latent_dim)) *
                    z_scale).astype(np.float32))
        return self.decode(z, label)
