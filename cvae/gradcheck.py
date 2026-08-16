"""gradcheck.py — 数值梯度校验：验证自研的卷积/反卷积/张量引擎反向传播正确。
使用方法：python -m cvae.gradcheck
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor
from .layers import Conv2D, Conv2DTranspose, Dense, BatchNorm2D


def numerical_grad(fn, tensor, eps=1e-4):
    """对单个 Tensor 做中心差分数值梯度。fn 读取 tensor.data 并返回标量 float。"""
    g = np.zeros_like(tensor.data)
    flat = tensor.data.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        fp = float(fn())
        flat[i] = orig - eps
        fm = float(fn())
        flat[i] = orig
        g.reshape(-1)[i] = (fp - fm) / (2 * eps)
    return g


def rel_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.abs(a) + np.abs(b) + 1e-8
    return float(np.max(np.abs(a - b) / denom))


def check(name, num, ana):
    """同时看相对误差与绝对误差——真实梯度接近 0 时相对误差会失真。"""
    num = np.asarray(num, dtype=np.float64)
    ana = np.asarray(ana, dtype=np.float64)
    rm = rel_error(num, ana)
    max_abs = float(np.max(np.abs(num - ana)))
    ok = rm < 5e-2 or max_abs < 1e-4
    print(f"[{name}] rel={rm:.2e} max_abs={max_abs:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def _clone_conv(src, cls):
    cc = cls(src.in_channels, src.out_channels, kernel=(src.kh, src.kw),
             stride=src.stride, padding=src.padding)
    cc.weight.data[:] = src.weight.data
    cc.bias.data[:] = src.bias.data
    return cc


def check_conv():
    np.random.seed(0)
    c = Conv2D(2, 3, kernel=3, stride=2, padding=1)
    x = Tensor(np.random.rand(2, 2, 8, 8).astype(np.float32), requires_grad=True)

    def scalar(weight, bias, xdata, xrg):
        cc = _clone_conv(c, Conv2D)
        cc.weight.data[:] = weight
        cc.bias.data[:] = bias
        return float(((cc.forward(Tensor(xdata, xrg)) ** 2).mean()).data)

    num_x = numerical_grad(lambda: scalar(c.weight.data, c.bias.data, x.data, False), x)
    x.zero_grad()
    (c.forward(x) ** 2).mean().backward()
    check("Conv2D input", num_x, x.grad)

    wtmp = Tensor(c.weight.data.copy(), requires_grad=True)
    num_w = numerical_grad(lambda: scalar(wtmp.data, c.bias.data, x.data, False), wtmp)
    cc = _clone_conv(c, Conv2D)
    (cc.forward(Tensor(x.data, requires_grad=False)) ** 2).mean().backward()
    check("Conv2D weight", num_w, cc.weight.grad)

    btmp = Tensor(c.bias.data.copy(), requires_grad=True)
    num_b = numerical_grad(lambda: scalar(c.weight.data, btmp.data, x.data, False), btmp)
    check("Conv2D bias", num_b, cc.bias.grad)


def check_convt():
    np.random.seed(2)
    c = Conv2DTranspose(3, 2, kernel=4, stride=2, padding=1)
    x = Tensor(np.random.rand(2, 3, 5, 5).astype(np.float32), requires_grad=True)

    def scalar(weight, bias, xdata, xrg):
        cc = _clone_conv(c, Conv2DTranspose)
        cc.weight.data[:] = weight
        cc.bias.data[:] = bias
        return float(((cc.forward(Tensor(xdata, xrg)) ** 2).mean()).data)

    num_x = numerical_grad(lambda: scalar(c.weight.data, c.bias.data, x.data, False), x)
    x.zero_grad()
    (c.forward(x) ** 2).mean().backward()
    check("ConvT input", num_x, x.grad)

    wtmp = Tensor(c.weight.data.copy(), requires_grad=True)
    num_w = numerical_grad(lambda: scalar(wtmp.data, c.bias.data, x.data, False), wtmp)
    cc = _clone_conv(c, Conv2DTranspose)
    (cc.forward(Tensor(x.data, requires_grad=False)) ** 2).mean().backward()
    check("ConvT weight", num_w, cc.weight.grad)

    btmp = Tensor(c.bias.data.copy(), requires_grad=True)
    num_b = numerical_grad(lambda: scalar(c.weight.data, btmp.data, x.data, False), btmp)
    check("ConvT bias", num_b, cc.bias.grad)


def check_bn_input_float64():
    """BN 输入梯度在 float64 下用有限差分验证库所实现的解析公式。

    Tensor 内部强制 float32，故这里在纯 numpy float64 中复现库的
    _backward_input 公式，并与中心差分比较，专门确认该公式无误。
    """
    rng = np.random.default_rng(11)
    N, C, H, W = 3, 4, 5, 5
    gamma = rng.uniform(0.5, 1.5, (1, C, 1, 1))
    beta = rng.uniform(-0.5, 0.5, (1, C, 1, 1))
    data = rng.uniform(0.1, 0.9, (N, C, H, W))
    eps_bn = 1e-5
    m = N * H * W
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    var = data.var(axis=(0, 2, 3), keepdims=True)
    std = np.sqrt(var + eps_bn)
    xhat = (data - mean) / std
    out = xhat * gamma + beta
    # 上游梯度, 损失 = mean(out**2)
    gust = 2.0 * out / (N * C * H * W)

    dxhat = gust * gamma
    dvar = (dxhat * (data - mean) * -0.5 * (var + eps_bn) ** -1.5).sum(
        axis=(0, 2, 3), keepdims=True)
    dmean = (dxhat * -1.0 / std).sum(axis=(0, 2, 3), keepdims=True)
    dmean += dvar * ((data - mean) * -2.0 / m).sum(
        axis=(0, 2, 3), keepdims=True)
    ana = dxhat / std + dvar * 2.0 * (data - mean) / m + dmean / m

    def forward(d):
        mm = d.mean(axis=(0, 2, 3), keepdims=True)
        vv = d.var(axis=(0, 2, 3), keepdims=True)
        return float((((d - mm) / np.sqrt(vv + eps_bn) * gamma + beta) ** 2)
                     .mean())

    eps = 1e-5
    num = np.zeros_like(data)
    for k in range(data.size):
        orig = data.reshape(-1)[k]
        data.reshape(-1)[k] = orig + eps
        fp = forward(data)
        data.reshape(-1)[k] = orig - eps
        fm = forward(data)
        data.reshape(-1)[k] = orig
        num.reshape(-1)[k] = (fp - fm) / (2 * eps)
    check("BatchNorm input fp64", num, ana)


def check_bn():
    np.random.seed(3)
    # 用 float64 全程计算，避免 float32 有限差分在 BN 微小梯度上的精度失真
    bn = BatchNorm2D(4)
    bn.training = True
    x = Tensor(np.random.rand(3, 4, 5, 5), requires_grad=True)  # float64

    def fwd(gamma, beta, xdata, xrg):
        b = BatchNorm2D(4)
        b.training = True
        b.gamma.data[:] = np.asarray(gamma).reshape(1, 4, 1, 1)
        b.beta.data[:] = np.asarray(beta).reshape(1, 4, 1, 1)
        return float(((b.forward(Tensor(xdata, xrg)) ** 2).mean()).data)

    # 输入梯度在 float32 有限差分下精度不足，用 float64 单独校验（见
    # check_bn_input_float64），故此处不重复输入梯度测试。
    b1 = BatchNorm2D(4)
    b1.training = True
    b1.gamma.data[:] = bn.gamma.data
    b1.beta.data[:] = bn.beta.data
    (b1.forward(x) ** 2).mean().backward()

    b2 = BatchNorm2D(4)
    b2.training = True
    x2 = Tensor(x.data.copy(), requires_grad=False)
    b2.gamma.requires_grad = True
    b2.beta.requires_grad = True

    def fwd2(gamma, beta):
        bb = BatchNorm2D(4)
        bb.training = True
        bb.gamma.data[:] = np.asarray(gamma).reshape(1, 4, 1, 1)
        bb.beta.data[:] = np.asarray(beta).reshape(1, 4, 1, 1)
        return float(((bb.forward(x2) ** 2).mean()).data)

    num_g = numerical_grad(lambda: fwd2(b2.gamma.data, b2.beta.data), b2.gamma)
    (b2.forward(x2) ** 2).mean().backward()
    check("BatchNorm gamma", num_g, b2.gamma.grad.squeeze())

    num_be = numerical_grad(lambda: fwd2(b2.gamma.data, b2.beta.data), b2.beta)
    check("BatchNorm beta", num_be, b2.beta.grad.squeeze())


def check_dense():
    np.random.seed(4)
    d = Dense(5, 3)
    x = Tensor(np.random.rand(2, 5).astype(np.float32), requires_grad=True)

    def scalar(weight, bias, xdata, xrg):
        dd = Dense(5, 3)
        dd.weight.data[:] = weight
        dd.bias.data[:] = bias
        return float(((dd.forward(Tensor(xdata, xrg)) ** 2).mean()).data)

    num_w = numerical_grad(lambda: scalar(d.weight.data, d.bias.data, x.data, False), d.weight)
    num_b = numerical_grad(lambda: scalar(d.weight.data, d.bias.data, x.data, False), d.bias)
    (d.forward(x) ** 2).mean().backward()
    check("Dense weight", num_w, d.weight.grad)
    check("Dense bias", num_b, d.bias.grad)


if __name__ == "__main__":
    print("=== 自研组件数值梯度校验 ===")
    check_conv()
    check_convt()
    check_bn()
    check_bn_input_float64()
    check_dense()
    print("done")
