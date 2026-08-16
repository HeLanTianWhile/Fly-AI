"""layers.py — 自研深度学习组件（全部手写反向传播，纯 NumPy）。

只依赖项目自身的 `tensor.Tensor` 自动微分，不引入任何 AI 封装库。
包含：
  - Dense               全连接层
  - Conv2D              二维卷积
  - Conv2DTranspose     转置卷积（上采样）
  - BatchNorm2D         批归一化（含条件开关）
  - Flatten / reshapes  用于桥接连结
  - initialize_*        权重初始化工具
"""
from __future__ import annotations

import math
import numpy as np

from .tensor import Tensor


# ----------------------------------------------------------------------
# 权重初始化
# ----------------------------------------------------------------------
def _he_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


def xavier_init(fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return (np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            ).astype(np.float32)


# ----------------------------------------------------------------------
# Dense 全连接
# ----------------------------------------------------------------------
class Dense:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(_he_init(in_features, out_features),
                             requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros((1, out_features), dtype=np.float32),
                               requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


# 为 Tensor 增加矩阵乘的临时实现（放在模块级，通过 mixin 方式附加）
def tensor_matmul(self, other):
    has_tensor_other = isinstance(other, Tensor)
    other_data = other.data if has_tensor_other else np.asarray(
        other, dtype=np.float32)
    out = Tensor(self.data @ other_data,
                 requires_grad=self.requires_grad or
                 (has_tensor_other and other.requires_grad))
    if has_tensor_other:
        parents = []
        if self.requires_grad:
            parents.append((self, lambda g: g @ other_data.T))
        if other.requires_grad:
            parents.append((other, lambda g: self.data.T @ g))
        out._parents = tuple(parents)
    else:
        out._parents = ((self, lambda g: g @ other_data.T),)
    return out


Tensor.__matmul__ = tensor_matmul


# ----------------------------------------------------------------------
# 2D 卷积（多输入通道 -> 多输出通道）。支撑任意核、padding、stride。
# 用 im2col 思想实现：把每个输出位置对应的输入窗口取出来做矩阵乘。
# ----------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.kh, self.kw = kernel
        self.stride = stride if isinstance(stride, (tuple, list)) else (stride,
                                                                        stride)
        self.padding = padding if isinstance(padding, (tuple, list)) else (
            padding, padding)
        ph, pw = self.padding
        limit = math.sqrt(1.0 / (in_channels * self.kh * self.kw))
        self.weight = Tensor(
            np.random.uniform(-limit, limit,
                              size=(out_channels, in_channels,
                                    self.kh, self.kw)).astype(np.float32),
            requires_grad=True)
        self.bias = Tensor(np.zeros((out_channels,), dtype=np.float32),
                           requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, C, H, W) -> (N, C_out, H_out, W_out)"""
        data = x.data
        N, C, H, W = data.shape
        in_H, in_W = H, W
        ph, pw = self.padding
        sh, sw = self.stride
        H_out = (H + 2 * ph - self.kh) // sh + 1
        W_out = (W + 2 * pw - self.kw) // sw + 1
        pads = ((0, 0), (0, 0), (ph, ph), (pw, pw))
        padded = np.pad(data, pads, mode="constant")
        # 构造 im2col: (N, H_out, W_out, C*kh*kw)
        kh, kw = self.kh, self.kw
        # 分步填充，收集每个输出位置对应的输入窗口
        view = np.zeros((N, H_out, W_out, C, kh, kw), dtype=np.float32)
        for i in range(kh):
            rows = np.arange(i, i + sh * H_out, sh)
            for j in range(kw):
                cols_idx = np.arange(j, j + sw * W_out, sw)
                view[:, :, :, :, i, j] = padded[:, :, rows][:, :, :,
                                                             cols_idx].transpose(
                    0, 2, 3, 1)
        cols = view.reshape(N, H_out, W_out, C * kh * kw)
        # 权重矩阵行序 = (cin*kh*kw + i*kw + j)，与 cols 的列序一致
        # weight: (Cout, Cin, kh, kw) -> (Cin*kh*kw, Cout)
        Wmat = self.weight.data.transpose(1, 2, 3, 0).reshape(C * kh * kw,
                                                              self.out_channels)
        out_data = np.einsum("nhwc,co->nhwo", cols, Wmat) + self.bias.data
        out_data = np.transpose(out_data, (0, 3, 1, 2))
        out = Tensor(out_data, requires_grad=self.requires_grad(x))

        # 反向：对 weight / bias 求梯度
        self._store_cols = cols
        self._store_shape = (N, C, in_H, in_W)
        parents = []
        if x.requires_grad:
            parents.append(
                (x, lambda g: self._backward_input(g, pads)))
        parents.append(
            (self.weight, lambda g: self._backward_weight(g)))
        parents.append(
            (self.bias, lambda g: np.einsum(
                "nhwo->o", np.transpose(g, (0, 2, 3, 1))).astype(np.float32)))
        self._active_bias = self.bias
        out._parents = tuple(parents)
        return out

    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def requires_grad(x):
        return x.requires_grad or True  # weight/bias 总是要梯度

    def _backward_input(self, gout, pads):
        # gout: (N, Co, H_out, W_out)
        sh, sw = self.stride
        ph, pw = self.padding
        kh, kw = self.kh, self.kw
        gin_shape = self._store_shape  # (N, C, H, W)
        N, C, H, Wd = gin_shape
        H_out, W_out = gout.shape[2], gout.shape[3]
        # 权重 (Cout, Cin, kh, kw) -> (Cout, Cin*kh*kw)，使列序 = cin*kh*kw+i*kw+j
        W = self.weight.data.transpose(1, 2, 3, 0).reshape(
            self.in_channels * kh * kw, self.out_channels).T
        grad_cols = np.transpose(gout, (0, 2, 3, 1))  # (N,Ho,Wo,Co)
        gcols = np.einsum("nhwo,oc->nhwc", grad_cols, W)  # (N,Ho,Wo,C*k*k)
        grad_padded = np.zeros((N, self.in_channels, H + 2 * ph,
                                Wd + 2 * pw), dtype=np.float32)
        for c in range(self.in_channels):
            for i in range(kh):
                rows = np.arange(i, i + sh * H_out, sh)
                for j in range(kw):
                    cidx = np.arange(j, j + sw * W_out, sw)
                    patch = gcols[:, :, :, c * kh * kw + i * kw + j]
                    np.add.at(grad_padded, (slice(None), c,
                                            rows[:, None], cidx[None, :]),
                              patch)
        if ph == 0 and pw == 0:
            return grad_padded
        sl = (slice(None), slice(None),
              slice(ph, ph + H), slice(pw, pw + Wd))
        grad_in = grad_padded[sl]
        return grad_in

    def _backward_weight(self, gout):
        cols = self._store_cols  # (N,Ho,Wo,C*k*k)
        gout_t = np.transpose(gout, (0, 2, 3, 1))  # (N,Ho,Wo,Co)
        grad = np.einsum("nhwc,nhwo->co", cols, gout_t)  # (C*k*k, Co)
        # grad 行序 = cin*kh*kw+i*kw+j，重排回 (Cout, Cin, kh, kw)
        return grad.reshape(self.in_channels, self.kh, self.kw,
                            self.out_channels).transpose(3, 0, 1, 2)


# ----------------------------------------------------------------------
# Conv2DTranspose 转置卷积（上采样）。用于解码器逐级放大到 100x100。
# 实现方式（scatter 视图）：
#   对每个输入像素，把 kernel 乘其值后，放到输出 (h*stride, w*stride) 处累加，
#   得到满尺寸输出，再用 pad 裁剪（并叠加 output_padding）。可精确得到目标尺寸。
# ----------------------------------------------------------------------
class Conv2DTranspose:
    def __init__(self, in_channels, out_channels, kernel, stride=1,
                 padding=0, output_padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.kh, self.kw = kernel
        self.stride = stride if isinstance(stride, (tuple, list)) else (stride,
                                                                        stride)
        self.padding = padding if isinstance(padding, (tuple, list)) else (
            padding, padding)
        self.output_padding = (output_padding if isinstance(
            output_padding, (tuple, list)) else (output_padding,
                                                 output_padding))
        ph, pw = self.padding
        limit = math.sqrt(1.0 / (in_channels * self.kh * self.kw))
        self.weight = Tensor(
            np.random.uniform(-limit, limit,
                              size=(in_channels, out_channels,
                                    self.kh, self.kw)).astype(np.float32),
            requires_grad=True)
        self.bias = Tensor(np.zeros((out_channels,), dtype=np.float32),
                           requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, Cin, H, W) -> (N, Cout, H_out, W_out)，H_out 精确等于公式值。"""
        data = x.data
        N, Cin, H, W = data.shape
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding
        kh, kw = self.kh, self.kw
        Cout = self.out_channels
        # 满尺寸再裁剪：full = (H-1)*s + k + op；实际输出 = full - 2*p。
        out_h = (H - 1) * sh - 2 * ph + kh + oph
        out_w = (W - 1) * sw - 2 * pw + kw + opw
        full_h = out_h + 2 * ph
        full_w = out_w + 2 * pw

        Wt = self.weight.data  # (Cin, Cout, kh, kw)

        # scatter 版前向（逐输入像素累加 kernel）
        full = np.zeros((N, Cout, full_h, full_w), dtype=np.float32)
        # 向量化：对每个 kernel 偏移 (i,j) 累加
        for i in range(kh):
            for j in range(kw):
                # src 有效输入像素 (h,w)，输出位置为 h*sh+i, w*sw+j
                # 由于 stride 可大于 1，输入索引到输出索引是映射；用 add.at 累加
                hs = np.arange(H)
                ws = np.arange(W)
                oh = hs * sh + i
                ow = ws * sw + j
                v = data[..., hs[:, None], ws[None, :]]            # (N,Cin,H,W)
                # v shape: (N, Cin, H, W) * weight[., :, i, j] (Cin, Cout)
                contrib = np.einsum("nchw,co->nohw", v,
                                    Wt[:, :, i, j])               # (N,Cout,H,W)
                np.add.at(full, (slice(None), slice(None),
                                 oh, ow[:, None]), contrib)
        # bias
        full = full + self.bias.data.reshape(1, Cout, 1, 1)
        # 裁剪掉每侧 ph/pw
        out_data = full[:, :, ph:ph + out_h, pw:pw + out_w]

        out = Tensor(out_data, requires_grad=True)
        self._store = (N, Cin, H, W, data.copy())
        parents = []
        if x.requires_grad:
            parents.append((x, lambda g: self._backward_input(g)))
        parents.append((self.weight, lambda g: self._backward_weight(g)))
        parents.append(
            (self.bias, lambda g: g.sum(axis=(0, 2, 3))))
        out._parents = tuple(parents)
        return out

    def __call__(self, x):
        return self.forward(x)

    def _full_grad(self, gout):
        """将裁剪后的输出梯度补零回满尺寸。"""
        N, Cout, out_h, out_w = gout.shape
        ph, pw = self.padding
        full = np.zeros((N, Cout, out_h + 2 * ph, out_w + 2 * pw),
                        dtype=np.float32)
        full[:, :, ph:ph + out_h, pw:pw + out_w] = gout
        return full

    def _backward_weight(self, gout):
        N, Cin, H, W, data = self._store
        sh, sw = self.stride
        kh, kw = self.kh, self.kw
        full = self._full_grad(gout)       # (N,Cout,full_h,full_w)
        grad_w = np.zeros_like(self.weight.data)
        for i in range(kh):
            for j in range(kw):
                oh = np.arange(H) * sh + i
                ow = np.arange(W) * sw + j
                v = full[:, :, oh, ow[:, None]]           # (N,Cout,H,W)
                d = np.einsum("nohw,nchw->co", v, data)   # (Cin,Cout)
                grad_w[:, :, i, j] = d
        return grad_w

    def _backward_input(self, gout):
        N, Cin, H, W, _ = self._store
        sh, sw = self.stride
        kh, kw = self.kh, self.kw
        full = self._full_grad(gout)       # (N,Cout,full_h,full_w)
        grad_in = np.zeros((N, Cin, H, W), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                oh = np.arange(H) * sh + i
                ow = np.arange(W) * sw + j
                v = full[:, :, oh, ow[:, None]]            # (N,Cout,H,W)
                d = np.einsum("nohw,co->nchw", v,
                              self.weight.data[:, :, i, j])  # (N,Cin,H,W)
                grad_in += d
        return grad_in


# ----------------------------------------------------------------------
# BatchNorm2D —— 批归一化。对 (N,C,H,W) 在 N、H、W 维上求均值/方差。
# ----------------------------------------------------------------------
class BatchNorm2D:
    def __init__(self, channels, momentum=0.9, eps=1e-5, affine=True):
        self.channels = channels
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.gamma = Tensor(np.ones((1, channels, 1, 1), dtype=np.float32),
                            requires_grad=affine)
        self.beta = Tensor(np.zeros((1, channels, 1, 1), dtype=np.float32),
                           requires_grad=affine)
        self.running_mean = np.zeros((channels,), dtype=np.float32)
        self.running_var = np.ones((channels,), dtype=np.float32)
        self.training = True

    def parameters(self):
        params = []
        if self.affine:
            params += [self.gamma, self.beta]
        return params

    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        N, C, H, W = data.shape
        if self.training:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
            var = data.var(axis=(0, 2, 3), keepdims=True)
            # 更新 running stats
            m = N * H * W
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * mean.reshape(-1))
            self.running_var = (self.momentum * self.running_var +
                                (1 - self.momentum) *
                                var.reshape(-1) * (m / max(m - 1, 1)))
        else:
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)
        xhat = (data - mean) / np.sqrt(var + self.eps)
        out = Tensor(xhat * self.gamma.data + self.beta.data,
                     requires_grad=self.requires_grad_value())
        self._store = (xhat, mean, var, data, N, H, W)
        parents = []
        if x.requires_grad:
            parents.append((x, lambda g: self._backward_input(g)))
        if self.affine:
            parents.append((self.gamma,
                            lambda g: (g * xhat).sum(axis=(0, 2, 3),
                                                     keepdims=True)))
            parents.append((self.beta,
                            lambda g: g.sum(axis=(0, 2, 3), keepdims=True)))
        out._parents = tuple(parents)
        return out

    def __call__(self, x):
        return self.forward(x)

    def requires_grad_value(self):
        return True

    def _backward_input(self, gout):
        if self.training:
            xhat, mean, var, data, N, H, W = self._store
            std = np.sqrt(var + self.eps)
            m = N * H * W
            # d xhat/d x
            dxhat = gout * self.gamma.data
            dvar = (dxhat * (data - mean) * -0.5 *
                    (var + self.eps) ** -1.5).sum(axis=(0, 2, 3), keepdims=True)
            dmean = (dxhat * -1.0 / std).sum(axis=(0, 2, 3), keepdims=True)
            dmean += dvar * ((data - mean) * -2.0 / m).sum(
                axis=(0, 2, 3), keepdims=True)
            dx = dxhat / std + dvar * 2.0 * (data - mean) / m + dmean / m
            return dx
        else:
            xhat, mean, var, data, N, H, W = self._store
            std = np.sqrt(var + self.eps)
            return gout * self.gamma.data / std


# ----------------------------------------------------------------------
# Flatten / reshape —— 桥接连结与卷积。
# ----------------------------------------------------------------------
def flatten(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1)


def reshape4d(x: Tensor, N, C, H, W) -> Tensor:
    return x.reshape(N, C, H, W)

