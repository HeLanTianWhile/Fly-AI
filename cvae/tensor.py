"""tensor.py — 微型自研自动微分引擎（纯 NumPy，不使用任何 AI 封装库）。

这是整套系统的地基：一个最小但完整的标量/张量级反向传播引擎。
它模仿现代框架的“反向图”思想，但完全用手写 numpy 实现：
每一个 Tensor 记录自己是怎么算出来的（_parents），
再通过链式法则把梯度传回去。整个项目只用它，不依赖 torch 等库。
"""
from __future__ import annotations

import copy
import numpy as np


class Tensor:
    """带自动微分的最小张量。

    属性：
        data : np.ndarray            前向计算结果
        grad : np.ndarray | None     反向传播累计的梯度（形状与 data 相同）
        requires_grad : bool         是否需要梯度
    """

    __slots__ = ("data", "grad", "requires_grad", "_parents")

    def __init__(self, data, requires_grad=False, _parents=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = (
            np.zeros_like(self.data, dtype=np.float32)
            if requires_grad
            else None
        )
        self.requires_grad = bool(requires_grad)
        self._parents = _parents if _parents else ()

    # ------------------------------------------------------------------
    # 前向运算 + 反向传播
    # ------------------------------------------------------------------
    def backward(self, grad=None):
        """对该 Tensor 执行反向传播，把梯度回传到所有叶子节点。"""
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad.astype(np.float32)
        # 拓扑排序完成回传
        stack = [self]
        visited = set()
        order = []

        def _dfs(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p, _ in t._parents:
                _dfs(p)
            order.append(t)

        _dfs(self)
        # 反向传播顺序：后序遍历产出依赖在前，需逆序让损失先被处理、
        # 梯度沿计算图向后流动。
        for t in reversed(order):
            if t.grad is None:
                continue
            for p, local_grad_fn in t._parents:
                if p.requires_grad:
                    lg = np.asarray(local_grad_fn(t.grad), dtype=np.float32)
                    p.grad = p.grad + lg

    def zero_grad(self):
        self.grad = (
            np.zeros_like(self.data, dtype=np.float32)
            if self.requires_grad
            else None
        )

    # ---- 二元运算（自动补广播）----
    def _binop(self, other, op, local_p):
        has_tensor_other = isinstance(other, Tensor)
        other_num = other.data if has_tensor_other else np.asarray(
            other, dtype=np.float32)
        left = self
        if has_tensor_other and other.requires_grad:
            out_requires = left.requires_grad or other.requires_grad
        else:
            out_requires = left.requires_grad
        out_data = op(left.data, other_num)
        out = Tensor(out_data, requires_grad=out_requires)
        out._parents = (
            (
                (left, lambda g: local_p(left.data, other_num, g, 0)),
                (other, lambda g: local_p(left.data, other_num, g, 1)),
            )
            if has_tensor_other and other.requires_grad
            else (
                (left, lambda g: local_p(left.data, other_num, g, 0)),
            )
        )
        return out

    def __add__(self, other):
        return self._binop(
            other,
            lambda a, b: a + b,
            lambda a, b, g, idx: g if idx == 0 else _broadcast(g, b.shape),
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binop(
            other,
            lambda a, b: a - b,
            lambda a, b, g, idx: g if idx == 0 else _broadcast(
                g * -1.0, b.shape),
        )

    def __rsub__(self, other):
        return self._binop(
            other,
            lambda a, b: b - a,
            lambda a, b, g, idx: _broadcast(g * -1.0, a.shape)
            if idx == 0
            else g,
        )

    def __mul__(self, other):
        return self._binop(
            other,
            lambda a, b: a * b,
            lambda a, b, g, idx: _broadcast(g * (b if idx == 0 else a),
                                            a.shape)
            if idx == 0
            else _broadcast(g * a, b.shape),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binop(
            other,
            lambda a, b: a / b,
            lambda a, b, g, idx: _broadcast(g / b, a.shape)
            if idx == 0
            else _broadcast(g * (-a / (b * b)), b.shape),
        )

    __array_priority__ = 1000

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._parents = ((self, lambda g: -g),)
        return out

    def __pow__(self, n):
        n = float(n)
        out = Tensor(self.data ** n, requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: g * n * self.data ** (n - 1)),)
        return out

    # ---- 激活 / 数学 ----
    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0), requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: g * (self.data > 0)),)
        return out

    def silu(self):
        a = self.data
        sig = 1.0 / (1.0 + np.exp(-a))
        out = Tensor(a * sig, requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: g * (sig * (1.0 + a * (1.0 - sig)))),)
        return out

    def tanh(self):
        a = np.tanh(self.data)
        out = Tensor(a, requires_grad=self.requires_grad)
        out._parents = ((self, lambda g: g * (1.0 - a * a)),)
        return out

    def exp(self):
        a = np.exp(self.data)
        out = Tensor(a, requires_grad=self.requires_grad)
        out._parents = ((self, lambda g: g * a),)
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._parents = ((self, lambda g: g / self.data),)
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: np.broadcast_to(g, self.data.shape)),)
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: np.broadcast_to(g / n, self.data.shape)),)
        return out

    def reshape(self, *shape):
        target = shape if len(shape) > 1 or isinstance(
            shape[0], tuple) else shape
        t = tuple(target)
        out = Tensor(self.data.reshape(t), requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: g.reshape(self.data.shape)),)
        return out

    def transpose(self):
        out = Tensor(np.transpose(self.data, (2, 0, 1)),
                     requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: np.transpose(g, (2, 0, 1))),)
        return out

    def permute(self, axes):
        axes = tuple(axes)
        out = Tensor(np.transpose(self.data, axes),
                     requires_grad=self.requires_grad)
        out._parents = (
            (self, lambda g: np.transpose(g, np.argsort(axes))),)
        return out

    # ---- 属性 ----
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, idx):
        dtype = self.data.dtype
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)
        out._parents = ((self, lambda g: _scatter_like(self.data, idx, g)),)
        return out

    def __repr__(self):
        return (f"Tensor(shape={tuple(self.data.shape)}, "
                f"grad=... requires_grad={self.requires_grad})")


# ----------------------------------------------------------------------
# 反向梯度中用到的工具函数
# ----------------------------------------------------------------------
def _broadcast(grad, shape):
    """把梯度缩回目标形状（sum 掉被广播的维度）。"""
    grad = np.asarray(grad)
    # 从后往前为每个维度求和，直到与目标形状一致
    target = list(shape)
    while grad.ndim > len(target):
        grad = grad.sum(axis=0)
    for i in range(len(target)):
        if target[-1 - i] == 1 and grad.shape[-1 - i] != 1:
            grad = grad.sum(axis=grad.ndim - 1 - i, keepdims=True)
    return grad


def _reduce_to_shape(grad, shape):
    """把梯度规约到给定形状（用于常量/标量操作数）。"""
    grad = np.asarray(grad)
    if np.ndim(grad) == 0:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    return grad


def _scatter_like(src, idx, grad):
    """把切片索引的反向梯度散射回原形状的零数组。"""
    out = np.zeros_like(src, dtype=np.float32)
    np.add.at(out, idx, grad)
    return out


def cat(tensors, axis):
    """在给定轴上拼接一批 Tensor（支持反向传播）。"""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("cat: at least one tensor required")
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires)
    # 构造每个输入的反向切片
    slices = []
    cum = 0
    for t in tensors:
        s = [slice(None)] * data.ndim
        size = t.data.shape[axis]
        s[axis] = slice(cum, cum + size)
        slices.append(tuple(s))
        cum += size
    parents = []
    for t, s in zip(tensors, slices):
        if t.requires_grad:
            parents.append((t, lambda g, s_=s: g[s_]))
    out._parents = tuple(parents)
    return out


Tensor.concat = staticmethod(cat)
