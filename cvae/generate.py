"""generate.py — 生成接口：输入数字 -> 一张 100x100 手写数字风格图像。

可输出：
  - .png 文件（纯 zlib+struct 手写 PNG 编码，无 matplotlib 依赖）
  - .npy 原始数组
  - 终端 ASCII 预览
"""
from __future__ import annotations

import os
import struct
import zlib
import numpy as np

from .tensor import Tensor
from .model import CVAE


def generate(model, digit, rng=None, z_scale=0.6, n=1, seed=None):
    """为给定数字生成 n 张 (1,100,100) 图像，返回 np.ndarray (n,100,100)。"""
    if rng is None:
        rng = np.random.RandomState(seed if seed is not None else 0)
    model.train_mode(False)
    out = np.empty((n, 100, 100), dtype=np.float32)
    for i in range(n):
        g = model.generate(np.array([digit]), rng)
        out[i] = np.clip(g.data[0, 0], 0.0, 1.0)
    return out


def write_png(path, img):
    """把 (H,W) 灰度图写成 8 位灰阶 PNG。
    用 numpy 与标准库（zlib/struct）手工构造，不依赖任何图像/AI 库。
    """
    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    img8 = (np.round(img * 255.0)).astype(np.uint8)
    H, W = img8.shape

    def chunk(tag, payload):
        data = tag + payload
        return (struct.pack(">I", len(payload)) + data +
                struct.pack(">I", zlib.crc32(data) & 0xffffffff))

    ihdr = struct.pack(">IIBBBBB", W, H, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + row.tobytes() for row in img8)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def to_ascii(img, width=48, chars="@%#*+=-:. "):
    """把灰度图转成终端 ASCII 预览。"""
    img = np.clip(np.asarray(img, dtype=np.float32), 0, 1)
    H, W = img.shape
    ys = (np.arange(width) * H // width).astype(np.int64)
    xs = (np.arange(int(width * W / H)) * W // int(width * W / H)).astype(np.int64)
    small = img[np.ix_(ys, xs)]
    idx = (small * (len(chars) - 1)).astype(np.int32)
    return "\n".join("".join(chars[v] for v in row) for row in idx)
