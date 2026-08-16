"""data.py — 数据加载与预处理。

需求：训练数据少、CPU 友好、输入为数字 -> 生成 100x100 图像。
这里使用公开手写数字数据集 MNIST（可先下载），取每类少量样本作为训练集，
并重采样到 100x100。如果找不到本地数据，也提供"合成/空跑"路径以便先跑通流程。

不使用任何 AI 封装库，仅依赖 numpy + 标准库。
"""
from __future__ import annotations

import os
import struct
import gzip
import urllib.request
import numpy as np

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
}


def _download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] 下载 {url} -> {path}")
    # 分块流式下载，带连接超时，避免在极慢/被墙源上无限挂起
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            f.flush()


def load_mnist(data_dir, download=True):
    """加载 MNIST 训练图像/标签（不下模型库，只是公开数据）。
    返回 (images np.ndarray (N,1,28,28) float32 值域[0,1], labels (N,) int)。
    """
    img_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    lbl_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
        if not download:
            return None, None
        try:
            _download(MNIST_URLS["train-images-idx3-ubyte.gz"], img_path)
            _download(MNIST_URLS["train-labels-idx1-ubyte.gz"], lbl_path)
        except Exception as e:  # 下载失败回退，由调用方处理
            print(f"[data] 下载 MNIST 失败：{e}")
            return None, None

    with gzip.open(img_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with gzip.open(lbl_path, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = imgs[:, None, :, :]  # (N,1,28,28)
    return imgs, labels.astype(np.int64)


def resize_nn(img, H, W):
    """最近邻重采样到 (H,W)。img: (..., h, w) -> (..., H, W)。"""
    h, w = img.shape[-2], img.shape[-1]
    ys = (np.arange(H) * h // H).astype(np.int64)
    xs = (np.arange(W) * w // W).astype(np.int64)
    return img[..., ys[:, None], xs[None, :]]


def per_class_subset(imgs, labels, per_class, rng=None):
    """从每个类别中取 per_class 张，打乱拼接，返回小数据集。"""
    rng = rng or np.random.RandomState(0)
    picked_x, picked_y = [], []
    for c in range(10):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        choose = rng.choice(len(idx), size=min(per_class, len(idx)),
                            replace=False)
        picked_x.append(imgs[idx[choose]])
        picked_y.append(labels[idx[choose]])
    X = np.concatenate(picked_x, axis=0)
    Y = np.concatenate(picked_y, axis=0)
    order = rng.permutation(len(X))
    return X[order], Y[order]


def make_dataset(data_dir, per_class=300, img_size=100, download=True,
                 seed=0):
    """加载 MNIST 并返回 (X (N,1,100,100) float32, Y (N,)) 小数据集。
    若网络不可用/无数据，回退为合成数据（空白 + 少量噪声），用于跑通流程。
    """
    rng = np.random.RandomState(seed)
    imgs, labels = load_mnist(data_dir, download=download)
    if imgs is None:
        print("[data] 未找到 MNIST，使用合成数据（仅用于流程验证）")
        return _synthetic(per_class=per_class, img_size=img_size, seed=seed)
    X, Y = per_class_subset(imgs, labels, per_class, rng)
    # 重采样到 100x100
    X = resize_nn(X, img_size, img_size)
    print(f"[data] 训练集 {X.shape[0]} 张，尺寸 {X.shape[2]}x{X.shape[3]}，"
          f"每类 {per_class}，类别 {sorted(set(Y.tolist()))}")
    return X.astype(np.float32), Y


def _synthetic(per_class=300, img_size=100, seed=0):
    """合成演示数据：在空白中央画一个粗 '0'，用于无外网时快速验证训练管线。"""
    rng = np.random.RandomState(seed)
    X = np.zeros((10 * per_class, 1, img_size, img_size), dtype=np.float32)
    Y = np.tile(np.arange(10), per_class)
    # 简单的类别条纹，让模型有类别信号可学
    c = int(img_size * 0.2)
    for k in range(10 * per_class):
        lab = Y[k]
        X[k, 0, lab * 8:(lab + 1) * 8, :] += 0.3
        X[k, 0, :, lab * 8:(lab + 1) * 8] += 0.3
    X += rng.normal(0, 0.02, X.shape).astype(np.float32)
    order = rng.permutation(len(X))
    return X[order], Y[order]


def iterate_batches(X, Y, batch_size, rng):
    N = len(X)
    idx = rng.permutation(N)
    for i in range(0, N, batch_size):
        b = idx[i:i + batch_size]
        if len(b) == 0:
            continue
        yield X[b], Y[b]
